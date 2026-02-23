#include "record_reader.h"

#include <cstring>
#include <iostream>
#include <limits>

namespace data {

RecordReader::RecordReader(RandomAccessFile* file,
                           const RecordReaderOptions& options)
    : file_(file), options_(options) {}

RecordReader::RecordReader(const std::string& filename,
                           const RecordReaderOptions& options)
    : file_(nullptr), options_(options) {
  auto file = std::make_unique<PosixRandomAccessFile>(filename);
  auto status = file->Open();
  if (!status.ok()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  owned_file_ = std::move(file);
  file_ = owned_file_.get();
}

RecordReader::~RecordReader() {
  if (z_stream_) {
    inflateEnd(z_stream_.get());
  }
}

absl::Status RecordReader::Init() {
  if (file_ == nullptr) {
    return absl::FailedPreconditionError("Reader file is null");
  }

  if (IsCompressed()) {
    auto status = InitZlib();
    if (!status.ok()) {
      return status;
    }
  }

  initialized_ = true;
  return absl::OkStatus();
}

absl::Status RecordReader::InitZlib() {
  const auto& zlib_opts = options_.zlib_options;

  input_buffer_capacity_ = zlib_opts.input_buffer_size;
  output_buffer_capacity_ = zlib_opts.output_buffer_size;

  z_stream_input_ = std::make_unique<uint8_t[]>(input_buffer_capacity_);
  z_stream_output_ = std::make_unique<uint8_t[]>(output_buffer_capacity_);
  z_stream_ = std::make_unique<z_stream>();

  memset(z_stream_.get(), 0, sizeof(z_stream));
  z_stream_->zalloc = Z_NULL;
  z_stream_->zfree = Z_NULL;
  z_stream_->opaque = Z_NULL;
  z_stream_->next_in = Z_NULL;
  z_stream_->avail_in = 0;

  int status = inflateInit2(z_stream_.get(), zlib_opts.window_bits);
  if (status != Z_OK) {
    z_stream_.reset();
    return absl::InvalidArgumentError("inflateInit2 failed with status " +
                                      std::to_string(status));
  }

  z_stream_->next_in = z_stream_input_.get();
  z_stream_->next_out = z_stream_output_.get();
  next_unread_byte_ = reinterpret_cast<char*>(z_stream_output_.get());
  z_stream_->avail_in = 0;
  z_stream_->avail_out = output_buffer_capacity_;

  return absl::OkStatus();
}

absl::Status RecordReader::ReadFromFile() {
  size_t bytes_to_read = input_buffer_capacity_;
  char* read_location = reinterpret_cast<char*>(z_stream_input_.get());

  // Move unread bytes to the head of the buffer.
  if (z_stream_->avail_in > 0) {
    size_t read_bytes = z_stream_->next_in - z_stream_input_.get();
    if (read_bytes > 0) {
      memmove(z_stream_input_.get(), z_stream_->next_in, z_stream_->avail_in);
    }
    bytes_to_read -= z_stream_->avail_in;
    read_location += z_stream_->avail_in;
  }

  size_t bytes_read = 0;
  auto status =
      file_->Read(file_offset_, bytes_to_read, read_location, &bytes_read);
  if (!status.ok() && !absl::IsNotFound(status)) {
    return status;
  }

  file_offset_ += bytes_read;
  z_stream_->next_in = z_stream_input_.get();
  z_stream_->avail_in += bytes_read;

  if (bytes_read == 0) {
    eof_reached_ = true;
    return absl::OutOfRangeError("EOF reached");
  }

  return absl::OkStatus();
}

size_t RecordReader::NumUnreadBytes() const {
  size_t read_bytes =
      next_unread_byte_ - reinterpret_cast<char*>(z_stream_output_.get());
  return output_buffer_capacity_ - z_stream_->avail_out - read_bytes;
}

size_t RecordReader::ReadBytesFromCache(size_t bytes_to_read,
                                        std::string* result) {
  size_t unread_bytes =
      reinterpret_cast<char*>(z_stream_->next_out) - next_unread_byte_;
  size_t can_read_bytes = std::min(bytes_to_read, unread_bytes);
  if (can_read_bytes > 0) {
    result->append(next_unread_byte_, can_read_bytes);
    next_unread_byte_ += can_read_bytes;
  }
  bytes_read_ += can_read_bytes;
  return can_read_bytes;
}

absl::Status RecordReader::Inflate() {
  int error = inflate(z_stream_.get(), options_.zlib_options.flush_mode);
  if (error != Z_OK && error != Z_STREAM_END && error != Z_BUF_ERROR) {
    std::string error_string =
        "inflate() failed with error " + std::to_string(error);
    if (z_stream_->msg != nullptr) {
      error_string += ": ";
      error_string += z_stream_->msg;
    }
    return absl::DataLossError(error_string);
  }
  // Handle concatenated gzip streams.
  if (error == Z_STREAM_END &&
      options_.zlib_options.window_bits == MAX_WBITS + 16) {
    inflateReset(z_stream_.get());
  }
  return absl::OkStatus();
}

absl::Status RecordReader::ReadNBytes(size_t n, std::string* result) {
  result->clear();

  if (!IsCompressed()) {
    // Direct file read for uncompressed files.
    result->resize(n);
    size_t bytes_read = 0;
    auto status = file_->Read(current_offset_, n, result->data(), &bytes_read);
    if (!status.ok()) {
      result->clear();
      return status;
    }
    if (bytes_read < n) {
      result->resize(bytes_read);
      if (bytes_read == 0) {
        return absl::OutOfRangeError("EOF");
      }
    }
    current_offset_ += bytes_read;
    return absl::OkStatus();
  }

  // Compressed read path.
  size_t bytes_to_read = n;
  bytes_to_read -= ReadBytesFromCache(bytes_to_read, result);

  while (bytes_to_read > 0) {
    // Reset output buffer.
    z_stream_->next_out = z_stream_output_.get();
    next_unread_byte_ = reinterpret_cast<char*>(z_stream_output_.get());
    z_stream_->avail_out = output_buffer_capacity_;

    // Try to inflate.
    auto status = Inflate();
    if (!status.ok()) return status;

    if (NumUnreadBytes() == 0) {
      // Need more input data.
      status = ReadFromFile();
      if (!status.ok()) {
        if (absl::IsOutOfRange(status) && result->size() > 0) {
          // Partial read at EOF.
          return absl::OkStatus();
        }
        return status;
      }
    } else {
      bytes_to_read -= ReadBytesFromCache(bytes_to_read, result);
    }
  }

  return absl::OkStatus();
}

absl::Status RecordReader::ReadChecksummed(uint64_t offset, size_t n,
                                           std::string* result) {
  if (n >= std::numeric_limits<size_t>::max() - sizeof(uint32_t)) {
    return absl::DataLossError("record size too large");
  }

  const size_t expected = n + sizeof(uint32_t);
  auto status = ReadNBytes(expected, result);
  if (!status.ok()) {
    return status;
  }

  if (result->size() != expected) {
    if (result->empty()) {
      return absl::OutOfRangeError("EOF");
    } else {
      return absl::DataLossError("truncated record at offset " +
                                 std::to_string(offset));
    }
  }

  const uint32_t masked_crc = DecodeFixed32(result->data() + n);
  if (crc32c::Unmask(masked_crc) != crc32c::Value(result->data(), n)) {
    return absl::DataLossError("corrupted record at offset " +
                               std::to_string(offset));
  }
  result->resize(n);
  return absl::OkStatus();
}

absl::Status RecordReader::ReadRecord(uint64_t* offset, std::string* record) {
  if (!initialized_) {
    return absl::FailedPreconditionError(
        "Reader not initialized. Call Init() first.");
  }

  // For uncompressed files, we can seek to the offset.
  if (!IsCompressed()) {
    current_offset_ = *offset;
  }

  // Read header (length + crc).
  std::string header_data;
  auto status = ReadChecksummed(*offset, sizeof(uint64_t), &header_data);
  if (!status.ok()) {
    return status;
  }
  const uint64_t length = DecodeFixed64(header_data.data());

  // Read data + crc.
  status = ReadChecksummed(*offset + kHeaderSize, length, record);
  if (!status.ok()) {
    if (absl::IsOutOfRange(status)) {
      return absl::DataLossError("truncated record at offset " +
                                 std::to_string(*offset));
    }
    return status;
  }

  *offset += kHeaderSize + length + kFooterSize;
  return absl::OkStatus();
}

// SequentialRecordReader implementation

SequentialRecordReader::SequentialRecordReader(
    RandomAccessFile* file, const RecordReaderOptions& options)
    : underlying_(file, options), offset_(0) {}

SequentialRecordReader::SequentialRecordReader(
    const std::string& filename, const RecordReaderOptions& options)
    : underlying_(filename, options), offset_(0) {}

absl::Status SequentialRecordReader::Init() { return underlying_.Init(); }

absl::Status SequentialRecordReader::ReadRecord(std::string* record) {
  return underlying_.ReadRecord(&offset_, record);
}

}  // namespace data
