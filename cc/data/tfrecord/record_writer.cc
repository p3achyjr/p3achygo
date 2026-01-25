#include "record_writer.h"

#include <cstring>
#include <iostream>

namespace data {
namespace {

bool IsSyncOrFullFlush(int flush_mode) {
  return flush_mode == Z_SYNC_FLUSH || flush_mode == Z_FULL_FLUSH;
}

}  // namespace

RecordWriter::RecordWriter(WritableFile* dest,
                           const RecordWriterOptions& options)
    : dest_(dest), options_(options) {}

RecordWriter::RecordWriter(const std::string& filename,
                           const RecordWriterOptions& options)
    : dest_(nullptr), options_(options) {
  auto file = std::make_unique<PosixWritableFile>(filename);
  auto status = file->Open();
  if (!status.ok()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  owned_file_ = std::move(file);
  dest_ = owned_file_.get();
}

RecordWriter::~RecordWriter() {
  if (!closed_ && dest_ != nullptr) {
    auto status = Close();
    if (!status.ok()) {
      std::cerr << "Could not finish writing file: " << status.message()
                << std::endl;
    }
  }
}

absl::Status RecordWriter::Init() {
  if (dest_ == nullptr) {
    return absl::FailedPreconditionError("Writer destination is null");
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

absl::Status RecordWriter::InitZlib() {
  const auto& zlib_opts = options_.zlib_options;

  input_buffer_capacity_ = zlib_opts.input_buffer_size;
  output_buffer_capacity_ = zlib_opts.output_buffer_size;

  if (output_buffer_capacity_ <= 1) {
    return absl::InvalidArgumentError(
        "output_buffer_size should be greater than 1");
  }

  z_stream_input_ = std::make_unique<uint8_t[]>(input_buffer_capacity_);
  z_stream_output_ = std::make_unique<uint8_t[]>(output_buffer_capacity_);
  z_stream_ = std::make_unique<z_stream>();

  memset(z_stream_.get(), 0, sizeof(z_stream));
  z_stream_->zalloc = Z_NULL;
  z_stream_->zfree = Z_NULL;
  z_stream_->opaque = Z_NULL;

  int status =
      deflateInit2(z_stream_.get(), zlib_opts.compression_level,
                   zlib_opts.compression_method, zlib_opts.window_bits,
                   zlib_opts.mem_level, zlib_opts.compression_strategy);
  if (status != Z_OK) {
    z_stream_.reset();
    return absl::InvalidArgumentError("deflateInit2 failed with status " +
                                      std::to_string(status));
  }

  z_stream_->next_in = z_stream_input_.get();
  z_stream_->next_out = z_stream_output_.get();
  z_stream_->avail_in = 0;
  z_stream_->avail_out = output_buffer_capacity_;

  return absl::OkStatus();
}

int32_t RecordWriter::AvailableInputSpace() const {
  return input_buffer_capacity_ - z_stream_->avail_in;
}

void RecordWriter::AddToInputBuffer(absl::string_view data) {
  size_t bytes_to_write = data.size();

  int32_t read_bytes = z_stream_->next_in - z_stream_input_.get();
  int32_t unread_bytes = z_stream_->avail_in;
  int32_t free_tail_bytes =
      input_buffer_capacity_ - (read_bytes + unread_bytes);

  if (static_cast<int32_t>(bytes_to_write) > free_tail_bytes) {
    memmove(z_stream_input_.get(), z_stream_->next_in, z_stream_->avail_in);
    z_stream_->next_in = z_stream_input_.get();
  }
  memcpy(z_stream_->next_in + z_stream_->avail_in, data.data(), bytes_to_write);
  z_stream_->avail_in += bytes_to_write;
}

absl::Status RecordWriter::DeflateBuffered(int flush_mode) {
  do {
    if (z_stream_->avail_out == 0 ||
        (IsSyncOrFullFlush(flush_mode) && z_stream_->avail_out < 6)) {
      auto status = FlushOutputBufferToFile();
      if (!status.ok()) return status;
    }
    auto status = Deflate(flush_mode);
    if (!status.ok()) return status;
  } while (z_stream_->avail_out == 0);

  z_stream_->next_in = z_stream_input_.get();
  return absl::OkStatus();
}

absl::Status RecordWriter::FlushOutputBufferToFile() {
  size_t bytes_to_write = output_buffer_capacity_ - z_stream_->avail_out;
  if (bytes_to_write > 0) {
    auto status = dest_->Append(absl::string_view(
        reinterpret_cast<char*>(z_stream_output_.get()), bytes_to_write));
    if (status.ok()) {
      z_stream_->next_out = z_stream_output_.get();
      z_stream_->avail_out = output_buffer_capacity_;
    }
    return status;
  }
  return absl::OkStatus();
}

absl::Status RecordWriter::Deflate(int flush) {
  int error = deflate(z_stream_.get(), flush);
  if (error == Z_OK || error == Z_BUF_ERROR ||
      (error == Z_STREAM_END && flush == Z_FINISH)) {
    return absl::OkStatus();
  }
  std::string error_string =
      "deflate() failed with error " + std::to_string(error);
  if (z_stream_->msg != nullptr) {
    error_string += ": ";
    error_string += z_stream_->msg;
  }
  return absl::DataLossError(error_string);
}

absl::Status RecordWriter::AppendImpl(absl::string_view data) {
  if (!IsCompressed()) {
    return dest_->Append(data);
  }

  size_t bytes_to_write = data.size();

  if (static_cast<int32_t>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return absl::OkStatus();
  }

  auto status = DeflateBuffered(options_.zlib_options.flush_mode);
  if (!status.ok()) return status;

  if (static_cast<int32_t>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return absl::OkStatus();
  }

  // Data is too large to fit in input buffer, deflate directly.
  z_stream_->next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
  z_stream_->avail_in = bytes_to_write;

  do {
    if (z_stream_->avail_out == 0) {
      status = FlushOutputBufferToFile();
      if (!status.ok()) return status;
    }
    status = Deflate(options_.zlib_options.flush_mode);
    if (!status.ok()) return status;
  } while (z_stream_->avail_out == 0);

  z_stream_->next_in = z_stream_input_.get();
  return absl::OkStatus();
}

absl::Status RecordWriter::WriteRecord(absl::string_view data) {
  if (dest_ == nullptr) {
    return absl::FailedPreconditionError(
        "Writer not initialized or previously closed");
  }
  if (!initialized_) {
    return absl::FailedPreconditionError(
        "Writer not initialized. Call Init() first.");
  }
  if (closed_) {
    return absl::FailedPreconditionError("Writer already closed");
  }

  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  char header[kHeaderSize];
  char footer[kFooterSize];
  PopulateHeader(header, data.data(), data.size());
  PopulateFooter(footer, data.data(), data.size());

  auto status = AppendImpl(absl::string_view(header, kHeaderSize));
  if (!status.ok()) return status;

  status = AppendImpl(data);
  if (!status.ok()) return status;

  return AppendImpl(absl::string_view(footer, kFooterSize));
}

absl::Status RecordWriter::Close() {
  if (closed_) return absl::OkStatus();
  if (dest_ == nullptr) return absl::OkStatus();

  absl::Status status = absl::OkStatus();

  if (IsCompressed() && z_stream_) {
    status = DeflateBuffered(Z_FINISH);
    if (status.ok()) {
      status = FlushOutputBufferToFile();
    }
    deflateEnd(z_stream_.get());
    z_stream_.reset();
  }

  if (status.ok()) {
    status = dest_->Close();
  }

  closed_ = true;
  return status;
}

absl::Status RecordWriter::Flush() {
  if (dest_ == nullptr) {
    return absl::FailedPreconditionError(
        "Writer not initialized or previously closed");
  }

  if (IsCompressed() && z_stream_) {
    auto status = DeflateBuffered(Z_PARTIAL_FLUSH);
    if (!status.ok()) return status;
    status = FlushOutputBufferToFile();
    if (!status.ok()) return status;
  }

  return dest_->Flush();
}

}  // namespace data
