#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <zlib.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "coding.h"
#include "compression_options.h"
#include "crc32.h"
#include "file_interface.h"

namespace data {

// RecordReader reads TFRecord format files with optional decompression.
//
// TFRecord format:
//   uint64    length
//   uint32    masked_crc32c of length
//   byte      data[length]
//   uint32    masked_crc32c of data
class RecordReader {
 public:
  static constexpr size_t kHeaderSize = sizeof(uint64_t) + sizeof(uint32_t);
  static constexpr size_t kFooterSize = sizeof(uint32_t);

  // Create a reader from a RandomAccessFile.
  // "*file" must remain live while this Reader is in use.
  explicit RecordReader(RandomAccessFile* file,
                        const RecordReaderOptions& options = RecordReaderOptions());

  // Convenience constructor that opens a file.
  explicit RecordReader(const std::string& filename,
                        const RecordReaderOptions& options = RecordReaderOptions());

  ~RecordReader();

  // Initialize the reader. Must be called after construction.
  absl::Status Init();

  // Read the record at "*offset" into *record and update *offset to
  // point to the offset of the next record. Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  absl::Status ReadRecord(uint64_t* offset, std::string* record);

  // Check if the reader is using compression.
  bool IsCompressed() const {
    return options_.compression_type != CompressionType::NONE;
  }

 private:
  // Read n+4 bytes, verify checksum of first n bytes matches the last 4 bytes.
  absl::Status ReadChecksummed(uint64_t offset, size_t n, std::string* result);

  // Read raw bytes from the source (file or decompressed stream).
  absl::Status ReadNBytes(size_t n, std::string* result);

  // Initialize zlib for decompression.
  absl::Status InitZlib();

  // Read more compressed data from the file into the input buffer.
  absl::Status ReadFromFile();

  // Perform inflate operation.
  absl::Status Inflate();

  // Number of unread decompressed bytes in output buffer.
  size_t NumUnreadBytes() const;

  // Read bytes from the decompressed cache.
  size_t ReadBytesFromCache(size_t bytes_to_read, std::string* result);

  RandomAccessFile* file_;
  std::unique_ptr<RandomAccessFile> owned_file_;
  RecordReaderOptions options_;
  bool initialized_ = false;

  // Current read position in the underlying file (for compressed streams).
  uint64_t file_offset_ = 0;

  // For uncompressed reads: current logical offset.
  uint64_t current_offset_ = 0;

  // Zlib state (only used when compression is enabled)
  std::unique_ptr<z_stream> z_stream_;
  std::unique_ptr<uint8_t[]> z_stream_input_;
  std::unique_ptr<uint8_t[]> z_stream_output_;
  size_t input_buffer_capacity_ = 0;
  size_t output_buffer_capacity_ = 0;
  char* next_unread_byte_ = nullptr;
  int64_t bytes_read_ = 0;
  bool eof_reached_ = false;

  RecordReader(const RecordReader&) = delete;
  void operator=(const RecordReader&) = delete;
};

// High-level interface that maintains its own offset.
class SequentialRecordReader {
 public:
  explicit SequentialRecordReader(
      RandomAccessFile* file,
      const RecordReaderOptions& options = RecordReaderOptions());

  explicit SequentialRecordReader(
      const std::string& filename,
      const RecordReaderOptions& options = RecordReaderOptions());

  ~SequentialRecordReader() = default;

  // Initialize the reader.
  absl::Status Init();

  // Read the next record.
  absl::Status ReadRecord(std::string* record);

  // Return the current offset.
  uint64_t TellOffset() const { return offset_; }

 private:
  RecordReader underlying_;
  uint64_t offset_ = 0;
};

}  // namespace data
