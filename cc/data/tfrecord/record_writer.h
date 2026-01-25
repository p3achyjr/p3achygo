#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <zlib.h>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "coding.h"
#include "compression_options.h"
#include "crc32.h"
#include "file_interface.h"

namespace data {

// RecordWriter writes TFRecord format files with optional compression.
//
// TFRecord format:
//   uint64    length
//   uint32    masked_crc32c of length
//   byte      data[length]
//   uint32    masked_crc32c of data
//
// When compression is enabled, records are written to an internal buffer
// and compressed before being flushed to the underlying file.
class RecordWriter {
 public:
  static constexpr size_t kHeaderSize = sizeof(uint64_t) + sizeof(uint32_t);
  static constexpr size_t kFooterSize = sizeof(uint32_t);

  // Create a writer that will append data to "*dest".
  // "*dest" must remain live while this Writer is in use.
  // Takes ownership of dest if compression is enabled.
  explicit RecordWriter(WritableFile* dest,
                        const RecordWriterOptions& options = RecordWriterOptions());

  // Convenience constructor that opens a file.
  explicit RecordWriter(const std::string& filename,
                        const RecordWriterOptions& options = RecordWriterOptions());

  // Calls Close() and logs if an error occurs.
  ~RecordWriter();

  // Initialize the writer. Must be called after construction.
  absl::Status Init();

  // Write a record to the file.
  absl::Status WriteRecord(absl::string_view data);

  // Flushes any buffered data held by underlying containers.
  absl::Status Flush();

  // Writes all output to the file.
  // After calling Close(), any further calls to WriteRecord() or Flush()
  // are invalid.
  absl::Status Close();

  // Check if the writer is using compression.
  bool IsCompressed() const {
    return options_.compression_type != CompressionType::NONE;
  }

  // Utility method to populate TFRecord headers.
  static void PopulateHeader(char* header, const char* data, size_t n);

  // Utility method to populate TFRecord footers.
  static void PopulateFooter(char* footer, const char* data, size_t n);

 private:
  static uint32_t MaskedCrc(const char* data, size_t n) {
    return crc32c::Mask(crc32c::Value(data, n));
  }

  // Initialize zlib for compression.
  absl::Status InitZlib();

  // Deflate buffered data.
  absl::Status DeflateBuffered(int flush_mode);

  // Flush compressed output to file.
  absl::Status FlushOutputBufferToFile();

  // Perform deflate operation.
  absl::Status Deflate(int flush);

  // Add data to the input buffer.
  void AddToInputBuffer(absl::string_view data);

  // Available space in input buffer.
  int32_t AvailableInputSpace() const;

  // Append data, handling compression if enabled.
  absl::Status AppendImpl(absl::string_view data);

  WritableFile* dest_;
  std::unique_ptr<WritableFile> owned_file_;  // For filename constructor
  RecordWriterOptions options_;
  bool initialized_ = false;
  bool closed_ = false;

  // Zlib state (only used when compression is enabled)
  std::unique_ptr<z_stream> z_stream_;
  std::unique_ptr<uint8_t[]> z_stream_input_;
  std::unique_ptr<uint8_t[]> z_stream_output_;
  size_t input_buffer_capacity_ = 0;
  size_t output_buffer_capacity_ = 0;

  RecordWriter(const RecordWriter&) = delete;
  void operator=(const RecordWriter&) = delete;
};

inline void RecordWriter::PopulateHeader(char* header, const char* data,
                                         size_t n) {
  EncodeFixed64(header, n);
  EncodeFixed32(header + sizeof(uint64_t), MaskedCrc(header, sizeof(uint64_t)));
}

inline void RecordWriter::PopulateFooter(char* footer, const char* data,
                                         size_t n) {
  EncodeFixed32(footer, MaskedCrc(data, n));
}

}  // namespace data
