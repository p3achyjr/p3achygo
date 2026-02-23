#pragma once

#include <zlib.h>

#include <cstdint>

namespace data {

// Zlib compression options mirroring TensorFlow's ZlibCompressionOptions.
struct ZlibCompressionOptions {
  // Defaults to Z_NO_FLUSH
  int8_t flush_mode = Z_NO_FLUSH;

  // Size of the buffer used for caching input data.
  int64_t input_buffer_size = 256 << 10;  // 256 KB

  // Size of the buffer for compressed/decompressed output.
  int64_t output_buffer_size = 256 << 10;  // 256 KB

  // Window size for compression. See zlib manual for details.
  // 8..15: Normal deflate with zlib header
  // 16+[8..15]: Gzip format
  int8_t window_bits = MAX_WBITS;  // 15

  // Compression level: 0 (none) to 9 (best), default 6.
  int8_t compression_level = Z_DEFAULT_COMPRESSION;

  // Only Z_DEFLATED is supported.
  int8_t compression_method = Z_DEFLATED;

  // Memory level: 1 (minimum) to 9 (maximum). Higher = faster but more memory.
  int8_t mem_level = 9;

  // Compression strategy.
  int8_t compression_strategy = Z_DEFAULT_STRATEGY;

  // Factory method for default configuration.
  static ZlibCompressionOptions DEFAULT() { return ZlibCompressionOptions(); }
};

// Compression type enum.
enum class CompressionType { NONE = 0, ZLIB = 1 };

// Options for RecordWriter.
struct RecordWriterOptions {
  CompressionType compression_type = CompressionType::NONE;
  ZlibCompressionOptions zlib_options;

  static RecordWriterOptions Default() { return RecordWriterOptions(); }

  static RecordWriterOptions Zlib() {
    RecordWriterOptions options;
    options.compression_type = CompressionType::ZLIB;
    options.zlib_options = ZlibCompressionOptions::DEFAULT();
    return options;
  }
};

// Options for RecordReader.
struct RecordReaderOptions {
  CompressionType compression_type = CompressionType::NONE;
  ZlibCompressionOptions zlib_options;

  static RecordReaderOptions Default() { return RecordReaderOptions(); }

  static RecordReaderOptions Zlib() {
    RecordReaderOptions options;
    options.compression_type = CompressionType::ZLIB;
    options.zlib_options = ZlibCompressionOptions::DEFAULT();
    return options;
  }
};

}  // namespace data
