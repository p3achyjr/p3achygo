#pragma once

#include <cstdio>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace data {

// Abstract interface for a writable file.
class WritableFile {
 public:
  virtual ~WritableFile() = default;

  // Append data to the file.
  virtual absl::Status Append(absl::string_view data) = 0;

  // Flush buffered data to the underlying storage.
  virtual absl::Status Flush() = 0;

  // Close the file. No further operations are valid after Close().
  virtual absl::Status Close() = 0;

  // Sync data to stable storage.
  virtual absl::Status Sync() = 0;
};

// Abstract interface for a randomly accessible file.
class RandomAccessFile {
 public:
  virtual ~RandomAccessFile() = default;

  // Read up to `n` bytes starting at `offset` into `result`.
  // Returns the number of bytes actually read in `bytes_read`.
  virtual absl::Status Read(uint64_t offset, size_t n, char* result,
                            size_t* bytes_read) = 0;
};

// Simple implementation of WritableFile using FILE*.
class PosixWritableFile : public WritableFile {
 public:
  explicit PosixWritableFile(const std::string& filename)
      : filename_(filename), file_(nullptr) {}

  ~PosixWritableFile() override {
    if (file_ != nullptr) {
      fclose(file_);
    }
  }

  absl::Status Open() {
    file_ = fopen(filename_.c_str(), "wb");
    if (file_ == nullptr) {
      return absl::NotFoundError("Failed to open file: " + filename_);
    }
    return absl::OkStatus();
  }

  absl::Status Append(absl::string_view data) override {
    if (file_ == nullptr) {
      return absl::FailedPreconditionError("File not open");
    }
    size_t written = fwrite(data.data(), 1, data.size(), file_);
    if (written != data.size()) {
      return absl::InternalError("Failed to write to file");
    }
    return absl::OkStatus();
  }

  absl::Status Flush() override {
    if (file_ == nullptr) {
      return absl::FailedPreconditionError("File not open");
    }
    if (fflush(file_) != 0) {
      return absl::InternalError("Failed to flush file");
    }
    return absl::OkStatus();
  }

  absl::Status Close() override {
    if (file_ == nullptr) {
      return absl::OkStatus();
    }
    if (fclose(file_) != 0) {
      return absl::InternalError("Failed to close file");
    }
    file_ = nullptr;
    return absl::OkStatus();
  }

  absl::Status Sync() override {
    auto status = Flush();
    if (!status.ok()) return status;
    // Note: For true durability, you'd use fsync() here.
    return absl::OkStatus();
  }

 private:
  std::string filename_;
  FILE* file_;
};

// Simple implementation of RandomAccessFile using FILE*.
class PosixRandomAccessFile : public RandomAccessFile {
 public:
  explicit PosixRandomAccessFile(const std::string& filename)
      : filename_(filename), file_(nullptr) {}

  ~PosixRandomAccessFile() override {
    if (file_ != nullptr) {
      fclose(file_);
    }
  }

  absl::Status Open() {
    file_ = fopen(filename_.c_str(), "rb");
    if (file_ == nullptr) {
      return absl::NotFoundError("Failed to open file: " + filename_);
    }
    return absl::OkStatus();
  }

  absl::Status Read(uint64_t offset, size_t n, char* result,
                    size_t* bytes_read) override {
    if (file_ == nullptr) {
      return absl::FailedPreconditionError("File not open");
    }
    if (fseek(file_, static_cast<long>(offset), SEEK_SET) != 0) {
      return absl::InternalError("Failed to seek in file");
    }
    *bytes_read = fread(result, 1, n, file_);
    if (*bytes_read < n && ferror(file_)) {
      return absl::InternalError("Failed to read from file");
    }
    return absl::OkStatus();
  }

 private:
  std::string filename_;
  FILE* file_;
};

}  // namespace data
