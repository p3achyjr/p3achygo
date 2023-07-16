#ifndef CORE_FILEPATH_H_
#define CORE_FILEPATH_H_

#include <string>

#include "absl/container/inlined_vector.h"

namespace core {

/*
 * Filepath?
 *
 * I just needed this because gcc refuses to behave :')
 */
class FilePath final {
 public:
  FilePath(const std::string& path) {
    auto last = path.find_last_not_of(kPathSeparator);
    path_ = path.substr(0, last + 1);
  }

  FilePath(std::string&& path) : FilePath(path) {}
  ~FilePath() = default;

  const char* c_str() const { return path_.c_str(); }

  FilePath& operator=(const std::string& s) {
    path_ = s;
    return *this;
  }

  FilePath& operator=(std::string&& s) {
    path_ = s;
    return *this;
  }

  friend FilePath operator/(const FilePath& lhs, const FilePath& rhs) {
    return FilePath(lhs.path_ + kPathSeparator + rhs.path_);
  }

  friend FilePath operator/(const FilePath& lhs, const std::string& rhs) {
    return FilePath(lhs.path_ + kPathSeparator + rhs);
  }

  friend FilePath operator/(const FilePath& lhs, std::string&& rhs) {
    return FilePath(lhs.path_ + kPathSeparator + rhs);
  }

  operator std::string() { return path_; }
  friend bool operator==(const FilePath& lhs, const FilePath& rhs) {
    return lhs.path_ == rhs.path_;
  }

  friend bool operator==(const FilePath& lhs, const std::string& rhs) {
    return lhs.path_ == rhs;
  }

  friend bool operator==(const FilePath& lhs, std::string&& rhs) {
    return lhs.path_ == rhs;
  }

 private:
  static constexpr char kPathSeparator = '/';
  std::string path_;
};
}  // namespace core

#endif
