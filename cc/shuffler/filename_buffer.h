#ifndef __SHUFFLER_FILENAME_BUFFER_H_
#define __SHUFFLER_FILENAME_BUFFER_H_

#include <deque>
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"

namespace shuffler {

/*
 * Manages a list of files, from which we return files to read from.
 *
 * Not thread safe.
 */
class FilenameBuffer final {
 public:
  FilenameBuffer(const absl::flat_hash_set<std::string>& files);
  ~FilenameBuffer() = default;

  // Disable Copy
  FilenameBuffer(FilenameBuffer const&) = delete;
  FilenameBuffer& operator=(FilenameBuffer const&) = delete;

  bool empty();
  int size();
  void AddNewFiles(std::vector<std::string> files);
  std::optional<std::string> PopFile();

 private:
  std::deque<std::string> files_;
};
}  // namespace shuffler

#endif
