#ifndef __SHUFFLER_FILENAME_BUFFER_H_
#define __SHUFFLER_FILENAME_BUFFER_H_

#include <deque>
#include <string>
#include <optional>

#include "absl/container/flat_hash_set.h"

namespace shuffler {

/*
 * Manages a list of files, from which we return files to read from.
 */
class FilenameBuffer final {
 public:
  FilenameBuffer(const absl::flat_hash_set<std::string>& files);
  ~FilenameBuffer() = default;

  bool empty();
  void AddNewFiles(std::vector<std::string>&& files);
  std::optional<std::string> PopFile();

 private:
  std::deque<std::string> files_;
};
}  // namespace shuffler

#endif
