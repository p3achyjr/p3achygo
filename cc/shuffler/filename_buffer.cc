#include "cc/shuffler/filename_buffer.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"

namespace shuffler {

FilenameBuffer::FilenameBuffer(const absl::flat_hash_set<std::string>& files) {
  for (const auto& f : files) {
    files_.emplace_back(f);
  }
}

bool FilenameBuffer::empty() { return files_.empty(); }

void FilenameBuffer::AddNewFiles(std::vector<std::string>&& files) {
  for (const auto& f : files) {
    files_.emplace_back(f);
  }
}

std::optional<std::string> FilenameBuffer::PopFile() {
  if (files_.empty()) {
    return std::nullopt;
  }

  std::string f = files_.front();
  files_.pop_front();
  return f;
}

}  // namespace shuffler
