#include "cc/shuffler/filename_buffer.h"

#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "cc/core/rand.h"

namespace shuffler {

FilenameBuffer::FilenameBuffer(const absl::flat_hash_set<std::string>& files) {
  std::vector<std::string> fv;  // for fast shuffling
  for (const auto& f : files) {
    fv.emplace_back(f);
  }

  std::shuffle(fv.begin(), fv.end(), core::PRng());
  std::move(fv.begin(), fv.end(), std::back_inserter(files_));
}

bool FilenameBuffer::empty() { return files_.empty(); }
int FilenameBuffer::size() { return files_.size(); }

void FilenameBuffer::AddNewFiles(std::vector<std::string> files) {
  if (files.empty()) {
    return;
  }

  // reshuffle entire deque
  std::vector<std::string> fv;
  std::move(files_.begin(), files_.end(), std::back_inserter(fv));
  files_.clear();

  for (const auto& f : files) {
    fv.emplace_back(f);
  }

  std::shuffle(fv.begin(), fv.end(), core::PRng());
  std::move(fv.begin(), fv.end(), std::back_inserter(files_));
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
