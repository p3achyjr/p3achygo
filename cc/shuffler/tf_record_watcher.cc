#include "cc/shuffler/tf_record_watcher.h"

#include <filesystem>  // make sure to compile with gcc 9+
#include <iterator>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "cc/core/util.h"
#include "cc/shuffler/constants.h"

namespace shuffler {
namespace fs = std::filesystem;

using namespace ::core;

TfRecordWatcher::TfRecordWatcher(std::string dir, std::vector<int> exclude_gens)
    : dir_(dir),
      exclude_gens_(exclude_gens),
      files_(GlobFiles()),
      num_new_games_(0) {}

const absl::flat_hash_set<std::string>& TfRecordWatcher::GetFiles() {
  return files_;
}

std::vector<std::string> TfRecordWatcher::UpdateAndGetNew() {
  absl::flat_hash_set<std::string> files = GlobFiles();
  std::vector<std::string> new_files;

  for (const auto& f : files) {
    if (!files_.contains(f)) {
      new_files.emplace_back(f);
      std::optional<ChunkInfo> chunk_info =
          ParseChunkFilename(fs::path(f).filename());
      CHECK(chunk_info);
      num_new_games_ += chunk_info->games;
    }
  }

  files_ = files;
  return new_files;
}

int TfRecordWatcher::NumGamesSinceInit() { return num_new_games_; }

absl::flat_hash_set<std::string> TfRecordWatcher::GlobFiles() {
  auto dir_it = fs::recursive_directory_iterator(dir_);
  absl::flat_hash_set<std::string> files;
  for (const auto& dir_entry : dir_it) {
    if (!dir_entry.is_regular_file()) {
      continue;
    }

    std::optional<ChunkInfo> chunk_info =
        ParseChunkFilename(dir_entry.path().filename());
    if (!chunk_info || VecContains(exclude_gens_, chunk_info->gen)) {
      continue;
    }

    files.insert(dir_entry.path());
  }

  return files;
}
}  // namespace shuffler
