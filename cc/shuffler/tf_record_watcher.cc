#include "cc/shuffler/tf_record_watcher.h"

#include <filesystem>  // make sure to compile with gcc 9+
#include <iterator>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/core/util.h"
#include "cc/shuffler/constants.h"

namespace shuffler {
namespace fs = std::filesystem;

using namespace ::core;

TfRecordWatcher::TfRecordWatcher(std::string dir, int train_window_size)
    : dir_(dir),
      files_(PopulateInitialTrainingWindow(train_window_size)),
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
      num_new_games_ += chunk_info->num_games;
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
    if (!chunk_info) {
      continue;
    }

    files.insert(dir_entry.path());
  }

  return files;
}

absl::flat_hash_set<std::string> TfRecordWatcher::PopulateInitialTrainingWindow(
    int train_window_size) {
  struct ChunkData {
    std::string filename;
    ChunkInfo info;
  };
  absl::flat_hash_set<std::string> files = GlobFiles();
  std::vector<ChunkData> file_data;
  for (const auto& file : files) {
    std::optional<ChunkInfo> chunk_info =
        ParseChunkFilename(fs::path(file).filename());
    CHECK(chunk_info);
    file_data.emplace_back(ChunkData{file, chunk_info.value()});
  }

  // reverse sort.
  std::sort(file_data.begin(), file_data.end(),
            [](const ChunkData& f0, const ChunkData& f1) {
              return f0.info.timestamp > f1.info.timestamp;
            });

  // iterate through all files, adding each file to a buffer until we consume
  // our `train_window_size`.
  int window_size = 0;
  absl::flat_hash_set<std::string> filtered_files;
  for (const auto& data : file_data) {
    if (window_size >= train_window_size) {
      break;
    }

    filtered_files.insert(data.filename);
    window_size += data.info.num_examples;
  }

  LOG(INFO) << "Total Number of Files: " << files.size()
            << ". Number of Files in Training Window: "
            << filtered_files.size();
  return filtered_files;
}
}  // namespace shuffler
