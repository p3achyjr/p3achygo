#include "cc/shuffler/tf_record_watcher.h"

#include <filesystem>  // make sure to compile with gcc 9+
#include <iterator>

#include "absl/container/flat_hash_set.h"
#include "cc/core/util.h"
#include "cc/shuffler/constants.h"

namespace shuffler {
namespace fs = std::filesystem;

using namespace ::core;

namespace {
static constexpr char kGenPrefix[] = "gen";
static constexpr char kTfRecordSuffix[] = ".tfrecord.zz";

// Find generation number from filepath. Assumes gen{i}/** format.
int FindGen(fs::path path) {
  for (const std::string& p : path) {
    if (p.find(kGenPrefix, 0) != std::string::npos) {
      return std::atoi(p.c_str() + 3);
    }
  }

  return -1;
}

fs::path RelativePath(fs::path base_path, fs::path path) {
  fs::path rel_path;
  auto path_it = path.begin();
  for (const auto& p : base_path) {
    if (p != *path_it) {
      return rel_path;
    }

    path_it = std::next(path_it);
  }

  for (auto it = path_it; it != path.end(); it = std::next(it)) {
    rel_path /= *it;
  }

  return rel_path;
}

}  // namespace

TfRecordWatcher::TfRecordWatcher(std::string dir, std::vector<int> exclude_gens)
    : dir_(dir), exclude_gens_(exclude_gens), files_(GlobFiles()) {}

const absl::flat_hash_set<std::string>& TfRecordWatcher::GetFiles() {
  return files_;
}

std::vector<std::string> TfRecordWatcher::UpdateAndGetNew() {
  absl::flat_hash_set<std::string> files = GlobFiles();
  std::vector<std::string> new_files;

  for (const auto& f : files) {
    if (!files_.contains(f)) {
      new_files.emplace_back(f);
    }
  }

  files_ = files;
  return new_files;
}

absl::flat_hash_set<std::string> TfRecordWatcher::GlobFiles() {
  auto should_include = [&](const fs::directory_entry& dir_entry) {
    fs::path rel_path = RelativePath(dir_, dir_entry);
    if (rel_path.empty()) {
      return false;
    }

    bool starts_with_gen_prefix =
        static_cast<std::string>(*rel_path.begin()).rfind(kDataGenPrefix, 0) ==
        0;
    if (!starts_with_gen_prefix) {
      return false;
    }

    if (VecContains(exclude_gens_, FindGen(rel_path))) {
      return false;
    }

    return true;
  };

  auto dir_it = fs::recursive_directory_iterator(dir_);
  absl::flat_hash_set<std::string> files;
  for (const auto& dir_entry : dir_it) {
    if (!dir_entry.is_regular_file()) {
      continue;
    }

    if (!should_include(dir_entry)) {
      continue;
    }

    files.insert(dir_entry.path());
  }

  return files;
}
}  // namespace shuffler
