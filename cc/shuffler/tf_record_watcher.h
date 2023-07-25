#ifndef SHUFFLER_TF_RECORD_WATCHER_H_
#define SHUFFLER_TF_RECORD_WATCHER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "cc/shuffler/chunk_info.h"

namespace shuffler {

/*
 * Scans `dir` for changes, where `dir` is the parent data path for tfrecords
 * from all workers.
 *
 * The file structure will look like this:
 *
 * `dir`/:
 *   gen{g}_b0_g{G}_n{N}.tfrecord.zz
 *   gen{g}_b1_g{G}_n{N}.tfrecord.zz
 *   goldens/
 *
 * where each batch contains `G` games and `N` examples.
 */
class TfRecordWatcher final {
 public:
  TfRecordWatcher(std::string dir, int train_window_size);
  ~TfRecordWatcher() = default;

  // Disable Copy
  TfRecordWatcher(TfRecordWatcher const&) = delete;
  TfRecordWatcher& operator=(TfRecordWatcher const&) = delete;

  const absl::flat_hash_set<std::string>& GetFiles();
  std::vector<std::string> UpdateAndGetNew();
  int NumGamesSinceInit();

 private:
  absl::flat_hash_set<std::string> GlobFiles();
  void PopulateInitialTrainingWindow(int train_window_size);

  std::string dir_;
  absl::flat_hash_set<std::string> files_;
  absl::flat_hash_set<std::string> excluded_files_;
  int num_new_games_;
};
}  // namespace shuffler

#endif
