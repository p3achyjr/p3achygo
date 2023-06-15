#ifndef __SHUFFLER_TF_RECORD_WATCHER_H_
#define __SHUFFLER_TF_RECORD_WATCHER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"

namespace shuffler {

/*
 * Scans `dir` for changes, where `dir` is the parent data path for tfrecords
 * from all workers.
 *
 * The file structure will look like this:
 *
 * `dir`/:
 *   batch_b0_g{G}_n{N}.tfrecord.zz
 *   batch_b1_g{G}_n{N}.tfrecord.zz
 *   goldens/
 *
 * where each batch contains `G` games and `N` examples.
 */
class TfRecordWatcher final {
 public:
  TfRecordWatcher(std::string dir, std::vector<int> exclude_gens);
  ~TfRecordWatcher() = default;

  // Disable Copy
  TfRecordWatcher(TfRecordWatcher const&) = delete;
  TfRecordWatcher& operator=(TfRecordWatcher const&) = delete;

  const absl::flat_hash_set<std::string>& GetFiles();
  std::vector<std::string> UpdateAndGetNew();

 private:
  absl::flat_hash_set<std::string> GlobFiles();
  std::string dir_;
  std::vector<int> exclude_gens_;
  absl::flat_hash_set<std::string> files_;
};
}  // namespace shuffler

#endif
