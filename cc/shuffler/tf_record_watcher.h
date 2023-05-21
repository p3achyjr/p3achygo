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
 *   gen0/
 *   gen1/
 *   gen2/
 *     mach0/
 *       batch_0_g0_{b}.tfrecord.zz
 *       batch_1_g{b}_{2b}.tfrecord.zz
 *       ...
 *     mach1/
 *     ...
 *     mach{m}/
 *   ...
 *   gen{n}/
 *
 * where there are `n` generations, `m` machines, and `b` games per batch.
 */
class TfRecordWatcher final {
 public:
  TfRecordWatcher(std::string dir, std::vector<int> exclude_gens);
  ~TfRecordWatcher() = default;

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
