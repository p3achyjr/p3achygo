#ifndef DATA_COORDINATOR_H_
#define DATA_COORDINATOR_H_

#include <atomic>

#include "absl/synchronization/mutex.h"
#include "cc/shuffler/filename_buffer.h"

namespace data {

class Coordinator final {
 public:
  Coordinator(int num_workers, std::string sgf_dir, std::string out_dir,
              bool is_dry_run);
  ~Coordinator() = default;

  // Disable Copy and Move.
  Coordinator(Coordinator const&) = delete;
  Coordinator& operator=(Coordinator const&) = delete;
  Coordinator(Coordinator&&) = delete;
  Coordinator& operator=(Coordinator&&) = delete;

  void Run();
  std::optional<std::string> GetFile() ABSL_LOCKS_EXCLUDED(mu_);
  void MarkDone(int num_examples) ABSL_LOCKS_EXCLUDED(mu_);
  void MarkError() ABSL_LOCKS_EXCLUDED(mu_);
  int GetShardNum() ABSL_LOCKS_EXCLUDED(mu_);

 private:
  void Heartbeat();
  bool IsFinished();

  const int num_workers_;
  const bool is_dry_run_;
  const std::string out_dir_;
  std::atomic<bool> finished_;

  int num_files_;
  int num_files_delegated_ ABSL_GUARDED_BY(mu_);
  int num_completions_ ABSL_GUARDED_BY(mu_);
  int num_errors_ ABSL_GUARDED_BY(mu_);
  int num_examples_ ABSL_GUARDED_BY(mu_);
  int shard_num_ ABSL_GUARDED_BY(mu_);

  shuffler::FilenameBuffer file_buffer_ ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
};

}  // namespace data

#endif
