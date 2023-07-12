#include "cc/data/coordinator.h"

#include <chrono>
#include <filesystem>
#include <thread>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "cc/data/worker.h"
#include "indicators/progress_bar.hpp"

namespace data {
namespace fs = std::filesystem;

using namespace ::indicators;

Coordinator::Coordinator(int num_workers, std::string sgf_dir,
                         std::string out_dir, bool is_dry_run)
    : num_workers_(num_workers),
      is_dry_run_(is_dry_run),
      out_dir_(out_dir),
      finished_(false),
      num_files_(0),
      num_files_delegated_(0),
      num_completions_(0),
      num_errors_(0),
      shard_num_(0) {
  std::vector<std::string> files;
  for (const auto& entry : fs::recursive_directory_iterator(sgf_dir)) {
    if (!(entry.is_regular_file() && entry.path().extension() == ".sgf")) {
      continue;
    }

    files.emplace_back(entry.path());
    num_files_++;
  }

  file_buffer_.AddNewFiles(files);
}

void Coordinator::Run() {
  std::thread heartbeat_thread(&Coordinator::Heartbeat, this);
  std::vector<std::thread> worker_pool;
  for (int worker_id = 0; worker_id < num_workers_; ++worker_id) {
    worker_pool.emplace_back(
        std::thread(Worker, worker_id, this, out_dir_, is_dry_run_));
  }

  for (auto& worker : worker_pool) {
    worker.join();
  }

  finished_.store(true, std::memory_order_release);
  heartbeat_thread.join();

  LOG(INFO) << "Total Num Examples: " << num_examples_;

  std::string length_file = fs::path(out_dir_) / "LENGTH.txt";
  FILE* const file = fopen(length_file.c_str(), "w");
  absl::FPrintF(file, "%d", num_examples_);
  fclose(file);
}

std::optional<std::string> Coordinator::GetFile() {
  absl::MutexLock l(&mu_);

  std::optional<std::string> file = file_buffer_.PopFile();
  if (file) ++num_files_delegated_;
  return file;
}

void Coordinator::MarkDone(int num_examples) {
  absl::MutexLock l(&mu_);
  ++num_completions_;
  num_examples_ += num_examples;
}

void Coordinator::MarkError() {
  absl::MutexLock l(&mu_);
  ++num_errors_;
}

int Coordinator::GetShardNum() {
  absl::MutexLock l(&mu_);
  ++shard_num_;

  return shard_num_ - 1;
}

void Coordinator::Heartbeat() {
  ProgressBar bar{option::BarWidth{50},
                  option::Start{"["},
                  option::Fill{"="},
                  option::Lead{">"},
                  option::Remainder{" "},
                  option::End{"]"},
                  option::PostfixText{
                      absl::StrFormat("Files Processed: 0/%d. Num Examples: %d",
                                      num_files_, num_examples_)},
                  option::ForegroundColor{Color::green},
                  option::ShowElapsedTime{true},
                  option::ShowRemainingTime{true},
                  option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};
  while (!IsFinished()) {
    mu_.LockWhenWithTimeout(absl::Condition(this, &Coordinator::IsFinished),
                            absl::Seconds(.5));
    int progress = (static_cast<float>(num_completions_) /
                    static_cast<float>(num_files_)) *
                   100;
    bar.set_progress(progress);
    bar.set_option(option::PostfixText{absl::StrFormat(
        "Files Processed: %d/%d. Num Examples: %d",
        num_completions_ + num_errors_, num_files_, num_examples_)});
    mu_.Unlock();
  }
}

bool Coordinator::IsFinished() {
  return finished_.load(std::memory_order_acquire);
}

}  // namespace data
