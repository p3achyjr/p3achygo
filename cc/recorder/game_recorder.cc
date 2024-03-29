#include "cc/recorder/game_recorder.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "absl/log/log.h"
#include "cc/core/filepath.h"
#include "cc/game/game.h"
#include "cc/recorder/dir.h"

namespace recorder {
namespace {

using ::core::FilePath;

static constexpr char kP3achyGoName[] = "p3achygo";

class GameRecorderImpl final : public GameRecorder {
 public:
  GameRecorderImpl(std::string path, int num_threads, int flush_interval,
                   int gen, std::string worker_id);
  ~GameRecorderImpl();

  // Disable Copy and Move.
  GameRecorderImpl(GameRecorderImpl const&) = delete;
  GameRecorderImpl& operator=(GameRecorderImpl const&) = delete;
  GameRecorderImpl(GameRecorderImpl&&) = delete;
  GameRecorderImpl& operator=(GameRecorderImpl&&) = delete;

  void RecordGame(
      int thread_id, const game::Board& init_board, const game::Game& game,
      const ImprovedPolicies& mcts_pis,
      const std::vector<uint8_t>& move_trainables,
      const std::vector<float>& root_qs,
      std::vector<std::unique_ptr<mcts::TreeNode>>&& roots) override;

  void RecordEvalGame(int thread_id, const game::Game& game,
                      const std::string& b_name,
                      const std::string& w_name) override;

  static std::unique_ptr<GameRecorderImpl> Create(std::string path,
                                                  int num_threads,
                                                  int flush_interval,
                                                  std::string worker_id);

 private:
  void IoThread() ABSL_LOCKS_EXCLUDED(mu_);
  void Flush() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  bool ShouldFlush();

  const std::unique_ptr<SgfRecorder> sgf_recorder_;
  const std::unique_ptr<TfRecorder> tf_recorder_;

  std::atomic<bool> running_;
  std::thread io_thread_;

  absl::Mutex mu_;
  std::array<absl::Mutex, constants::kMaxNumThreads> thread_mus_;
  int games_buffered_ ABSL_GUARDED_BY(mu_);
  int games_written_ ABSL_GUARDED_BY(mu_);
  bool should_flush_ ABSL_GUARDED_BY(mu_);
  const int num_threads_;
  const int flush_interval_;
  const std::string path_;
};

GameRecorderImpl::GameRecorderImpl(std::string path, int num_threads,
                                   int flush_interval, int gen,
                                   std::string worker_id)
    : sgf_recorder_(SgfRecorder::Create(FilePath(path) / recorder::kSgfDir,
                                        num_threads, gen, worker_id)),
      tf_recorder_(TfRecorder::Create(FilePath(path) / recorder::kChunkDir,
                                      num_threads, gen, worker_id)),
      running_(true),
      games_buffered_(0),
      games_written_(0),
      should_flush_(false),
      num_threads_(num_threads),
      flush_interval_(flush_interval),
      path_(path) {
  io_thread_ = std::thread(&GameRecorderImpl::IoThread, this);
}

GameRecorderImpl::~GameRecorderImpl() {
  mu_.Lock();
  running_.store(false, std::memory_order_release);
  mu_.Unlock();

  if (io_thread_.joinable()) {
    io_thread_.join();
  }
}

void GameRecorderImpl::RecordGame(
    int thread_id, const game::Board& init_board, const game::Game& game,
    const ImprovedPolicies& mcts_pis,
    const std::vector<uint8_t>& is_move_trainable,
    const std::vector<float>& root_qs,
    std::vector<std::unique_ptr<mcts::TreeNode>>&& roots) {
  if (path_.empty()) {
    return;
  }

  thread_mus_[thread_id].Lock();
  if (init_board.IsEmpty()) {
    sgf_recorder_->RecordGame(
        thread_id, game, kP3achyGoName, kP3achyGoName,
        std::forward<std::vector<std::unique_ptr<mcts::TreeNode>>>(roots));
  }
  tf_recorder_->RecordGame(thread_id, init_board, game, mcts_pis,
                           is_move_trainable, root_qs);
  thread_mus_[thread_id].Unlock();

  absl::MutexLock lock(&mu_);
  ++games_buffered_;
  should_flush_ = games_buffered_ >= flush_interval_;

  DLOG_EVERY_N_SEC(INFO, 5) << games_buffered_ << " games buffered.";
}

void GameRecorderImpl::RecordEvalGame(int thread_id, const game::Game& game,
                                      const std::string& b_name,
                                      const std::string& w_name) {
  if (path_.empty()) {
    return;
  }

  // Only log to SGF recorder. We also rely on flush-on-exit to flush SGFs.
  absl::MutexLock l(&thread_mus_[thread_id]);
  sgf_recorder_->RecordGame(thread_id, game, b_name, w_name, {});
}

void GameRecorderImpl::IoThread() {
  if (path_.empty()) {
    return;
  }

  while (running_.load(std::memory_order_acquire)) {
    mu_.LockWhen(absl::Condition(this, &GameRecorderImpl::ShouldFlush));

    auto begin = std::chrono::high_resolution_clock::now();
    Flush();
    auto end = std::chrono::high_resolution_clock::now();

    games_written_ += games_buffered_;
    games_buffered_ = 0;
    should_flush_ = false;

    auto flush_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    LOG(INFO) << "Flushing took " << flush_us << "us. Written "
              << games_written_ << " games so far.";
    mu_.Unlock();
  }
}

void GameRecorderImpl::Flush() {
  // Lock all threads.
  if (path_.empty()) {
    return;
  }

  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_mus_[thread_id].Lock();
  }

  sgf_recorder_->Flush();
  tf_recorder_->Flush();

  // Unlock all threads.
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_mus_[thread_id].Unlock();
  }
}

bool GameRecorderImpl::ShouldFlush() {
  return should_flush_ || !running_.load(std::memory_order_acquire);
}
}  // namespace

/* static */ std::unique_ptr<GameRecorder> GameRecorder::Create(
    std::string path, int num_threads, int flush_interval, int gen,
    std::string worker_id) {
  return std::make_unique<GameRecorderImpl>(path, num_threads, flush_interval,
                                            gen, worker_id);
}
}  // namespace recorder
