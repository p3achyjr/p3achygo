#include "cc/recorder/game_recorder.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "absl/log/log.h"
#include "cc/core/filepath.h"
#include "cc/game/game.h"
#include "cc/recorder/dir.h"

namespace recorder {
namespace {

using ::core::FilePath;

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

  void RecordGame(int thread_id, const game::Game& game,
                  const ImprovedPolicies& mcts_pis,
                  const std::vector<uint8_t>& is_move_trainable) override;

  static std::unique_ptr<GameRecorderImpl> Create(std::string path,
                                                  int num_threads,
                                                  int flush_interval,
                                                  std::string worker_id);

 private:
  void IoThread();
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
      flush_interval_(flush_interval) {
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
    int thread_id, const game::Game& game, const ImprovedPolicies& mcts_pis,
    const std::vector<uint8_t>& is_move_trainable) {
  thread_mus_[thread_id].Lock();
  sgf_recorder_->RecordGame(thread_id, game);
  tf_recorder_->RecordGame(thread_id, game, mcts_pis, is_move_trainable);
  thread_mus_[thread_id].Unlock();

  absl::MutexLock lock(&mu_);
  ++games_buffered_;
  should_flush_ = games_buffered_ >= flush_interval_;

  DLOG_EVERY_N_SEC(INFO, 5) << games_buffered_ << " games buffered.";
}

void GameRecorderImpl::IoThread() {
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
