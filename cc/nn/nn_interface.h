#ifndef NN_INTERFACE_H_
#define NN_INTERFACE_H_

#include <array>
#include <atomic>
#include <optional>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/lru_cache.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/symmetry.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/nn_interface_base.h"

namespace tensorflow {
class GraphDef;
class Tensor;
}  // namespace tensorflow

namespace nn {

/*
 * Bridge between `Board` and other game objects to NN evaluation.
 *
 * This is an interface containing a single model. Each thread will fill its
 * slot in the neural network batch, and when the batch is full, inference will
 * be performed.
 */
class NNInterface final : public NNInterfaceBase {
 public:
  NNInterface(int num_threads, std::unique_ptr<Engine> engine);
  NNInterface(int num_threads, int64_t timeout, size_t cache_size,
              std::unique_ptr<Engine> engine);
  // Convenience ctor for testing: same as above but with explicit WakeStrategy.
  NNInterface(int num_threads, int64_t timeout, size_t cache_size,
              std::unique_ptr<Engine> engine, WakeStrategy wake_strategy);

  // We only specify signal_kind during parallel search, meaning we also need to
  // specify how many search tasks share this interface.
  NNInterface(int num_threads, std::unique_ptr<Engine> engine,
              SignalKind signal_kind, int num_shared_search_tasks);
  NNInterface(int num_threads, int64_t timeout, size_t cache_size,
              std::unique_ptr<Engine> engine, SignalKind signal_kind,
              int num_shared_search_tasks,
              WakeStrategy wake_strategy = WakeStrategy::kGenCounter);
  ~NNInterface() override;

  // Disable Copy
  NNInterface(NNInterface const&) = delete;
  NNInterface& operator=(NNInterface const&) = delete;

  SignalKind signal_kind() const override { return signal_kind_; }

  // Controls how many of the most-recent moves are included in the NN cache
  // key. Range [0, kNumLastMoves]. Default: kNumLastMoves (5).
  void SetNumCacheLastMoves(int n) { num_cache_last_moves_ = n; }

  // Whether to include the board's simple-ko point in the NN cache key.
  void SetCacheKeyKo(bool b) { cache_key_ko_ = b; }

  // Blocks until result is ready.
  NNInferResult LoadAndGetInference(int thread_id, const game::Game& game,
                                    game::Color color_to_move,
                                    core::Probability& probability)
      ABSL_LOCKS_EXCLUDED(mu_) override;

  // Just get ownership.
  std::array<float, constants::kNumBoardLocs> LoadAndGetOwnership(
      int thread_id, const game::Game& game, game::Color color_to_move)
      ABSL_LOCKS_EXCLUDED(mu_) override;

  void RegisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);
  void UnregisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);

  // async API
  void LoadEntry(int thread_id, int offset, const game::Game& game,
                 game::Color color_to_move, core::Probability& probability)
      ABSL_LOCKS_EXCLUDED(mu_) override;
  NNInferResult FetchEntry(int thread_id, int offset, const game::Game& game,
                           game::Color color_to_move)
      ABSL_LOCKS_EXCLUDED(mu_) override;
  // One wave: `wave` is always 0.
  inline void SignalReadyForInference(int wave = 0) override {
    {
      absl::MutexLock l(&mu_);
      ++num_signaled_search_tasks_;
    }
    if (num_threads_ == 1) {
      Infer();  // No infer_thread_ exists; run synchronously.
    }
  }
  inline void UnregisterSearchTask(int wave = 0) override {
    absl::MutexLock l(&mu_);
    ++num_exited_search_tasks_;
  }
  int num_waves() const override { return 1; }
  int wave_size() const override { return num_threads_; }

 private:
  static constexpr int64_t kTimeoutUs = 400;

  struct ThreadInfo {
    bool registered = true;  // If this thread is registered.
    bool loaded_for_inference =
        false;  // If this thread has loaded its data for inference.
    std::atomic<bool> res_ready{false};  // Whether inference result is ready.
    bool res_cached = false;  // Whether the cache key is cached. Also toggles
                              // whether to cache the NN inference result.

    ThreadInfo() = default;
    ThreadInfo(const ThreadInfo&) = delete;
    ThreadInfo& operator=(const ThreadInfo&) = delete;
  };

  // Cache Helpers.
  void InitializeCache(size_t cache_size);
  bool CacheContains(int thread_id, const NNKey& key);
  std::optional<NNInferResult> CacheGet(int thread_id, const NNKey& key);
  void CacheInsert(int thread_id, const NNKey& key,
                   const NNInferResult& result);
  void LoadBatch(int thread_id, const game::Game& game,
                 game::Color color_to_move, game::Symmetry sym);
  inline NNInferResult GetBatch(int thread_id, game::Symmetry sym)
      ABSL_LOCKS_EXCLUDED(mu_) {
    NNInferResult infer_result;
    engine_->GetBatch(thread_id, infer_result);

    // Clear res_ready without the lock. Safe because:
    // - engine_->GetBatch() completes before this store (release ordering).
    // - Infer() checks res_ready with acquire ordering before RunInference().
    // - So if infer sees false, GetBatch has already consumed the engine
    // buffer.
    thread_info_[thread_id].res_ready.store(false, std::memory_order_release);

    UnapplySymmetry(sym, infer_result);
    return infer_result;
  }

  // Called by worker.
  inline void SignalLoadedAndBlockUntilReady(int thread_id)
      ABSL_LOCKS_EXCLUDED(mu_) {
    if (num_threads_ == 1) {
      engine_->RunInference();
      return;
    }
    ThreadInfo& thread_info = thread_info_[thread_id];
    {
      absl::MutexLock l(&mu_);
      thread_info.loaded_for_inference = true;
      thread_info.res_ready.store(false, std::memory_order_relaxed);
      thread_info.res_cached = false;
    }
    // mu_ released before calling waker — same convention for both strategies.
    waker_->Wait(thread_info.res_ready, mu_);
  }

  inline void BlockUntilReady(int thread_id) ABSL_LOCKS_EXCLUDED(mu_) {
    waker_->Wait(thread_info_[thread_id].res_ready, mu_);
  }

  NNKey MakeKey(const game::Game& game, game::Color color_to_move) const;

  // Inference Loop.
  void InferLoop();
  void Infer() ABSL_LOCKS_EXCLUDED(mu_);
  bool ShouldInfer() const;

  std::string id_;
  int num_registered_threads_ ABSL_GUARDED_BY(mu_);
  const int num_threads_;
  const SignalKind signal_kind_;
  int num_cache_last_moves_ = constants::kNumLastMoves;
  bool cache_key_ko_ = false;

  // Synchronization
  absl::Mutex mu_;
  absl::InlinedVector<ThreadInfo, constants::kMaxNumThreads> thread_info_;

  // Inference thread. Runs inference until told to stop.
  std::thread infer_thread_;
  std::atomic<bool> running_;

  std::array<core::LRUCache<NNKey, NNInferResult>, constants::kMaxNumThreads>
      thread_caches_;  // Per-thread cache.
  const int64_t timeout_;

  // Engine.
  std::unique_ptr<Engine> engine_;

  // Per-thread symmetries.
  absl::InlinedVector<game::Symmetry, constants::kMaxNumThreads> symmetries_;

  // Explicit signalling.
  const int num_shared_search_tasks_;
  int num_signaled_search_tasks_ = 0;
  int num_exited_search_tasks_ = 0;

  // Wakeup strategy: owns the waiting/notification mechanism.
  std::unique_ptr<Waker> waker_;
};

}  // namespace nn

#endif
