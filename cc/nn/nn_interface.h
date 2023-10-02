#ifndef NN_INTERFACE_H_
#define NN_INTERFACE_H_

#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "cc/constants/constants.h"
#include "cc/core/cache.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/symmetry.h"
#include "cc/nn/engine/engine.h"

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
class NNInterface final {
 public:
  NNInterface(int num_threads, std::unique_ptr<Engine> engine);
  NNInterface(int num_threads, int64_t timeout, size_t cache_size,
              std::unique_ptr<Engine> engine);
  ~NNInterface();

  // Disable Copy
  NNInterface(NNInterface const&) = delete;
  NNInterface& operator=(NNInterface const&) = delete;

  // Blocks until result is ready.
  NNInferResult LoadAndGetInference(int thread_id, const game::Game& game,
                                    game::Color color_to_move,
                                    core::Probability& probability)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Just get ownership.
  std::array<float, constants::kNumBoardLocs> LoadAndGetOwnership(
      int thread_id, const game::Game& game, game::Color color_to_move)
      ABSL_LOCKS_EXCLUDED(mu_);

  void RegisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);
  void UnregisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);

 private:
  static constexpr int64_t kTimeoutUs = 30000;
  struct NNKey {
    game::Color color_to_move;
    game::Zobrist::Hash board_hash;

    friend bool operator==(const NNKey& c0, const NNKey& c1) {
      return c0.color_to_move == c1.color_to_move &&
             c0.board_hash == c1.board_hash;
    }

    template <typename H>
    friend H AbslHashValue(H h, const NNKey& c) {
      return H::combine(std::move(h), c.color_to_move, c.board_hash);
    }
  };

  struct ThreadInfo {
    bool registered = true;  // If this thread is registered.
    bool loaded_for_inference =
        false;  // If this thread has loaded its data for inference.
    bool res_ready = false;   // Whether inference result is ready.
    bool res_cached = false;  // Whether the cache key is cached. Also toggles
                              // whether to cache the NN inference result.
  };

  // Cache Helpers.
  void InitializeCache(size_t cache_size);
  bool CacheContains(int thread_id, const NNKey& key);
  std::optional<NNInferResult> CacheGet(int thread_id, const NNKey& key);
  void CacheInsert(int thread_id, const NNKey& key,
                   const NNInferResult& result);
  void LoadBatch(int thread_id, const game::Game& game,
                 game::Color color_to_move, game::Symmetry sym);

  // Called by worker.
  inline void SignalLoadedAndBlockUntilReady(int thread_id)
      ABSL_LOCKS_EXCLUDED(mu_) {
    if (num_threads_ == 1) {
      // Fast, non-locking path.
      engine_->RunInference();
      return;
    }

    ThreadInfo& thread_info = thread_info_[thread_id];

    mu_.Lock();
    thread_info.loaded_for_inference = true;
    thread_info.res_ready = false;
    thread_info.res_cached = false;

    // Wait for result.
    mu_.Await(absl::Condition(&thread_info.res_ready));
    mu_.Unlock();
  }

  // Inference Loop.
  void InferLoop();
  void Infer() ABSL_LOCKS_EXCLUDED(mu_);
  bool ShouldInfer() const;

  std::string id_;
  int num_registered_threads_ ABSL_GUARDED_BY(mu_);
  const int num_threads_;

  // Synchronization
  absl::Mutex mu_;
  absl::InlinedVector<ThreadInfo, constants::kMaxNumThreads> thread_info_;

  // Inference thread. Runs inference until told to stop.
  std::thread infer_thread_;
  std::atomic<bool> running_;

  std::array<core::Cache<NNKey, NNInferResult>, constants::kMaxNumThreads>
      thread_caches_;  // Per-thread cache.
  const int64_t timeout_;

  // Engine.
  std::unique_ptr<Engine> engine_;
};

}  // namespace nn

#endif
