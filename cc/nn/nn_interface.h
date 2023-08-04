#ifndef NN_INTERFACE_H_
#define NN_INTERFACE_H_

#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "cc/constants/constants.h"
#include "cc/core/cache.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/symmetry.h"
#include "cc/nn/feed_fetch_names.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/saved_model/loader.h"

namespace tensorflow {
class GraphDef;
class Tensor;
}  // namespace tensorflow

namespace nn {

struct NNInferResult {
  std::array<float, constants::kMaxMovesPerPosition> move_logits;
  std::array<float, constants::kMaxMovesPerPosition> move_probs;
  std::array<float, constants::kNumValueLogits> value_probs;
  std::array<float, constants::kNumScoreLogits> score_probs;
};

/*
 * Bridge between `Board` and other game objects to NN evaluation.
 *
 * This is an interface containing a single model. Each thread will fill its
 * slot in the neural network batch, and when the batch is full, inference will
 * be performed.
 */
class NNInterface final {
 public:
  NNInterface(int num_threads);
  NNInterface(int num_threads, int64_t timeout, size_t cache_size);
  ~NNInterface();

  // Disable Copy
  NNInterface(NNInterface const&) = delete;
  NNInterface& operator=(NNInterface const&) = delete;

  absl::Status Initialize(std::string&& model_path);

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

  // Called by worker.
  inline void SignalLoadedAndBlockUntilReady(int thread_id)
      ABSL_LOCKS_EXCLUDED(mu_) {
    if (num_threads_ == 1) {
      // Fast, non-locking path.
      std::vector<std::pair<std::string, ::tensorflow::Tensor>> nn_input = {
          {kInputNames[0], nn_input_buf_[0]},
          {kInputNames[1], nn_input_buf_[1]}};
      TF_CHECK_OK(model_bundle_.GetSession()->Run(nn_input, kOutputNames, {},
                                                  &nn_output_buf_));
      nn_input_buf_[0].SubSlice(thread_id).unaligned_flat<float>().setZero();
      nn_input_buf_[1].SubSlice(thread_id).unaligned_flat<float>().setZero();
      return;
    }

    ThreadInfo& thread_info = thread_info_[thread_id];

    mu_.Lock();
    thread_info.loaded_for_inference = true;
    thread_info.res_ready = false;
    thread_info.res_cached = false;

    // Wait for result.
    mu_.Await(absl::Condition(&thread_info.res_ready));
    thread_info.res_ready = false;
    mu_.Unlock();
  }

  // Inference Loop.
  void InferLoop();
  void Infer() ABSL_LOCKS_EXCLUDED(mu_);
  bool ShouldInfer() const;

  ::tensorflow::SavedModelBundleLite model_bundle_;
  ::tensorflow::SessionOptions session_options_;
  ::tensorflow::RunOptions run_options_;

  std::vector<::tensorflow::Tensor> nn_input_buf_;
  std::vector<::tensorflow::Tensor> nn_output_buf_;

  std::string id_;
  bool is_initialized_;
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
};

}  // namespace nn

#endif
