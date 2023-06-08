#ifndef __NN_INTERFACE_H_
#define __NN_INTERFACE_H_

#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cc/constants/constants.h"
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
  float move_logits[constants::kMaxNumMoves];
  float move_probs[constants::kMaxNumMoves];
  float value_probs[constants::kNumValueLogits];
  float score_probs[constants::kNumScoreLogits];
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
  ~NNInterface();

  // Disable Copy
  NNInterface(NNInterface const&) = delete;
  NNInterface& operator=(NNInterface const&) = delete;

  absl::Status Initialize(std::string&& model_path);

  // Blocks until result is ready.
  NNInferResult LoadAndGetInference(int thread_id, const game::Game& game,
                                    game::Color color_to_move)
      ABSL_LOCKS_EXCLUDED(mu_);

  void RegisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);
  void UnregisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);

 private:
  static constexpr int64_t kTimeoutUs = 30000;
  struct CacheKey {
    game::Color color_to_move;
    game::Zobrist::Hash board_hash;

    friend bool operator==(const CacheKey& c0, const CacheKey& c1) {
      return c0.color_to_move == c1.color_to_move &&
             c0.board_hash == c1.board_hash;
    }

    template <typename H>
    friend H AbslHashValue(H h, const CacheKey& c) {
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

  class Cache final {
   public:
    Cache(int num_threads);
    ~Cache() = default;

    // Disable Copy and Move.
    Cache(Cache const&) = delete;
    Cache& operator=(Cache const&) = delete;
    Cache(Cache&&) = delete;
    Cache& operator=(Cache&&) = delete;

    void Insert(int thread_id, const CacheKey& cache_key,
                const NNInferResult& infer_result);
    bool Contains(int thread_id, const CacheKey& cache_key);
    std::optional<NNInferResult> Get(int thread_id, const CacheKey& cache_key);

   private:
    struct CacheElem {
      size_t hash;
      NNInferResult infer_res;
    };

    const int num_threads_;
    const size_t thread_cache_size_;
    std::array<std::vector<std::optional<CacheElem>>, constants::kMaxNumThreads>
        cache_;
  };

  void InferLoop();
  void Infer() ABSL_LOCKS_EXCLUDED(mu_);
  bool ShouldInfer() const;

  ::tensorflow::SavedModelBundleLite model_bundle_;
  ::tensorflow::SessionOptions session_options_;
  ::tensorflow::RunOptions run_options_;

  ::tensorflow::Scope scope_preprocess_;
  ::tensorflow::Scope scope_postprocess_;
  ::tensorflow::GraphDef gdef_preprocess_;
  ::tensorflow::GraphDef gdef_postprocess_;
  ::tensorflow::Tensor input_feature_buf_;
  ::tensorflow::Tensor input_state_buf_;
  std::vector<::tensorflow::Tensor> nn_input_buf_;
  std::vector<::tensorflow::Tensor> nn_output_buf_;
  std::vector<::tensorflow::Tensor> result_buf_;

  std::unique_ptr<::tensorflow::Session> session_preprocess_;
  std::unique_ptr<::tensorflow::Session> session_postprocess_;

  bool is_initialized_;
  int num_registered_threads_ ABSL_GUARDED_BY(mu_);
  const int num_threads_;

  Cache nn_cache_;  // Per-thread cache.

  // Synchronization
  absl::Mutex mu_;
  absl::InlinedVector<ThreadInfo, constants::kMaxNumThreads> thread_info_;

  // Inference thread. Runs inference until told to stop.
  std::thread infer_thread_;
  std::atomic<bool> running_;

  std::chrono::time_point<std::chrono::steady_clock> last_infer_time_;
};

}  // namespace nn

#endif
