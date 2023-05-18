#ifndef __NN_INTERFACE_H_
#define __NN_INTERFACE_H_

#include <atomic>
#include <chrono>
#include <thread>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cc/constants/constants.h"
#include "cc/game/game.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/saved_model/loader.h"

namespace tensorflow {
class Tensor;
class GraphDef;
}  // namespace tensorflow

namespace nn {

struct NNInferResult {
  float move_logits[constants::kMaxNumMoves];
  float move_probs[constants::kMaxNumMoves];
  float value_probs[constants::kNumValueLogits];
  float ownership[constants::kMaxNumBoardLocs];
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

  // In order to ensure thread safety, threads must first call `LoadBatch`, and
  // then get the result for the corresponding thread index from
  // `GetInferenceResult`.
  // `GetInferenceResult` will block until the result for all threads is ready.
  absl::Status LoadBatch(int thread_id, const game::Game& game,
                         int color_to_move);
  NNInferResult GetInferenceResult(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);

  void RegisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);
  void UnregisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);

 private:
  static constexpr int64_t kTimeoutUs = 30000;
  struct ThreadInfo {
    bool registered = true;
    bool loaded = false;
    bool res_ready = false;
  };

  void InferLoop();
  void Infer() ABSL_LOCKS_EXCLUDED(mu_);
  bool ShouldInfer() const;

  ::tensorflow::SavedModelBundleLite model_bundle_;
  ::tensorflow::SessionOptions session_options_;
  ::tensorflow::RunOptions run_options_;

  ::tensorflow::Scope scope_cast_input_;
  ::tensorflow::Scope scope_cast_output_;
  ::tensorflow::GraphDef gdef_cast_input_;
  ::tensorflow::GraphDef gdef_cast_output_;
  ::tensorflow::Tensor input_feature_buf_;
  ::tensorflow::Tensor input_state_buf_;
  std::vector<::tensorflow::Tensor> nn_input_buf_;
  std::vector<::tensorflow::Tensor> nn_output_buf_;
  std::vector<::tensorflow::Tensor> result_buf_;

  std::unique_ptr<::tensorflow::Session> session_cast_input_;
  std::unique_ptr<::tensorflow::Session> session_cast_output_;

  bool is_initialized_;
  int num_threads_ ABSL_GUARDED_BY(mu_);
  const int batch_size_;

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
