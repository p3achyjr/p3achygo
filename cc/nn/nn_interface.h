#ifndef __NN_INTERFACE_H_
#define __NN_INTERFACE_H_

#include <atomic>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/blocking_counter.h"
#include "cc/constants/constants.h"
#include "cc/game/board.h"
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
  float move_probabilities[constants::kMaxNumMoves];
  float value_probability[constants::kNumValueLogits];
  float ownership[constants::kMaxNumBoardLocs];
  float score_probabilities[constants::kNumScoreLogits];
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
  absl::Status LoadBatch(int thread_id, const game::Board& board,
                         const std::vector<game::Loc> last_moves,
                         int color_to_move);
  NNInferResult GetInferenceResult(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);

  void RegisterThread(int thread_id);
  void UnregisterThread(int thread_id);

 private:
  void InferLoop();
  void Infer() ABSL_LOCKS_EXCLUDED(mu_);

  ::tensorflow::SavedModelBundleLite model_bundle_;
  ::tensorflow::SessionOptions session_options_;
  ::tensorflow::RunOptions run_options_;

  ::tensorflow::Scope scope_cast_input_;
  ::tensorflow::Scope scope_cast_output_;
  ::tensorflow::GraphDef gdef_cast_input_;
  ::tensorflow::GraphDef gdef_cast_output_;
  std::unique_ptr<::tensorflow::Session> session_cast_input_;
  std::unique_ptr<::tensorflow::Session> session_cast_output_;
  ::tensorflow::Tensor input_feature_buf_;
  ::tensorflow::Tensor input_state_buf_;
  std::vector<::tensorflow::Tensor> nn_input_buf_;
  std::vector<::tensorflow::Tensor> nn_output_buf_;
  std::vector<::tensorflow::Tensor> result_buf_;

  bool is_initialized_;
  int num_threads_;
  int batch_size_;

  // Synchronization
  std::unique_ptr<absl::BlockingCounter> load_counter_;
  absl::Mutex mu_;
  std::vector<uint8_t> registered_;
  std::vector<uint8_t> batch_ready_;  // index `i` indicates whether
                                      // result for thread `i` is ready.

  // inference thread. Runs inference until told to stop.
  std::thread infer_thread_;
  std::atomic<bool> running_;
};

}  // namespace nn

#endif
