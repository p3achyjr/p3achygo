#ifndef __NN_INTERFACE_H_
#define __NN_INTERFACE_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/nn/nn_evaluator.h"

namespace tensorflow {
class Tensor;
}

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
 */
class NNInterface final {
 public:
  NNInterface();
  ~NNInterface() = default;

  // Disable Copy and Move
  NNInterface(NNInterface const&) = delete;
  NNInterface& operator=(NNInterface const&) = delete;
  NNInterface(NNInterface&&) = delete;
  NNInterface& operator=(NNInterface&&) = delete;

  absl::Status Initialize(std::string&& model_path);
  absl::StatusOr<NNInferResult> GetInferenceResult(
      const game::Board& board, const std::vector<game::Loc> last_moves,
      int color_to_move);

 private:
  NNEvaluator nn_evaluator_;
  std::vector<::tensorflow::Tensor> nn_output_buf_;
  std::vector<::tensorflow::Tensor> result_buf_;

  bool is_initialized_ = false;
};

}  // namespace nn

#endif