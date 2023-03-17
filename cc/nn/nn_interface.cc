#include "cc/nn/nn_interface.h"

#include <stdlib.h>

#include "absl/log/check.h"
#include "cc/nn/nn_board_utils.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

namespace nn {
namespace {

using namespace ::tensorflow;

static constexpr auto kMoveIndex = 0;
static constexpr auto kValueIndex = 1;
static constexpr auto kOwnershipIndex = 2;
static constexpr auto kScoreIndex = 3;
static constexpr auto kMoveProbabilitiesIndex = 4;

static const std::vector<std::string> kInputNames = {
    "infer_mixed_board_state:0",
    "infer_mixed_game_state:0",
};

static const std::vector<std::string> kOutputNames = {
    "StatefulPartitionedCall:0", "StatefulPartitionedCall:1",
    "StatefulPartitionedCall:2", "StatefulPartitionedCall:3",
    "StatefulPartitionedCall:4",
};

}  // namespace

NNInterface::NNInterface() {
  nn_output_buf_ = {// move logits
                    Tensor(DataType::DT_HALF, {1, constants::kMaxNumMoves}),
                    // win percent
                    Tensor(DataType::DT_HALF, {1, constants::kNumValueLogits}),
                    // ownership
                    Tensor(DataType::DT_HALF, {1, BOARD_LEN, BOARD_LEN, 1}),
                    // score prediction
                    Tensor(DataType::DT_HALF, {1, constants::kNumScoreLogits}),
                    // gamma, just ignore
                    Tensor(DataType::DT_HALF, {1, 1})};

  // mirrors `nn_output_buf_`, but omits gamma and casts to float.
  result_buf_ = {Tensor(DataType::DT_FLOAT, {1, constants::kMaxNumMoves}),
                 Tensor(DataType::DT_FLOAT, {1, constants::kNumValueLogits}),
                 Tensor(DataType::DT_FLOAT, {1, constants::kMaxNumBoardLocs}),
                 Tensor(DataType::DT_FLOAT, {1, constants::kNumScoreLogits}),
                 Tensor(DataType::DT_FLOAT, {1, constants::kMaxNumMoves})};
}

absl::Status NNInterface::Initialize(std::string&& model_path) {
  auto status =
      nn_evaluator_.InitFromPath(std::forward<std::string>(model_path));
  if (!status.ok()) {
    return ToAbslStatus(status);
  }

  is_initialized_ = true;
  return absl::OkStatus();
}

absl::StatusOr<NNInferResult> NNInterface::GetInferenceResult(
    const game::Board& board, const std::vector<game::Loc> last_moves,
    int color_to_move) {
  if (!is_initialized_) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "Need to initialize NNInterface before using inference.");
  }

  DCHECK(last_moves.size() >= constants::kNumLastMoves);
  Scope scope = Scope::NewRootScope();
  ClientSession session(scope);
  std::vector<std::pair<std::string, Tensor>> nn_input =
      NNBoardUtils::ConstructNNInput(session, scope, board, color_to_move,
                                     last_moves, kInputNames);
  auto status = nn_evaluator_.Infer(nn_input, kOutputNames, &nn_output_buf_);
  if (!status.ok()) {
    return ToAbslStatus(status);
  }

  // cast back to float
  status = session.Run(
      {ops::Cast(scope, nn_output_buf_[kMoveIndex], DataType::DT_FLOAT),
       ops::Softmax(scope, ops::Cast(scope, nn_output_buf_[kValueIndex],
                                     DataType::DT_FLOAT)),
       ops::Cast(scope, nn_output_buf_[kOwnershipIndex], DataType::DT_FLOAT),
       ops::Softmax(scope, ops::Cast(scope, nn_output_buf_[kScoreIndex],
                                     DataType::DT_FLOAT)),
       ops::Softmax(scope, ops::Cast(scope, nn_output_buf_[kMoveIndex],
                                     DataType::DT_FLOAT))},
      &result_buf_);

  const auto move_logits = result_buf_[kMoveIndex].flat<float>().data();
  const auto value_probability = result_buf_[kValueIndex].flat<float>().data();
  const auto ownership = result_buf_[kOwnershipIndex].flat<float>().data();
  const auto score_probabilities =
      result_buf_[kScoreIndex].flat<float>().data();
  const auto move_probabilities =
      result_buf_[kMoveProbabilitiesIndex].flat<float>().data();
  NNInferResult infer_result;

  std::copy(move_logits, move_logits + constants::kMaxNumMoves,
            infer_result.move_logits);
  std::copy(move_probabilities, move_probabilities + constants::kMaxNumMoves,
            infer_result.move_probabilities);
  std::copy(value_probability, value_probability + constants::kNumValueLogits,
            infer_result.value_probability);
  std::copy(ownership, ownership + constants::kMaxNumBoardLocs,
            infer_result.ownership);
  std::copy(score_probabilities,
            score_probabilities + constants::kNumScoreLogits,
            infer_result.score_probabilities);

  return absl::OkStatus();
}

}  // namespace nn