#include "cc/nn/nn_interface.h"

#include <stdlib.h>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "cc/nn/nn_board_utils.h"
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
static constexpr auto kMoveProbsIndex = 4;

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

NNInterface::NNInterface(int num_threads)
    : is_initialized_(false),
      num_threads_(num_threads),
      load_counter_(std::make_unique<absl::BlockingCounter>(num_threads)),
      infer_result_ready_(
          std::vector<char>(num_threads_, static_cast<char>(false))),
      running_(true) {
  input_feature_buf_ = Tensor(
      DataType::DT_FLOAT,
      {num_threads_, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes});
  input_state_buf_ = Tensor(DataType::DT_FLOAT, {num_threads_, 1});
  nn_input_buf_ = {
      Tensor(DataType::DT_HALF, {num_threads_, BOARD_LEN, BOARD_LEN,
                                 constants::kNumInputFeaturePlanes}),
      Tensor(DataType::DT_HALF, {num_threads_, 1})};

  input_feature_buf_.flat<float>().setZero();
  input_state_buf_.flat<float>().setZero();

  nn_output_buf_ = {
      // move logits
      Tensor(DataType::DT_HALF, {num_threads_, constants::kMaxNumMoves}),
      // win percent
      Tensor(DataType::DT_HALF, {num_threads_, constants::kNumValueLogits}),
      // ownership
      Tensor(DataType::DT_HALF, {num_threads_, BOARD_LEN, BOARD_LEN, 1}),
      // score prediction
      Tensor(DataType::DT_HALF, {num_threads_, constants::kNumScoreLogits}),
      // gamma, just ignore
      Tensor(DataType::DT_HALF, {num_threads_, 1})};

  result_buf_ = {
      // move logits
      Tensor(DataType::DT_FLOAT, {num_threads_, constants::kMaxNumMoves}),
      // win percent
      Tensor(DataType::DT_FLOAT, {num_threads_, constants::kNumValueLogits}),
      // ownership
      Tensor(DataType::DT_FLOAT, {num_threads_, BOARD_LEN, BOARD_LEN, 1}),
      // score prediction
      Tensor(DataType::DT_FLOAT, {num_threads_, constants::kNumScoreLogits}),
      // move softmax
      Tensor(DataType::DT_FLOAT, {num_threads_, constants::kMaxNumMoves})};
}

NNInterface::~NNInterface() {
  running_.store(false, std::memory_order_release);

  if (infer_thread_.joinable()) {
    infer_thread_.join();
  }
}

absl::Status NNInterface::Initialize(std::string&& model_path) {
  auto status =
      nn_evaluator_.InitFromPath(std::forward<std::string>(model_path));
  if (!status.ok()) {
    return ToAbslStatus(status);
  }

  is_initialized_ = true;

  infer_thread_ = std::move(std::thread(&NNInterface::InferLoop, this));
  infer_thread_.detach();
  return absl::OkStatus();
}

absl::Status NNInterface::LoadBatch(int thread_id, const game::Board& board,
                                    const std::vector<game::Loc> last_moves,
                                    int color_to_move) {
  if (!is_initialized_) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "Need to initialize NNInterface before using inference.");
  }

  DCHECK(last_moves.size() >= constants::kNumLastMoves);
  Scope scope = Scope::NewRootScope();
  ClientSession session(scope);
  NNBoardUtils::FillNNInput(session, scope, thread_id, num_threads_,
                            input_feature_buf_, input_state_buf_, board,
                            color_to_move, last_moves);

  load_counter_->DecrementCount();
  return absl::OkStatus();
}

NNInferResult NNInterface::GetInferenceResult(int thread_id) {
  mu_.LockWhen(absl::Condition(
      reinterpret_cast<bool*>(infer_result_ready_.data() + thread_id)));
  infer_result_ready_[thread_id] = false;
  mu_.Unlock();

  const auto move_logits =
      result_buf_[kMoveIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto value_probability =
      result_buf_[kValueIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto ownership =
      result_buf_[kOwnershipIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto score_probabilities =
      result_buf_[kScoreIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto move_probabilities =
      result_buf_[kMoveProbsIndex].SubSlice(thread_id).unaligned_flat<float>();

  // need hand-rolled for loops b/c of potential alignment issues.
  NNInferResult infer_result;
  for (int i = 0; i < constants::kMaxNumMoves; ++i) {
    infer_result.move_logits[i] = move_logits(i);
  }
  for (int i = 0; i < constants::kMaxNumMoves; ++i) {
    infer_result.move_probabilities[i] = move_probabilities(i);
  }
  for (int i = 0; i < constants::kNumValueLogits; ++i) {
    infer_result.value_probability[i] = value_probability(i);
  }
  for (int i = 0; i < constants::kMaxNumBoardLocs; ++i) {
    infer_result.ownership[i] = ownership(i);
  }
  for (int i = 0; i < constants::kNumScoreLogits; ++i) {
    infer_result.score_probabilities[i] = score_probabilities(i);
  }

  return infer_result;
}

void NNInterface::InferLoop() {
  Scope scope = Scope::NewRootScope();
  ClientSession session(scope);

  while (running_.load(std::memory_order_acquire)) {
    Infer(scope, session);
  }
}

void NNInterface::Infer(Scope& scope, ClientSession& session) {
  load_counter_->Wait();
  absl::MutexLock l(&mu_);

  // Cast to half.
  CHECK_OK(ToAbslStatus(
      session.Run({ops::Cast(scope, input_feature_buf_, DataType::DT_HALF),
                   ops::Cast(scope, input_state_buf_, DataType::DT_HALF)},
                  &nn_input_buf_)));

  std::vector<std::pair<std::string, Tensor>> nn_input = {
      {kInputNames[0], nn_input_buf_[0]}, {kInputNames[1], nn_input_buf_[1]}};

  CHECK_OK(ToAbslStatus(
      nn_evaluator_.Infer(nn_input, kOutputNames, &nn_output_buf_)));

  input_feature_buf_.flat<float>().setZero();
  input_state_buf_.flat<float>().setZero();

  // cast back to float
  CHECK_OK(ToAbslStatus(session.Run(
      {ops::Cast(scope, nn_output_buf_[kMoveIndex], DataType::DT_FLOAT),
       ops::Softmax(scope, ops::Cast(scope, nn_output_buf_[kValueIndex],
                                     DataType::DT_FLOAT)),
       ops::Cast(scope, nn_output_buf_[kOwnershipIndex], DataType::DT_FLOAT),
       ops::Softmax(scope, ops::Cast(scope, nn_output_buf_[kScoreIndex],
                                     DataType::DT_FLOAT)),
       ops::Softmax(scope, ops::Cast(scope, nn_output_buf_[kMoveIndex],
                                     DataType::DT_FLOAT))},
      &result_buf_)));

  load_counter_ = std::make_unique<absl::BlockingCounter>(num_threads_);
  std::fill(infer_result_ready_.begin(), infer_result_ready_.end(),
            static_cast<char>(true));
}

}  // namespace nn