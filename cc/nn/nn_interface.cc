#include "cc/nn/nn_interface.h"

#include <stdlib.h>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "cc/nn/nn_board_utils.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

namespace nn {
namespace {

using namespace ::tensorflow;

static constexpr int kMoveIndex = 0;
static constexpr int kValueIndex = 1;
static constexpr int kOwnerIndex = 2;
static constexpr int kScoreIndex = 3;
static constexpr int kMoveProbsIndex = 4;

static const std::vector<std::string> kInputNames = {
    "infer_mixed_board_state:0",
    "infer_mixed_game_state:0",
};

static const std::vector<std::string> kOutputNames = {
    "StatefulPartitionedCall:0", "StatefulPartitionedCall:1",
    "StatefulPartitionedCall:2", "StatefulPartitionedCall:3",
    "StatefulPartitionedCall:4",
};

// input feed and fetch names for F32 -> F16 cast.
static constexpr char kInput32BoardName[] = "input32_board_state";
static constexpr char kInput32GameName[] = "input32_game_state";

static constexpr char kInput16BoardName[] = "input16_board_state";
static constexpr char kInput16GameName[] = "input16_game_state";

// output feed and fetch names for F16 -> F32 cast.
static constexpr char kOut16MoveLogitsName[] = "out16_move_logits";
static constexpr char kOut16ValueLogitsName[] = "out16_value_logits";
static constexpr char kOut16OwnerLogitsName[] = "out16_owner_logits";
static constexpr char kOut16ScoreLogitsName[] = "out16_score_logits";

static constexpr char kOut32MoveLogitsName[] = "out32_move_logits";
static constexpr char kOut32ValueProbsName[] = "out32_value_probs";
static constexpr char kOut32OwnerLogitsName[] = "out32_owner_logits";
static constexpr char kOut32ScoreProbsName[] = "out32_score_probs";
static constexpr char kOut32MoveProbsName[] = "out32_move_probs";

}  // namespace

NNInterface::NNInterface(int num_threads)
    : session_options_(SessionOptions()),
      run_options_(RunOptions()),
      scope_cast_input_(Scope::NewRootScope()),
      scope_cast_output_(Scope::NewRootScope()),
      session_cast_input_(tensorflow::NewSession(session_options_)),
      session_cast_output_(tensorflow::NewSession(session_options_)),
      is_initialized_(false),
      num_threads_(num_threads),
      batch_size_(num_threads),
      thread_info_(batch_size_),
      running_(true),
      last_infer_time_(
          std::chrono::time_point<std::chrono::steady_clock>::max()) {
  input_feature_buf_ = Tensor(
      DataType::DT_FLOAT,
      {batch_size_, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes});
  input_state_buf_ = Tensor(DataType::DT_FLOAT, {batch_size_, 1});
  nn_input_buf_ = {
      Tensor(DataType::DT_HALF, {batch_size_, BOARD_LEN, BOARD_LEN,
                                 constants::kNumInputFeaturePlanes}),
      Tensor(DataType::DT_HALF, {batch_size_, 1})};

  input_feature_buf_.flat<float>().setZero();
  input_state_buf_.flat<float>().setZero();

  nn_output_buf_ = {
      // move logits
      Tensor(DataType::DT_HALF, {batch_size_, constants::kMaxNumMoves}),
      // win percent
      Tensor(DataType::DT_HALF, {batch_size_, constants::kNumValueLogits}),
      // ownership
      Tensor(DataType::DT_HALF, {batch_size_, BOARD_LEN, BOARD_LEN, 1}),
      // score prediction
      Tensor(DataType::DT_HALF, {batch_size_, constants::kNumScoreLogits}),
      // gamma, just ignore
      Tensor(DataType::DT_HALF, {batch_size_, 1})};

  result_buf_ = {
      // move logits
      Tensor(DataType::DT_FLOAT, {batch_size_, constants::kMaxNumMoves}),
      // win percent
      Tensor(DataType::DT_FLOAT, {batch_size_, constants::kNumValueLogits}),
      // ownership
      Tensor(DataType::DT_FLOAT, {batch_size_, BOARD_LEN, BOARD_LEN, 1}),
      // score prediction
      Tensor(DataType::DT_FLOAT, {batch_size_, constants::kNumScoreLogits}),
      // move softmax
      Tensor(DataType::DT_FLOAT, {batch_size_, constants::kMaxNumMoves})};
}

NNInterface::~NNInterface() {
  running_.store(false, std::memory_order_release);

  if (infer_thread_.joinable()) {
    infer_thread_.join();
  }

  if (session_cast_input_) {
    session_cast_input_->Close().IgnoreError();
  }

  if (session_cast_output_) {
    session_cast_output_->Close().IgnoreError();
  }
}

absl::Status NNInterface::Initialize(std::string&& model_path) {
  auto status =
      LoadSavedModel(session_options_, run_options_, model_path,
                     {tensorflow::kSavedModelTagServe}, &model_bundle_);
  if (!status.ok()) {
    return ToAbslStatus(status);
  }

  // TODO: Look into whether tensorrt can do this mess automatically.
  // Create graph to cast input to half precision.
  auto board = ops::Placeholder(scope_cast_input_.WithOpName(kInput32BoardName),
                                DataType::DT_FLOAT);
  auto game = ops::Placeholder(scope_cast_input_.WithOpName(kInput32GameName),
                               DataType::DT_FLOAT);
  auto cast_board = ops::Cast(scope_cast_input_.WithOpName(kInput16BoardName),
                              board, DataType::DT_HALF);
  auto cast_game = ops::Cast(scope_cast_input_.WithOpName(kInput16GameName),
                             game, DataType::DT_HALF);

  TF_CHECK_OK(scope_cast_input_.ToGraphDef(&gdef_cast_input_));
  TF_CHECK_OK(session_cast_input_->Create(gdef_cast_input_));

  // Create graph to cast output to float.
  auto move_logits = ops::Placeholder(
      scope_cast_output_.WithOpName(kOut16MoveLogitsName), DataType::DT_HALF);
  auto value_logits = ops::Placeholder(
      scope_cast_output_.WithOpName(kOut16ValueLogitsName), DataType::DT_HALF);
  auto owner_logits = ops::Placeholder(
      scope_cast_output_.WithOpName(kOut16OwnerLogitsName), DataType::DT_HALF);
  auto score_logits = ops::Placeholder(
      scope_cast_output_.WithOpName(kOut16ScoreLogitsName), DataType::DT_HALF);

  auto cast_move_logits =
      ops::Cast(scope_cast_output_.WithOpName(kOut32MoveLogitsName),
                move_logits, DataType::DT_FLOAT);
  auto value_probs = ops::Softmax(
      scope_cast_output_.WithOpName(kOut32ValueProbsName),
      ops::Cast(scope_cast_output_, value_logits, DataType::DT_FLOAT));
  auto cast_owner_logits =
      ops::Cast(scope_cast_output_.WithOpName(kOut32OwnerLogitsName),
                owner_logits, DataType::DT_FLOAT);
  auto score_probs = ops::Softmax(
      scope_cast_output_.WithOpName(kOut32ScoreProbsName),
      ops::Cast(scope_cast_output_, score_logits, DataType::DT_FLOAT));
  auto move_probs = ops::Softmax(
      scope_cast_output_.WithOpName(kOut32MoveProbsName),
      ops::Cast(scope_cast_output_, move_logits, DataType::DT_FLOAT));

  TF_CHECK_OK(scope_cast_output_.ToGraphDef(&gdef_cast_output_));
  TF_CHECK_OK(session_cast_output_->Create(gdef_cast_output_));

  infer_thread_ = std::move(std::thread(&NNInterface::InferLoop, this));
  is_initialized_ = true;

  return absl::OkStatus();
}

absl::Status NNInterface::LoadBatch(int thread_id, const game::Game& game,
                                    int color_to_move) {
  if (!is_initialized_) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "Need to initialize NNInterface before using inference.");
  }

  DCHECK(game.moves().size() >= constants::kNumLastMoves);
  NNBoardUtils::FillNNInput(thread_id, batch_size_, input_feature_buf_,
                            input_state_buf_, game, color_to_move);

  mu_.Lock();
  thread_info_[thread_id].loaded = true;
  mu_.Unlock();

  return absl::OkStatus();
}

NNInferResult NNInterface::GetInferenceResult(int thread_id) {
  mu_.LockWhen(absl::Condition(&thread_info_[thread_id].res_ready));
  thread_info_[thread_id].res_ready = false;
  mu_.Unlock();

  const auto move_logits =
      result_buf_[kMoveIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto value_probs =
      result_buf_[kValueIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto ownership =
      result_buf_[kOwnerIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto score_probs =
      result_buf_[kScoreIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto move_probs =
      result_buf_[kMoveProbsIndex].SubSlice(thread_id).unaligned_flat<float>();

  // need hand-rolled for loops b/c of potential alignment issues.
  NNInferResult infer_result;
  for (int i = 0; i < constants::kMaxNumMoves; ++i) {
    infer_result.move_logits[i] = move_logits(i);
  }
  for (int i = 0; i < constants::kMaxNumMoves; ++i) {
    infer_result.move_probs[i] = move_probs(i);
  }
  for (int i = 0; i < constants::kNumValueLogits; ++i) {
    infer_result.value_probs[i] = value_probs(i);
  }
  for (int i = 0; i < constants::kMaxNumBoardLocs; ++i) {
    infer_result.ownership[i] = ownership(i);
  }
  for (int i = 0; i < constants::kNumScoreLogits; ++i) {
    infer_result.score_probs[i] = score_probs(i);
  }
  return infer_result;
}

void NNInterface::RegisterThread(int thread_id) {
  absl::MutexLock l(&mu_);
  auto& thread = thread_info_[thread_id];
  if (thread.registered) {
    LOG(WARNING) << "Thread " << thread_id << " already registered.";
    return;
  }

  thread.registered = true;
  thread.loaded = false;
  ++num_threads_;
}

void NNInterface::UnregisterThread(int thread_id) {
  absl::MutexLock l(&mu_);
  auto& thread = thread_info_[thread_id];
  if (!thread.registered) {
    LOG(WARNING) << "Thread " << thread_id << " already unregistered.";
    return;
  }

  --num_threads_;
  thread.registered = false;
}

void NNInterface::InferLoop() {
  while (running_.load(std::memory_order_acquire)) {
    Infer();
  }
}

void NNInterface::Infer() {
  absl::MutexLock l(&mu_, absl::Condition(this, &NNInterface::ShouldInfer));
  if (num_threads_ == 0) {
    return;
  }

  // Cast to half.
  std::vector<std::pair<std::string, Tensor>> cast_input = {
      {kInput32BoardName, input_feature_buf_},
      {kInput32GameName, input_state_buf_}};
  TF_CHECK_OK(session_cast_input_->Run(
      cast_input, {kInput16BoardName, kInput16GameName}, {}, &nn_input_buf_));

  // Run Inference.
  std::vector<std::pair<std::string, Tensor>> nn_input = {
      {kInputNames[0], nn_input_buf_[0]}, {kInputNames[1], nn_input_buf_[1]}};
  TF_CHECK_OK(model_bundle_.GetSession()->Run(nn_input, kOutputNames, {},
                                              &nn_output_buf_));

  // cast back to float
  std::vector<std::pair<std::string, Tensor>> result_input = {
      {kOut16MoveLogitsName, nn_output_buf_[kMoveIndex]},
      {kOut16ValueLogitsName, nn_output_buf_[kValueIndex]},
      {kOut16OwnerLogitsName, nn_output_buf_[kOwnerIndex]},
      {kOut16ScoreLogitsName, nn_output_buf_[kScoreIndex]}};
  TF_CHECK_OK(session_cast_output_->Run(
      result_input,
      {kOut32MoveLogitsName, kOut32ValueProbsName, kOut32OwnerLogitsName,
       kOut32ScoreProbsName, kOut32MoveProbsName},
      {}, &result_buf_));

  // reset input buffers
  input_feature_buf_.flat<float>().setZero();
  input_state_buf_.flat<float>().setZero();

  for (auto& thread : thread_info_) {
    if (thread.registered) {
      thread.loaded = false;
      thread.res_ready = true;
    }
  }

  last_infer_time_ = std::chrono::steady_clock::now();
}

bool NNInterface::ShouldInfer() const {
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      now - last_infer_time_);
  if (duration.count() >= kTimeoutUs) {
    return true;
  }

  for (int thread_id = 0; thread_id < thread_info_.size(); ++thread_id) {
    const auto& thread = thread_info_[thread_id];
    if (thread.registered && !thread.loaded) {
      return false;
    }
  }

  return true;
}

}  // namespace nn
