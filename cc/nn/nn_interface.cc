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

static constexpr auto kMoveIndex = 0;
static constexpr auto kValueIndex = 1;
static constexpr auto kOwnerIndex = 2;
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

// input feed and fetch names for F32 -> F16 cast.
static constexpr char kInputBoardName[] = "input_board_state";
static constexpr char kInputGameName[] = "input_board_game";

static constexpr char kInputBoardHalfName[] = "input_board_state_half";
static constexpr char kInputGameHalfName[] = "input_board_game_half";

// output feed and fetch names for F16 -> F32 cast.
static constexpr char kOutputMoveLogitsName[] = "output_move_logits_half";
static constexpr char kOutputValueLogitsName[] = "output_value_logits_half";
static constexpr char kOutputOwnerLogitsName[] = "output_owner_logits_half";
static constexpr char kOutputScoreLogitsName[] = "output_score_logits_half";

static constexpr char kResultMoveLogitsName[] = "result_move_logits";
static constexpr char kResultValueProbsName[] = "result_value_probs";
static constexpr char kResultOwnerLogitsName[] = "result_owner_logits";
static constexpr char kResultScoreProbsName[] = "result_score_probs";
static constexpr char kResultMoveProbsName[] = "result_move_probs";

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
      load_counter_(std::make_unique<absl::BlockingCounter>(num_threads)),
      registered_(std::vector<uint8_t>(num_threads_, static_cast<char>(true))),
      batch_ready_(
          std::vector<uint8_t>(num_threads_, static_cast<char>(false))),
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
  auto board = ops::Placeholder(scope_cast_input_.WithOpName(kInputBoardName),
                                DataType::DT_FLOAT);
  auto game = ops::Placeholder(scope_cast_input_.WithOpName(kInputGameName),
                               DataType::DT_FLOAT);
  auto cast_board = ops::Cast(scope_cast_input_.WithOpName(kInputBoardHalfName),
                              board, DataType::DT_HALF);
  auto cast_game = ops::Cast(scope_cast_input_.WithOpName(kInputGameHalfName),
                             game, DataType::DT_HALF);

  TF_CHECK_OK(scope_cast_input_.ToGraphDef(&gdef_cast_input_));
  TF_CHECK_OK(session_cast_input_->Create(gdef_cast_input_));

  // Create graph to cast output to float.
  auto move_logits = ops::Placeholder(
      scope_cast_output_.WithOpName(kOutputMoveLogitsName), DataType::DT_HALF);
  auto value_logits = ops::Placeholder(
      scope_cast_output_.WithOpName(kOutputValueLogitsName), DataType::DT_HALF);
  auto owner_logits = ops::Placeholder(
      scope_cast_output_.WithOpName(kOutputOwnerLogitsName), DataType::DT_HALF);
  auto score_logits = ops::Placeholder(
      scope_cast_output_.WithOpName(kOutputScoreLogitsName), DataType::DT_HALF);

  auto cast_move_logits =
      ops::Cast(scope_cast_output_.WithOpName(kResultMoveLogitsName),
                move_logits, DataType::DT_FLOAT);
  auto value_probs = ops::Softmax(
      scope_cast_output_.WithOpName(kResultValueProbsName),
      ops::Cast(scope_cast_output_, value_logits, DataType::DT_FLOAT));
  auto cast_owner_logits =
      ops::Cast(scope_cast_output_.WithOpName(kResultOwnerLogitsName),
                owner_logits, DataType::DT_FLOAT);
  auto score_probs = ops::Softmax(
      scope_cast_output_.WithOpName(kResultScoreProbsName),
      ops::Cast(scope_cast_output_, score_logits, DataType::DT_FLOAT));
  auto move_probs = ops::Softmax(
      scope_cast_output_.WithOpName(kResultMoveProbsName),
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

  load_counter_->DecrementCount();
  return absl::OkStatus();
}

NNInferResult NNInterface::GetInferenceResult(int thread_id) {
  mu_.LockWhen(absl::Condition(
      reinterpret_cast<bool*>(batch_ready_.data() + thread_id)));
  batch_ready_[thread_id] = false;
  mu_.Unlock();

  const auto move_logits =
      result_buf_[kMoveIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto value_probability =
      result_buf_[kValueIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto ownership =
      result_buf_[kOwnerIndex].SubSlice(thread_id).unaligned_flat<float>();
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

void NNInterface::RegisterThread(int thread_id) {
  absl::MutexLock l(&mu_);
  if (registered_[thread_id]) {
    LOG(WARNING) << "Thread " << thread_id << " already registered.";
    return;
  }

  // Note that after registering this thread, the first inference result
  // the thread fetches may be invalid, as the thread cannot guarantee that it
  // loads its batch in time for inference.
  num_threads_++;
  registered_[thread_id] = true;
}

void NNInterface::UnregisterThread(int thread_id) {
  absl::MutexLock l(&mu_);
  if (!registered_[thread_id]) {
    LOG(WARNING) << "Thread " << thread_id << " already unregistered.";
    return;
  }

  num_threads_--;
  load_counter_->DecrementCount();
  registered_[thread_id] = false;
}

void NNInterface::InferLoop() {
  while (running_.load(std::memory_order_acquire)) {
    Infer();
  }
}

void NNInterface::Infer() {
  load_counter_->Wait();
  absl::MutexLock l(&mu_);

  // Cast to half.
  std::vector<std::pair<std::string, Tensor>> cast_input = {
      {kInputBoardName, input_feature_buf_},
      {kInputGameName, input_state_buf_}};
  TF_CHECK_OK(session_cast_input_->Run(
      cast_input, {kInputBoardHalfName, kInputGameHalfName}, {},
      &nn_input_buf_));

  // Run Inference.
  std::vector<std::pair<std::string, Tensor>> nn_input = {
      {kInputNames[0], nn_input_buf_[0]}, {kInputNames[1], nn_input_buf_[1]}};
  TF_CHECK_OK(model_bundle_.GetSession()->Run(nn_input, kOutputNames, {},
                                              &nn_output_buf_));

  // cast back to float
  std::vector<std::pair<std::string, Tensor>> result_input = {
      {kOutputMoveLogitsName, nn_output_buf_[kMoveIndex]},
      {kOutputValueLogitsName, nn_output_buf_[kValueIndex]},
      {kOutputOwnerLogitsName, nn_output_buf_[kOwnerIndex]},
      {kOutputScoreLogitsName, nn_output_buf_[kScoreIndex]}};
  TF_CHECK_OK(session_cast_output_->Run(
      result_input,
      {kResultMoveLogitsName, kResultValueProbsName, kResultOwnerLogitsName,
       kResultScoreProbsName, kResultMoveProbsName},
      {}, &result_buf_));

  // reset input buffers
  input_feature_buf_.flat<float>().setZero();
  input_state_buf_.flat<float>().setZero();

  load_counter_ = std::make_unique<absl::BlockingCounter>(num_threads_);
  std::fill(batch_ready_.begin(), batch_ready_.end(), static_cast<char>(true));
}

}  // namespace nn