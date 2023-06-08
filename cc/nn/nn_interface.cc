#include "cc/nn/nn_interface.h"

#include <stdlib.h>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "cc/nn/create_tensor_shape.h"
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
using ::game::Color;
using ::game::Game;
using ::game::Symmetry;

// 2 ** 20. Assuming ~3kb per inference result, this will be about ~3GB of RAM.
static constexpr size_t kMaxCacheSize = 1048576;

static constexpr int kPolicyLogitIndex = 0;
static constexpr int kPolicyProbIndex = 1;
static constexpr int kValueIndex = 2;
static constexpr int kScoreIndex = 3;

// Pre-processing feed/fetch names.
static constexpr char kInputBoardName[] = "input_board_state";
static constexpr char kInputGameName[] = "input_game_state";
static constexpr char kNNInputBoardName[] = "nn_input_board_state";
static constexpr char kNNInputGameName[] = "nn_input_game_state";

// Post-processing feed/fetch names.
static constexpr char kNNOutMoveLogitsName[] = "nn_out_move_logits";
static constexpr char kNNOutValueLogitsName[] = "nn_out_value_logits";
static constexpr char kNNOutScoreLogitsName[] = "nn_out_score_logits";

static constexpr char kOutMoveLogitsName[] = "out_move_logits";
static constexpr char kOutMoveProbsName[] = "out_move_probs";
static constexpr char kOutValueProbsName[] = "out_value_probs";
static constexpr char kOutScoreProbsName[] = "out_score_probs";

static constexpr char kSavedModelTagServe[] = "serve";

}  // namespace

NNInterface::Cache::Cache(int num_threads)
    : num_threads_(num_threads),
      thread_cache_size_(num_threads == 0 ? 0 : kMaxCacheSize / num_threads) {
  for (int i = 0; i < num_threads; ++i) {
    cache_[i].resize(thread_cache_size_);
  }
}

void NNInterface::Cache::Insert(int thread_id, const CacheKey& cache_key,
                                const NNInferResult& infer_result) {
  size_t hash = absl::HashOf(cache_key);
  size_t tbl_index = hash % thread_cache_size_;
  cache_[thread_id][tbl_index] = CacheElem{hash, infer_result};
}

bool NNInterface::Cache::Contains(int thread_id, const CacheKey& cache_key) {
  size_t hash = absl::HashOf(cache_key);
  size_t tbl_index = hash % thread_cache_size_;
  const std::optional<CacheElem>& elem = cache_[thread_id][tbl_index];
  if (!elem) {
    return false;
  } else if (elem->hash != hash) {
    return false;
  }

  return true;
}

std::optional<NNInferResult> NNInterface::Cache::Get(
    int thread_id, const CacheKey& cache_key) {
  size_t hash = absl::HashOf(cache_key);
  size_t tbl_index = hash % thread_cache_size_;
  const std::optional<CacheElem>& elem = cache_[thread_id][tbl_index];
  if (!elem) {
    return {};
  }

  return elem->infer_res;
}

NNInterface::NNInterface(int num_threads)
    : session_options_(SessionOptions()),
      run_options_(RunOptions()),
      scope_preprocess_(Scope::NewRootScope()),
      scope_postprocess_(Scope::NewRootScope()),
      session_preprocess_(tensorflow::NewSession(session_options_)),
      session_postprocess_(tensorflow::NewSession(session_options_)),
      is_initialized_(false),
      num_registered_threads_(num_threads),
      num_threads_(num_threads),
      nn_cache_(num_threads_),
      thread_info_(num_threads_),
      running_(true),
      last_infer_time_(
          std::chrono::time_point<std::chrono::steady_clock>::max()) {
  input_feature_buf_ =
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, BOARD_LEN, BOARD_LEN,
                                constants::kNumInputFeaturePlanes}));
  input_state_buf_ =
      Tensor(DataType::DT_FLOAT, CreateTensorShape({num_threads_, 1}));
  nn_input_buf_ = {
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, BOARD_LEN, BOARD_LEN,
                                constants::kNumInputFeaturePlanes})),
      Tensor(DataType::DT_HALF, CreateTensorShape({num_threads_, 1}))};

  input_feature_buf_.flat<float>().setZero();
  input_state_buf_.flat<float>().setZero();

  nn_output_buf_ = {
      // move logits
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kMaxNumMoves})),
      // win percent
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kNumValueLogits})),
      // ownership
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, BOARD_LEN, BOARD_LEN, 1})),
      // score prediction
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kNumScoreLogits})),
      // gamma, just ignore
      Tensor(DataType::DT_FLOAT, CreateTensorShape({num_threads_, 1}))};

  result_buf_ = {
      // move logits
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kMaxNumMoves})),
      // move softmax
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kMaxNumMoves})),
      // win percent
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kNumValueLogits})),
      // score prediction
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kNumScoreLogits}))};
}

NNInterface::~NNInterface() {
  running_.store(false, std::memory_order_release);

  if (infer_thread_.joinable()) {
    infer_thread_.join();
  }

  if (session_preprocess_) {
    session_preprocess_->Close().IgnoreError();
  }

  if (session_postprocess_) {
    session_postprocess_->Close().IgnoreError();
  }
}

absl::Status NNInterface::Initialize(std::string&& model_path) {
  auto status = LoadSavedModel(session_options_, run_options_, model_path,
                               {kSavedModelTagServe}, &model_bundle_);
  if (!status.ok()) {
    return ToAbslStatus(status);
  }

  // Create graph to cast game state to half precision.
  auto board = ops::Placeholder(scope_preprocess_.WithOpName(kInputBoardName),
                                DataType::DT_FLOAT);
  auto game = ops::Placeholder(scope_preprocess_.WithOpName(kInputGameName),
                               DataType::DT_FLOAT);
  auto nn_board =
      ops::Identity(scope_preprocess_.WithOpName(kNNInputBoardName), board);
  auto nn_game = ops::Cast(scope_preprocess_.WithOpName(kNNInputGameName), game,
                           DataType::DT_HALF);

  TF_CHECK_OK(scope_preprocess_.ToGraphDef(&gdef_preprocess_));
  TF_CHECK_OK(session_preprocess_->Create(gdef_preprocess_));

  // Create graph to wrangle NN result into usable data.
  auto nn_move_logits = ops::Placeholder(
      scope_postprocess_.WithOpName(kNNOutMoveLogitsName), DataType::DT_FLOAT);
  auto nn_value_logits = ops::Placeholder(
      scope_postprocess_.WithOpName(kNNOutValueLogitsName), DataType::DT_FLOAT);
  auto nn_score_logits = ops::Placeholder(
      scope_postprocess_.WithOpName(kNNOutScoreLogitsName), DataType::DT_FLOAT);

  auto move_logits = ops::Identity(
      scope_postprocess_.WithOpName(kOutMoveLogitsName), nn_move_logits);
  auto move_probs = ops::Softmax(
      scope_postprocess_.WithOpName(kOutMoveProbsName), nn_move_logits);
  auto value_probs = ops::Softmax(
      scope_postprocess_.WithOpName(kOutValueProbsName), nn_value_logits);
  auto score_probs = ops::Softmax(
      scope_postprocess_.WithOpName(kOutScoreProbsName), nn_score_logits);

  TF_CHECK_OK(scope_postprocess_.ToGraphDef(&gdef_postprocess_));
  TF_CHECK_OK(session_postprocess_->Create(gdef_postprocess_));

  infer_thread_ = std::thread(&NNInterface::InferLoop, this);
  is_initialized_ = true;

  return absl::OkStatus();
}

absl::Status NNInterface::LoadBatch(int thread_id, const Game& game,
                                    Color color_to_move) {
  if (!is_initialized_) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "Need to initialize NNInterface before using inference.");
  }

  ThreadInfo& thread_info = thread_info_[thread_id];
  CacheKey cache_key = CacheKey{
      color_to_move,
      game.board().hash(),
  };

  if (nn_cache_.Contains(thread_id, cache_key)) {
    // Cached. Immediately signal that result is ready.
    absl::MutexLock l(&mu_);
    thread_info.res_ready = true;
    thread_info.res_cached = true;
    thread_info.symmetry = Symmetry::kIdentity;
    thread_info.cache_key = cache_key;

    return absl::OkStatus();
  }

  // Not cached. Load for inference.
  DCHECK(game.moves().size() >= constants::kNumLastMoves);
  board_utils::FillNNInput(thread_id, num_threads_, input_feature_buf_,
                           input_state_buf_, game, color_to_move);

  absl::MutexLock l(&mu_);
  thread_info.loaded_for_inference = true;
  thread_info.res_ready = false;  // Force thread to wait for inference.
  thread_info.res_cached = false;
  thread_info.symmetry = Symmetry::kIdentity;
  thread_info.cache_key = cache_key;

  return absl::OkStatus();
}

NNInferResult NNInterface::GetInferenceResult(int thread_id) {
  auto& thread_info = thread_info_[thread_id];
  mu_.LockWhen(absl::Condition(&thread_info_[thread_id].res_ready));
  thread_info.res_ready = false;
  mu_.Unlock();

  if (thread_info.res_cached) {
    // Fetch from cache.
    DCHECK(nn_cache_.Contains(thread_id, thread_info.cache_key));
    return nn_cache_.Get(thread_id, thread_info.cache_key).value();
  }

  // Not found in cache. Fetch from NN output buffer.
  const auto move_logits = result_buf_[kPolicyLogitIndex]
                               .SubSlice(thread_id)
                               .unaligned_flat<float>();
  const auto move_probs =
      result_buf_[kPolicyProbIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto value_probs =
      result_buf_[kValueIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto score_probs =
      result_buf_[kScoreIndex].SubSlice(thread_id).unaligned_flat<float>();

  // Need hand-rolled for loops b/c of potential alignment issues.
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
  for (int i = 0; i < constants::kNumScoreLogits; ++i) {
    infer_result.score_probs[i] = score_probs(i);
  }

  // Cache this result.
  nn_cache_.Insert(thread_id, thread_info.cache_key, infer_result);
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
  thread.loaded_for_inference = false;
  ++num_registered_threads_;
}

void NNInterface::UnregisterThread(int thread_id) {
  absl::MutexLock l(&mu_);
  auto& thread = thread_info_[thread_id];
  if (!thread.registered) {
    LOG(WARNING) << "Thread " << thread_id << " already unregistered.";
    return;
  }

  --num_registered_threads_;
  thread.registered = false;
}

void NNInterface::InferLoop() {
  while (running_.load(std::memory_order_acquire)) {
    Infer();
  }
}

void NNInterface::Infer() {
  absl::MutexLock l(&mu_, absl::Condition(this, &NNInterface::ShouldInfer));
  if (num_registered_threads_ == 0) {
    return;
  }

  // Pre-process (cast game state to half).
  std::vector<std::pair<std::string, Tensor>> preprocess_input = {
      {kInputBoardName, input_feature_buf_},
      {kInputGameName, input_state_buf_}};
  TF_CHECK_OK(session_preprocess_->Run(preprocess_input,
                                       {kNNInputBoardName, kNNInputGameName},
                                       {}, &nn_input_buf_));

  // Run Inference.
  std::vector<std::pair<std::string, Tensor>> nn_input = {
      {kInputNames[0], nn_input_buf_[0]}, {kInputNames[1], nn_input_buf_[1]}};
  TF_CHECK_OK(model_bundle_.GetSession()->Run(nn_input, kOutputNames, {},
                                              &nn_output_buf_));

  // Post-process (softmax for moves, value, score).
  // Keep indices into nn_output_buf_ consistent with order of fetch names from
  // model.
  std::vector<std::pair<std::string, Tensor>> postprocess_input = {
      {kNNOutMoveLogitsName, nn_output_buf_[kNNPolicyIndex]},
      {kNNOutValueLogitsName, nn_output_buf_[kNNOutcomeIndex]},
      {kNNOutScoreLogitsName, nn_output_buf_[kNNScoreIndex]}};
  TF_CHECK_OK(
      session_postprocess_->Run(postprocess_input,
                                // Keep the order consistent with kOut*Index
                                {kOutMoveLogitsName, kOutMoveProbsName,
                                 kOutValueProbsName, kOutScoreProbsName},
                                {}, &result_buf_));

  // reset input buffers
  input_feature_buf_.flat<float>().setZero();
  input_state_buf_.flat<float>().setZero();

  for (auto& thread : thread_info_) {
    if (thread.registered && !thread.res_cached) {
      thread.res_ready = true;
      thread.loaded_for_inference = false;
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
    if (thread.registered && !thread.res_cached &&
        !thread.loaded_for_inference) {
      return false;
    }
  }

  return true;
}

}  // namespace nn
