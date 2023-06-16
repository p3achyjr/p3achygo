#include "cc/nn/nn_interface.h"

#include <stdlib.h>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "cc/game/symmetry.h"
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
using namespace ::core;
using namespace ::game;
using namespace ::tensorflow;

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

NNInterface::NNInterface(int num_threads)
    : NNInterface(num_threads, kTimeoutUs) {}

NNInterface::NNInterface(int num_threads, int64_t timeout)
    : session_options_(SessionOptions()),
      run_options_(RunOptions()),
      scope_preprocess_(Scope::NewRootScope()),
      scope_postprocess_(Scope::NewRootScope()),
      session_preprocess_(tensorflow::NewSession(session_options_)),
      session_postprocess_(tensorflow::NewSession(session_options_)),
      is_initialized_(false),
      num_registered_threads_(num_threads),
      num_threads_(num_threads),
      thread_info_(num_threads_),
      running_(true),
      timeout_(timeout) {
  InitializeCache();

  // Allocate inference buffers.
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

NNInferResult NNInterface::LoadAndGetInference(int thread_id, const Game& game,
                                               Color color_to_move) {
  DCHECK(is_initialized_);
  ThreadInfo& thread_info = thread_info_[thread_id];
  NNKey cache_key = NNKey{
      color_to_move,
      game.board().hash(),
  };

  if (CacheContains(thread_id, cache_key)) {
    // Cached. Immediately return result.
    absl::MutexLock l(&mu_);
    thread_info.res_cached = true;

    return CacheGet(thread_id, cache_key).value();
  }

  // Not cached. Load for inference.
  DCHECK(game.moves().size() >= constants::kNumLastMoves);
  Symmetry sym = GetRandomSymmetry(prng_);
  board_utils::FillNNInput(thread_id, num_threads_, input_feature_buf_,
                           input_state_buf_, game, color_to_move, sym);

  mu_.Lock();
  thread_info.loaded_for_inference = true;
  thread_info.res_ready = false;
  thread_info.res_cached = false;

  // Wait for result.
  mu_.Await(absl::Condition(&thread_info.res_ready));
  thread_info.res_ready = false;
  mu_.Unlock();

  // Inference result is now ready.
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

  // Unapply symmetry.
  std::array<float, constants::kMaxNumBoardLocs> grid_logits_sym;
  std::array<float, constants::kMaxNumBoardLocs> grid_probs_sym;
  std::copy(infer_result.move_logits.begin(),
            infer_result.move_logits.begin() + constants::kMaxNumBoardLocs,
            grid_logits_sym.begin());
  std::copy(infer_result.move_probs.begin(),
            infer_result.move_probs.begin() + constants::kMaxNumBoardLocs,
            grid_probs_sym.begin());
  std::array<float, constants::kMaxNumBoardLocs> grid_logits =
      ApplyInverse(sym, grid_logits_sym, game.board_len());
  std::array<float, constants::kMaxNumBoardLocs> grid_probs =
      ApplyInverse(sym, grid_probs_sym, game.board_len());
  std::copy(grid_logits.begin(), grid_logits.end(),
            infer_result.move_logits.begin());
  std::copy(grid_probs.begin(), grid_probs.end(),
            infer_result.move_probs.begin());

  // Cache this result.
  CacheInsert(thread_id, cache_key, infer_result);
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

void NNInterface::InitializeCache() {
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_caches_[thread_id] =
        Cache<NNKey, NNInferResult>(kMaxCacheSize / num_threads_);
  }
}

bool NNInterface::CacheContains(int thread_id, const NNKey& key) {
  return thread_caches_[thread_id].Contains(key);
}

std::optional<NNInferResult> NNInterface::CacheGet(int thread_id,
                                                   const NNKey& key) {
  return thread_caches_[thread_id].Get(key);
}

void NNInterface::CacheInsert(int thread_id, const NNKey& key,
                              const NNInferResult& result) {
  thread_caches_[thread_id].Insert(key, result);
}

void NNInterface::InferLoop() {
  while (running_.load(std::memory_order_acquire)) {
    Infer();
  }
}

void NNInterface::Infer() {
  mu_.LockWhenWithTimeout(absl::Condition(this, &NNInterface::ShouldInfer),
                          absl::Microseconds(timeout_));
  auto begin = std::chrono::steady_clock::now();
  if (num_registered_threads_ == 0) {
    mu_.Unlock();
    return;
  }

  // Pre-process (cast game state to half).
  std::vector<std::pair<std::string, Tensor>> preprocess_input = {
      {kInputBoardName, input_feature_buf_},
      {kInputGameName, input_state_buf_}};
  TF_CHECK_OK(session_preprocess_->Run(preprocess_input,
                                       {kNNInputBoardName, kNNInputGameName},
                                       {}, &nn_input_buf_));
  auto end_preprocess = std::chrono::steady_clock::now();

  // Run Inference.
  std::vector<std::pair<std::string, Tensor>> nn_input = {
      {kInputNames[0], nn_input_buf_[0]}, {kInputNames[1], nn_input_buf_[1]}};
  auto begin_graph = std::chrono::steady_clock::now();
  TF_CHECK_OK(model_bundle_.GetSession()->Run(nn_input, kOutputNames, {},
                                              &nn_output_buf_));
  auto end_graph = std::chrono::steady_clock::now();

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
  auto end = std::chrono::steady_clock::now();
  LOG_EVERY_N_SEC(INFO, 5)
      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
             .count()
      << "us for INFERENCE.";
  LOG_EVERY_N_SEC(INFO, 5)
      << std::chrono::duration_cast<std::chrono::microseconds>(end_preprocess -
                                                               begin)
             .count()
      << "us for PREPROCESS.";
  LOG_EVERY_N_SEC(INFO, 5)
      << std::chrono::duration_cast<std::chrono::microseconds>(end_graph -
                                                               begin_graph)
             .count()
      << "us for JUST GRAPH.";
  LOG_EVERY_N_SEC(INFO, 5)
      << std::chrono::duration_cast<std::chrono::microseconds>(end - end_graph)
             .count()
      << "us for POSTPROCESS.";

  mu_.Unlock();
}

bool NNInterface::ShouldInfer() const {
  // Only return true if at least one leaf evaluation is pending.
  bool exists_pending = false;
  for (int thread_id = 0; thread_id < thread_info_.size(); ++thread_id) {
    const auto& thread = thread_info_[thread_id];
    if (!thread.registered) continue;
    if (!thread.res_cached && !thread.loaded_for_inference) {
      return false;
    }

    if (!thread.res_cached) exists_pending = true;
  }

  return exists_pending;
}

}  // namespace nn
