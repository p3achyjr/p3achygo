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

// 2 ** 20. Inference Result is 6kb, so this will be about ~6GB of RAM.
static constexpr char kSavedModelTagServe[] = "serve";

}  // namespace

NNInterface::NNInterface(int num_threads)
    : NNInterface(num_threads, kTimeoutUs, constants::kDefaultNNCacheSize) {}

NNInterface::NNInterface(int num_threads, int64_t timeout, size_t cache_size)
    : session_options_(SessionOptions()),
      run_options_(RunOptions()),
      is_initialized_(false),
      num_registered_threads_(num_threads),
      num_threads_(num_threads),
      thread_info_(num_threads_),
      running_(true),
      timeout_(timeout) {
  InitializeCache(cache_size);

  // Allocate inference buffers.
  nn_input_buf_ = {
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, BOARD_LEN, BOARD_LEN,
                                constants::kNumInputFeaturePlanes})),
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape(
                 {num_threads_, constants::kNumInputFeatureScalars}))};

  nn_input_buf_[0].flat<float>().setZero();
  nn_input_buf_[1].flat<float>().setZero();

  nn_output_buf_ = {
      // move logits
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kMaxNumMoves})),
      // q30
      Tensor(DataType::DT_FLOAT, CreateTensorShape({num_threads_, 1})),
      // q100
      Tensor(DataType::DT_FLOAT, CreateTensorShape({num_threads_, 1})),
      // q200
      Tensor(DataType::DT_FLOAT, CreateTensorShape({num_threads_, 1})),
      // move softmax
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kMaxNumMoves})),
      // win logits
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kNumValueLogits})),
      // win percent
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kNumValueLogits})),
      // ownership
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, BOARD_LEN, BOARD_LEN, 1})),
      // score logits
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kNumScoreLogits})),
      // score probabilities
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kNumScoreLogits})),
      // gamma, just ignore
      Tensor(DataType::DT_FLOAT, CreateTensorShape({num_threads_, 1})),
      // auxiliary move logits
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({num_threads_, constants::kMaxNumMoves}))};
}

NNInterface::~NNInterface() {
  running_.store(false, std::memory_order_release);

  if (infer_thread_.joinable()) {
    infer_thread_.join();
  }
}

absl::Status NNInterface::Initialize(std::string&& model_path) {
  auto status = LoadSavedModel(session_options_, run_options_, model_path,
                               {kSavedModelTagServe}, &model_bundle_);
  if (!status.ok()) {
    return ToAbslStatus(status);
  }

  infer_thread_ = std::thread(&NNInterface::InferLoop, this);
  is_initialized_ = true;

  return absl::OkStatus();
}

NNInferResult NNInterface::LoadAndGetInference(int thread_id, const Game& game,
                                               Color color_to_move,
                                               Probability& probability) {
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
  Symmetry sym = GetRandomSymmetry(probability.prng());
  board_utils::FillNNInput(thread_id, num_threads_, nn_input_buf_[0],
                           nn_input_buf_[1], game, color_to_move, sym);

  mu_.Lock();
  thread_info.loaded_for_inference = true;
  thread_info.res_ready = false;
  thread_info.res_cached = false;

  // Wait for result.
  mu_.Await(absl::Condition(&thread_info.res_ready));
  thread_info.res_ready = false;
  mu_.Unlock();

  // Inference result is now ready.
  const auto move_logits = nn_output_buf_[kPiLogitsIndex]
                               .SubSlice(thread_id)
                               .unaligned_flat<float>();
  const auto move_probs =
      nn_output_buf_[kPiProbsIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto value_probs =
      nn_output_buf_[kOutcomeIndex].SubSlice(thread_id).unaligned_flat<float>();
  const auto score_probs = nn_output_buf_[kScoreProbsIndex]
                               .SubSlice(thread_id)
                               .unaligned_flat<float>();

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
      ApplyInverse(sym, grid_logits_sym, BOARD_LEN);
  std::array<float, constants::kMaxNumBoardLocs> grid_probs =
      ApplyInverse(sym, grid_probs_sym, BOARD_LEN);
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

void NNInterface::InitializeCache(size_t cache_size) {
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_caches_[thread_id] =
        Cache<NNKey, NNInferResult>(cache_size / num_threads_);
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
  if (num_registered_threads_ == 0) {
    mu_.Unlock();
    return;
  }

  // Run Inference.
  std::vector<std::pair<std::string, Tensor>> nn_input = {
      {kInputNames[0], nn_input_buf_[0]}, {kInputNames[1], nn_input_buf_[1]}};
  TF_CHECK_OK(model_bundle_.GetSession()->Run(nn_input, kOutputNames, {},
                                              &nn_output_buf_));

  // reset input buffers.
  nn_input_buf_[0].flat<float>().setZero();
  nn_input_buf_[1].flat<float>().setZero();

  for (auto& thread : thread_info_) {
    if (thread.registered && !thread.res_cached) {
      thread.res_ready = true;
      thread.loaded_for_inference = false;
    }
  }

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
