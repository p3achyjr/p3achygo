#include "cc/nn/nn_interface.h"

#include <stdlib.h>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "cc/game/symmetry.h"
#include "cc/nn/engine/go_features.h"

namespace nn {
namespace {
using namespace ::core;
using namespace ::game;

// 2 ** 20. Inference Result is 6kb, so this will be about ~6GB of RAM.
static constexpr char kSavedModelTagServe[] = "serve";

}  // namespace

NNInterface::NNInterface(int num_threads, std::unique_ptr<Engine> engine)
    : NNInterface(num_threads, kTimeoutUs, constants::kDefaultNNCacheSize,
                  std::move(engine)) {}

NNInterface::NNInterface(int num_threads, int64_t timeout, size_t cache_size,
                         std::unique_ptr<Engine> engine)
    : num_registered_threads_(num_threads),
      num_threads_(num_threads),
      thread_info_(num_threads_),
      running_(true),
      timeout_(timeout),
      engine_(std::move(engine)) {
  InitializeCache(cache_size);
  if (num_threads_ > 1) {
    infer_thread_ = std::thread(&NNInterface::InferLoop, this);
  }

  LOG(INFO) << "NNInterface Initialized. Engine Type: `"
            << KindToString(engine_->kind()) << "`";
}

NNInterface::~NNInterface() {
  mu_.Lock();
  running_.store(false, std::memory_order_release);
  mu_.Unlock();

  if (infer_thread_.joinable()) {
    infer_thread_.join();
  }
}

NNInferResult NNInterface::LoadAndGetInference(int thread_id, const Game& game,
                                               Color color_to_move,
                                               Probability& probability) {
  NNKey cache_key = NNKey{
      color_to_move,
      game.board().hash(),
  };

  if (CacheContains(thread_id, cache_key)) {
    // Cached. Immediately return result.
    absl::MutexLock l(&mu_);
    thread_info_[thread_id].res_cached = true;

    return CacheGet(thread_id, cache_key).value();
  }

  // Not cached. Load for inference.
  DCHECK(game.moves().size() >= constants::kNumLastMoves);

  Symmetry sym = GetRandomSymmetry(probability.prng());
  LoadBatch(thread_id, game, color_to_move, sym);
  SignalLoadedAndBlockUntilReady(thread_id);

  // Inference result is now ready.
  NNInferResult infer_result;
  engine_->GetBatch(thread_id, infer_result);

  // We have finished retrieving data. This should not need to be lock-guarded.
  mu_.Lock();
  thread_info_[thread_id].res_ready = false;
  mu_.Unlock();

  // Unapply symmetry.
  std::array<float, constants::kNumBoardLocs> grid_logits_sym;
  std::array<float, constants::kNumBoardLocs> grid_probs_sym;
  std::copy(infer_result.move_logits.begin(),
            infer_result.move_logits.begin() + constants::kNumBoardLocs,
            grid_logits_sym.begin());
  std::copy(infer_result.move_probs.begin(),
            infer_result.move_probs.begin() + constants::kNumBoardLocs,
            grid_probs_sym.begin());
  std::array<float, constants::kNumBoardLocs> grid_logits =
      ApplyInverse(sym, grid_logits_sym, BOARD_LEN);
  std::array<float, constants::kNumBoardLocs> grid_probs =
      ApplyInverse(sym, grid_probs_sym, BOARD_LEN);
  std::copy(grid_logits.begin(), grid_logits.end(),
            infer_result.move_logits.begin());
  std::copy(grid_probs.begin(), grid_probs.end(),
            infer_result.move_probs.begin());

  // Cache this result.
  CacheInsert(thread_id, cache_key, infer_result);
  return infer_result;
}

std::array<float, constants::kNumBoardLocs> NNInterface::LoadAndGetOwnership(
    int thread_id, const Game& game, Color color_to_move) {
  LoadBatch(thread_id, game, color_to_move, Symmetry::kIdentity);
  SignalLoadedAndBlockUntilReady(thread_id);

  // Inference result is now ready.
  std::array<float, constants::kNumBoardLocs> ownership;
  engine_->GetOwnership(thread_id, ownership);
  return ownership;
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

void NNInterface::LoadBatch(int thread_id, const game::Game& game,
                            game::Color color_to_move, game::Symmetry sym) {
  GoFeatures input_features;
  input_features.bsize = BOARD_LEN;
  input_features.color = color_to_move;

  int num_moves = game.num_moves();
  for (int i = 0; i < constants::kNumLastMoves; ++i) {
    int mv_offset = num_moves - constants::kNumLastMoves + i;
    if (mv_offset < 0) {
      input_features.last_moves[i] = kNoopLoc;
      continue;
    } else if (game.move(mv_offset).loc == kPassLoc) {
      input_features.last_moves[i] = kPassLoc;
      continue;
    }
    input_features.last_moves[i] =
        ApplySymmetry(sym, game.move(mv_offset).loc, BOARD_LEN);
  }

  input_features.board = ApplySymmetry(sym, game.board().position(), BOARD_LEN);
  input_features.stones_atari =
      ApplySymmetry(sym, game.board().GetStonesInAtari(), BOARD_LEN);
  input_features.stones_two_liberties =
      ApplySymmetry(sym, game.board().GetStonesWithLiberties(2), BOARD_LEN);
  input_features.stones_three_liberties =
      ApplySymmetry(sym, game.board().GetStonesWithLiberties(3), BOARD_LEN);

  engine_->LoadBatch(thread_id, input_features);
}

void NNInterface::InferLoop() {
  CHECK(num_threads_ > 1);
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

  // Do not run inference if there is a thread that still has not retrieved its
  // inference result.
  for (const ThreadInfo& thread : thread_info_) {
    if (thread.res_ready) {
      mu_.Unlock();
      return;
    }
  }

  // Run Inference.
  engine_->RunInference();

  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    ThreadInfo& thread = thread_info_[thread_id];
    if (thread.registered && thread.loaded_for_inference) {
      // If we do not check `thread.loaded_for_inference` before clearing input,
      // the following sequence of events is possible:
      // - worker_thread fills their NN Input.
      // - infer_thread acquires lock via timeout.
      // - infer_thread runs inference.
      // - worker_thread acquires lock.
      // - worker_thread marks `thread.loaded_for_inference = true`.
      // - infer_thread acquires lock and runs inference. The input is empty, so
      //   inference gives back bogus data.
      thread.res_ready = true;
      thread.loaded_for_inference = false;
    }
  }

  mu_.Unlock();
}

bool NNInterface::ShouldInfer() const {
  if (!running_.load(std::memory_order_acquire)) {
    return true;
  }

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
