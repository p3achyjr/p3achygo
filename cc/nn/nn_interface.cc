#include "cc/nn/nn_interface.h"

#include <stdlib.h>

#include <atomic>
#include <thread>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "cc/core/lru_cache.h"
#include "cc/game/symmetry.h"
#include "cc/nn/engine/go_features.h"

namespace nn {

namespace {
using namespace ::core;
using namespace ::game;

std::unique_ptr<Waker> MakeWaker(NNInterface::WakeStrategy strategy) {
  switch (strategy) {
    case NNInterface::WakeStrategy::kMutex:
      return std::make_unique<MutexWaker>();
    case NNInterface::WakeStrategy::kGenCounter:
    default:
      return std::make_unique<GenCounterWaker>();
  }
}

// 2 ** 20. Inference Result is 6kb, so this will be about ~6GB of RAM.
static constexpr char kSavedModelTagServe[] = "serve";

}  // namespace

NNInterface::NNInterface(int num_threads, std::unique_ptr<Engine> engine)
    : NNInterface(num_threads, kTimeoutUs, constants::kDefaultNNCacheSize,
                  std::move(engine), SignalKind::kAuto, -1) {}

NNInterface::NNInterface(int num_threads, int64_t timeout, size_t cache_size,
                         std::unique_ptr<Engine> engine)
    : NNInterface(num_threads, timeout, cache_size, std::move(engine),
                  SignalKind::kAuto, -1) {}

NNInterface::NNInterface(int num_threads, int64_t timeout, size_t cache_size,
                         std::unique_ptr<Engine> engine,
                         WakeStrategy wake_strategy)
    : NNInterface(num_threads, timeout, cache_size, std::move(engine),
                  SignalKind::kAuto, -1, wake_strategy) {}

NNInterface::NNInterface(int num_threads, std::unique_ptr<Engine> engine,
                         SignalKind signal_kind, int num_shared_search_tasks)
    : NNInterface(num_threads, kTimeoutUs, constants::kDefaultNNCacheSize,
                  std::move(engine), signal_kind, num_shared_search_tasks) {}

NNInterface::NNInterface(int num_threads, int64_t timeout, size_t cache_size,
                         std::unique_ptr<Engine> engine, SignalKind signal_kind,
                         int num_shared_search_tasks,
                         WakeStrategy wake_strategy)
    : num_registered_threads_(num_threads),
      num_threads_(num_threads),
      thread_info_(num_threads_),
      running_(true),
      timeout_(timeout),
      engine_(std::move(engine)),
      symmetries_(num_threads_),
      signal_kind_(signal_kind),
      num_shared_search_tasks_(num_shared_search_tasks),
      num_signaled_search_tasks_(0),
      num_exited_search_tasks_(0),
      waker_(MakeWaker(wake_strategy)) {
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

NNInterface::NNKey NNInterface::MakeKey(const game::Game& game,
                                        game::Color color_to_move) const {
  const int num_moves = game.num_moves();
  std::array<game::Loc, constants::kNumLastMoves> last_moves;
  last_moves.fill(kNoopLoc);
  for (int i = 0; i < num_cache_last_moves_; ++i) {
    const int off = num_moves - num_cache_last_moves_ + i;
    if (off >= 0) {
      last_moves[constants::kNumLastMoves - num_cache_last_moves_ + i] =
          game.move(off).loc;
    }
  }
  return NNKey{color_to_move, game.board().hash(), last_moves, game.komi()};
}

NNInferResult NNInterface::LoadAndGetInference(int thread_id, const Game& game,
                                               Color color_to_move,
                                               Probability& probability) {
  NNKey cache_key = MakeKey(game, color_to_move);

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

  NNInferResult infer_result = GetBatch(thread_id, sym);

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

void NNInterface::LoadEntry(int thread_id, int offset, const game::Game& game,
                            game::Color color_to_move,
                            core::Probability& probability) {
  const int canonical_tid = thread_id + offset;
  NNKey cache_key = MakeKey(game, color_to_move);

  if (CacheContains(canonical_tid, cache_key)) {
    // Cached. Do not need to wait for this thread.
    absl::MutexLock l(&mu_);
    thread_info_[canonical_tid].res_cached = true;
    return;
  }

  // Not cached. Load for inference.
  DCHECK(game.moves().size() >= constants::kNumLastMoves);

  Symmetry sym = GetRandomSymmetry(probability.prng());
  symmetries_[canonical_tid] = sym;
  LoadBatch(canonical_tid, game, color_to_move, sym);

  // signal loaded.
  {
    absl::MutexLock l(&mu_);
    ThreadInfo& thread_info = thread_info_[canonical_tid];
    thread_info.loaded_for_inference = true;
    thread_info.res_ready.store(false, std::memory_order_relaxed);
    thread_info.res_cached = false;
  }
}

NNInferResult NNInterface::FetchEntry(int thread_id, int offset,
                                      const game::Game& game,
                                      game::Color color_to_move) {
  const int canonical_tid = thread_id + offset;
  NNKey cache_key = MakeKey(game, color_to_move);

  ThreadInfo& thread_info = thread_info_[canonical_tid];
  if (CacheContains(canonical_tid, cache_key)) {
    // Cached. Return cache entry.
    absl::MutexLock l(&mu_);
    thread_info.res_cached = false;
    return CacheGet(canonical_tid, cache_key).value();
  }

  waker_->Wait(thread_info.res_ready, mu_);

  NNInferResult infer_result =
      GetBatch(canonical_tid, symmetries_[canonical_tid]);

  // Cache this result.
  CacheInsert(canonical_tid, cache_key, infer_result);
  return infer_result;
}

void NNInterface::InitializeCache(size_t cache_size) {
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_caches_[thread_id] =
        LRUCache<NNKey, NNInferResult>(cache_size / num_threads_);
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
  input_features.stones_laddered =
      ApplySymmetry(sym, game.board().GetLadderedStones(), BOARD_LEN);
  input_features.komi = game.komi();

  engine_->LoadBatch(thread_id, input_features);
}

void NNInterface::InferLoop() {
  CHECK(num_threads_ > 1);
  while (running_.load(std::memory_order_acquire)) {
    Infer();
  }
}

void NNInterface::Infer() {
  struct RAIIFlagReset {
    NNInterface* self = nullptr;
    RAIIFlagReset() = delete;
    explicit RAIIFlagReset(NNInterface* self) : self(self) {};
    ~RAIIFlagReset() { self->num_signaled_search_tasks_ = 0; }
  };
  // Always use a timeout (both kAuto and kExplicit).
  //
  // kExplicit would otherwise deadlock in eval: cur_nn is shared by 16 games
  // where cur plays black and 16 where cur plays white, so at most half the
  // games signal per ply and ShouldInfer() (which requires all tasks to signal)
  // never returns true.
  //
  // With the timeout, inference fires with whatever is loaded. Correctness is
  // preserved because:
  //   1. LoadEntry() sets loaded_for_inference before
  //   SignalReadyForInference().
  //   2. Infer() only sets res_ready for slots with loaded_for_inference =
  //   true,
  //      so workers whose LoadEntry hasn't run yet are skipped and wait for the
  //      next inference cycle.
  //   3. FetchEntry() blocks on res_ready, so it reads GetBatch(task_offset +
  //      worker_id) only after that exact slot's data was processed — the
  //      queue/fetch pairing is never violated.
  //   4. SearchTask's Barrier 1 (descent_remaining==0 && pending==0) still
  //      sequences correctly even if a game's workers are split across two
  //      inference cycles: the barrier only opens once every worker has both
  //      finished descending (LoadEntry called) and fetched its eval result.
  if (timeout_ > 0) {
    mu_.LockWhenWithTimeout(absl::Condition(this, &NNInterface::ShouldInfer),
                            absl::Microseconds(timeout_));
  } else {
    mu_.LockWhen(absl::Condition(this, &NNInterface::ShouldInfer));
  }

  RAIIFlagReset r(this);
  if (num_registered_threads_ == 0) {
    mu_.Unlock();
    return;
  }

  // Do not run inference if any thread still has an unread result: running
  // RunInference() would overwrite the engine output buffer before the worker
  // calls GetBatch(). Apply to both kAuto and kExplicit.
  for (const ThreadInfo& thread : thread_info_) {
    if (thread.res_ready.load(std::memory_order_acquire)) {
      mu_.Unlock();
      return;
    }
  }

  // Timeout may fire before any leaf has been loaded. Skip rather than running
  // the engine on stale input slots (the outputs would be discarded anyway
  // since loaded_for_inference would be false for every slot).
  bool any_loaded = false;
  for (const ThreadInfo& thread : thread_info_) {
    if (thread.loaded_for_inference) {
      any_loaded = true;
      break;
    }
  }
  if (!any_loaded) {
    mu_.Unlock();
    return;
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
      thread.res_ready.store(true, std::memory_order_release);
      thread.loaded_for_inference = false;
    }
    thread.res_cached = false;
  }

  // Notify all waiting workers. mu_ is still held here; MutexWaker is a no-op
  // (unlock below triggers Await re-evaluation); GenCounterWaker bumps the
  // generation counter and calls notify_all() — one FUTEX_WAKE for all.
  waker_->NotifyAll();
  mu_.Unlock();
}

bool NNInterface::ShouldInfer() const {
  if (!running_.load(std::memory_order_acquire)) {
    return true;
  }

  if (signal_kind_ == SignalKind::kExplicit) {
    const int remaining = num_shared_search_tasks_ - num_exited_search_tasks_;
    if (remaining <= 0) return false;  // All tasks done; wait for destructor.
    return num_signaled_search_tasks_ == remaining;
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
