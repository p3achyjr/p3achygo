#include "cc/nn/nn_interface.h"

#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
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
      symmetries_(num_threads_),
      signal_kind_(signal_kind),
      num_shared_search_tasks_(num_shared_search_tasks) {
  num_waves_ = 1;
  wave_size_ = num_threads_;
  engines_.push_back(std::move(engine));
  wakers_.push_back(MakeWaker(wake_strategy));
  InitializeCache(cache_size);
  if (num_threads_ > 1) {
    infer_thread_ = std::thread(&NNInterface::InferLoop, this);
  }

  LOG(INFO) << "NNInterface Initialized. Engine Type: `"
            << KindToString(engines_[0]->kind()) << "`";
}

NNInterface::NNInterface(int num_threads, int64_t timeout, size_t cache_size,
                         std::vector<std::unique_ptr<Engine>> wave_engines,
                         SignalKind signal_kind, int num_shared_search_tasks,
                         WakeStrategy wake_strategy, int num_infer_threads)
    : num_registered_threads_(num_threads),
      num_threads_(num_threads),
      thread_info_(num_threads_),
      running_(true),
      timeout_(timeout),
      symmetries_(num_threads_),
      signal_kind_(signal_kind),
      num_shared_search_tasks_(num_shared_search_tasks) {
  num_waves_ = static_cast<int>(wave_engines.size());
  CHECK(num_waves_ >= 2 && num_waves_ <= kMaxWaves) << num_waves_;
  CHECK(num_threads_ % num_waves_ == 0 && num_threads_ / num_waves_ > 1)
      << num_threads_ << " threads, " << num_waves_ << " waves";
  CHECK(num_infer_threads == 1 || num_infer_threads == num_waves_)
      << num_infer_threads;
  wave_size_ = num_threads_ / num_waves_;
  engines_ = std::move(wave_engines);
  for (int w = 0; w < num_waves_; ++w) {
    wakers_.push_back(MakeWaker(wake_strategy));
  }
  InitializeCache(cache_size);
  if (num_infer_threads == 1) {
    wave_infer_threads_.emplace_back(&NNInterface::InferWavesLoop, this, -1);
  } else {
    for (int w = 0; w < num_waves_; ++w) {
      wave_infer_threads_.emplace_back(&NNInterface::InferWavesLoop, this, w);
    }
  }

  LOG(INFO) << "NNInterface Initialized (shared, " << num_waves_ << " waves, "
            << num_infer_threads << " infer threads). Engine Type: `"
            << KindToString(engines_[0]->kind()) << "`";
}

NNInterface::NNInterface(int num_threads, size_t cache_size,
                         std::unique_ptr<Engine> engine, int engine_batch,
                         int min_fill, int64_t min_fill_timeout_us,
                         WakeStrategy wake_strategy)
    : num_registered_threads_(num_threads),
      num_threads_(num_threads),
      thread_info_(num_threads_),
      running_(true),
      timeout_(0),
      symmetries_(num_threads_),
      signal_kind_(SignalKind::kExplicit),
      num_shared_search_tasks_(1) {
  CHECK(num_threads_ > 1) << num_threads_;
  CHECK(engine_batch > 0 && min_fill <= engine_batch)
      << engine_batch << " " << min_fill;
  num_waves_ = 1;
  wave_size_ = num_threads_;
  queue_mode_ = true;
  q_engine_batch_ = engine_batch;
  q_min_fill_ = std::max(1, min_fill);
  q_min_fill_timeout_us_ = min_fill_timeout_us;
  q_staged_.resize(num_threads_);
  q_results_.resize(num_threads_);
  engines_.push_back(std::move(engine));
  wakers_.push_back(MakeWaker(wake_strategy));
  InitializeCache(cache_size);
  infer_thread_ = std::thread(&NNInterface::InferQueueLoop, this);
  LOG(INFO) << "NNInterface Initialized (queue mode, " << num_threads_
            << " threads, batch " << q_engine_batch_ << ", min fill "
            << q_min_fill_ << "). Engine Type: `"
            << KindToString(engines_[0]->kind()) << "`";
}

NNInterface::~NNInterface() {
  mu_.Lock();
  running_.store(false, std::memory_order_release);
  mu_.Unlock();

  if (infer_thread_.joinable()) {
    infer_thread_.join();
  }
  for (std::thread& t : wave_infer_threads_) {
    t.join();
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
  return NNKey{color_to_move, game.board().hash(), last_moves, game.komi(),
               cache_key_ko_ ? game.board().ko_loc() : kNoopLoc};
}

NNInferResult NNInterface::LoadAndGetInference(int thread_id, const Game& game,
                                               Color color_to_move,
                                               Probability& probability) {
  NNKey cache_key = MakeKey(game, color_to_move);

  if (std::optional<NNInferResult> hit = CacheLookup(thread_id, cache_key)) {
    // Cached. Immediately return result.
    absl::MutexLock l(&mu_);
    thread_info_[thread_id].res_cached = true;

    return *hit;
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
  engines_[WaveOf(thread_id)]->GetOwnership(RowOf(thread_id), ownership);
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

  load_hits_[canonical_tid].reset();
  if (std::optional<NNInferResult> hit = CacheLookup(canonical_tid, cache_key)) {
    load_hits_[canonical_tid].emplace(cache_key, std::move(*hit));
    // Cached. Do not need to wait for this thread. FetchEntry returns the hit.
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
    if (queue_mode_) q_.push_back(canonical_tid);
  }
}

NNInferResult NNInterface::FetchEntry(int thread_id, int offset,
                                      const game::Game& game,
                                      game::Color color_to_move) {
  const int canonical_tid = thread_id + offset;
  NNKey cache_key = MakeKey(game, color_to_move);

  ThreadInfo& thread_info = thread_info_[canonical_tid];
  std::optional<NNInferResult> hit;
  if (load_hits_[canonical_tid].has_value() &&
      load_hits_[canonical_tid]->first == cache_key) {
    // Cached at LoadEntry. Return that entry (not a second lookup: with the
    // shared cache it may have been evicted since).
    hit = std::move(load_hits_[canonical_tid]->second);
  } else if (!shared_cache_) {
    // Per-thread cache: only this thread writes it (previous behavior).
    hit = thread_caches_[canonical_tid].Get(cache_key);
  }
  load_hits_[canonical_tid].reset();
  if (hit.has_value()) {
    absl::MutexLock l(&mu_);
    thread_info.res_cached = false;
    return *hit;
  }

  wakers_[WaveOf(canonical_tid)]->Wait(thread_info.res_ready, mu_);

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
  load_hits_.resize(num_threads_);
}

void NNInterface::EnableSharedCache(size_t total_size) {
  CHECK(queue_mode_) << "Shared NN cache is only supported in queue mode.";
  shared_cache_lru_ = LRUCache<NNKey, NNInferResult>(total_size);
  shared_cache_ = true;
  LOG(INFO) << "NNInterface: shared NN cache, " << total_size << " entries.";
}

std::optional<NNInferResult> NNInterface::CacheLookup(int thread_id,
                                                      const NNKey& key) {
  std::optional<NNInferResult> hit;
  if (shared_cache_) {
    SpinMutex::Lock l(shared_cache_mu_);
    hit = shared_cache_lru_.Get(key);
  } else {
    hit = thread_caches_[thread_id].Get(key);
  }
  nnc_lookups_.fetch_add(1, std::memory_order_relaxed);
  if (hit.has_value()) nnc_hits_.fetch_add(1, std::memory_order_relaxed);
  return hit;
}

void NNInterface::CacheInsert(int thread_id, const NNKey& key,
                              const NNInferResult& result) {
  if (shared_cache_) {
    SpinMutex::Lock l(shared_cache_mu_);
    shared_cache_lru_.Insert(key, result);
    return;
  }
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

  if (queue_mode_) {
    // EXP queue mode: the inference thread copies it into a batch row.
    q_staged_[thread_id] = input_features;
    return;
  }
  engines_[WaveOf(thread_id)]->LoadBatch(RowOf(thread_id), input_features);
}

void NNInterface::InferLoop() {
  CHECK(num_threads_ > 1);
  while (running_.load(std::memory_order_acquire)) {
    Infer();
  }
}

void NNInterface::Infer() {
  // Owns the mu_ hold for the rest of this function: resets the signal count
  // and *then* releases the lock. The reset must land inside the same
  // uninterrupted hold as the ShouldInfer() predicate check -- otherwise a
  // worker's ++num_signaled_search_tasks_ (taken under mu_) can slip between
  // the check and the reset and be lost. Only one signal is sent per search
  // round, so a lost one strands every worker waiting on inference, and the
  // GTP client passes an effectively infinite timeout, making that permanent.
  struct RAIIFlagReset {
    NNInterface* self = nullptr;
    absl::Mutex* mu = nullptr;
    RAIIFlagReset() = delete;
    RAIIFlagReset(NNInterface* self, absl::Mutex* mu) : self(self), mu(mu) {};
    ~RAIIFlagReset() {
      self->num_signaled_search_tasks_[0] = 0;
      mu->Unlock();
    }
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

  RAIIFlagReset r(this, &mu_);
  if (num_registered_threads_ == 0) {
    return;
  }

  // Do not run inference if any thread still has an unread result: running
  // RunInference() would overwrite the engine output buffer before the worker
  // calls GetBatch(). Apply to both kAuto and kExplicit.
  for (const ThreadInfo& thread : thread_info_) {
    if (thread.res_ready.load(std::memory_order_acquire)) {
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
    return;
  }

  // Run Inference.
  auto inst_ns = [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  // INST: local start/end so two interfaces (pipelined search) do not mix
  // their timestamps through the globals.
  const int64_t inst_t0 = inst_ns();
  g_inst_infer_start.store(inst_t0);
  engines_[0]->RunInference();
  // INST (CPU pilot): P3_FAKE_INFER_US busy-waits this long per inference so a
  // dummy-engine build has a GPU-like round structure.
  static const int64_t kFakeInferNs = [] {
    const char* v = std::getenv("P3_FAKE_INFER_US");
    return v == nullptr ? int64_t{0} : int64_t{std::atoll(v)} * 1000;
  }();
  if (kFakeInferNs > 0) {
    const int64_t t_end = inst_t0 + kFakeInferNs;
    while (inst_ns() < t_end) {
    }
  }
  const int64_t inst_t1 = inst_ns();
  g_inst_infer_end.store(inst_t1);
  inst_infer_start.store(inst_t0);
  inst_infer_end.store(inst_t1);
  inst_wave_start[0].store(inst_t0);
  inst_wave_end[0].store(inst_t1);
  {
    absl::MutexLock l(&inst_iv_mu_);
    // Bounded: only searches drain this log (InstTakeIntervals).
    if (inst_iv_.size() < (1u << 20)) inst_iv_.emplace_back(inst_t0, inst_t1);
  }
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
  wakers_[0]->NotifyAll();
}

bool NNInterface::ShouldInfer() const {
  if (!running_.load(std::memory_order_acquire)) {
    return true;
  }

  if (signal_kind_ == SignalKind::kExplicit) {
    const int remaining =
        num_shared_search_tasks_ - num_exited_search_tasks_[0];
    if (remaining <= 0) return false;  // All tasks done; wait for destructor.
    return num_signaled_search_tasks_[0] == remaining;
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

/* static */ bool NNInterface::QueueReady(NNInterface* self) {
  return !self->running_.load(std::memory_order_acquire) ||
         static_cast<int>(self->q_.size()) >= self->q_min_fill_;
}

void NNInterface::InferQueueLoop() {
  while (running_.load(std::memory_order_acquire)) {
    InferQueue();
  }
}

// EXP queue mode: one batch. Waits for min_fill queued rows (with min_fill > 1
// and a timeout: or the timeout with at least one queued), takes up to
// engine_batch in queue order, and runs the engine with mu_ released. Workers
// only queue a row after reading their previous result (FetchEntry), so the
// staging and result slots of a taken row are not touched by their owner
// until res_ready is set below.
void NNInterface::InferQueue() {
  if (q_min_fill_ > 1 && q_min_fill_timeout_us_ > 0) {
    mu_.LockWhenWithTimeout(absl::Condition(&QueueReady, this),
                            absl::Microseconds(q_min_fill_timeout_us_));
  } else {
    mu_.LockWhen(absl::Condition(&QueueReady, this));
  }
  if (!running_.load(std::memory_order_acquire) || q_.empty()) {
    mu_.Unlock();
    return;
  }
  absl::InlinedVector<int, constants::kMaxNumThreads> rows;
  while (!q_.empty() && static_cast<int>(rows.size()) < q_engine_batch_) {
    rows.push_back(q_.front());
    q_.pop_front();
  }
  mu_.Unlock();

  auto inst_ns = [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  Engine* engine = engines_[0].get();
  const int64_t t_in = inst_ns();
  for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
    engine->LoadBatch(i, q_staged_[rows[i]]);
  }
  const int64_t inst_t0 = inst_ns();
  engine->RunInference();
  const int64_t inst_t1 = inst_ns();
  for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
    engine->GetBatch(i, q_results_[rows[i]]);
  }
  const int64_t t_out = inst_ns();
  inst_infer_start.store(inst_t0);
  inst_infer_end.store(inst_t1);
  inst_wave_start[0].store(inst_t0);
  inst_wave_end[0].store(inst_t1);
  inst_q_batches.fetch_add(1, std::memory_order_relaxed);
  inst_q_rows.fetch_add(rows.size(), std::memory_order_relaxed);
  inst_q_in_ns.fetch_add(inst_t0 - t_in, std::memory_order_relaxed);
  inst_q_out_ns.fetch_add(t_out - inst_t1, std::memory_order_relaxed);
  {
    absl::MutexLock l(&inst_iv_mu_);
    if (inst_iv_.size() < (1u << 20)) inst_iv_.emplace_back(inst_t0, inst_t1);
  }

  absl::MutexLock l(&mu_);
  for (int t : rows) {
    thread_info_[t].res_ready.store(true, std::memory_order_release);
    thread_info_[t].loaded_for_inference = false;
  }
  wakers_[0]->NotifyAll();
}

// EXP shared interface. Same per-wave readiness rule as ShouldInfer().
bool NNInterface::WaveShouldInfer(int wave) const {
  if (signal_kind_ == SignalKind::kExplicit) {
    const int remaining =
        num_shared_search_tasks_ - num_exited_search_tasks_[wave];
    if (remaining <= 0) return false;
    return num_signaled_search_tasks_[wave] == remaining;
  }
  bool exists_pending = false;
  for (int t = wave * wave_size_; t < (wave + 1) * wave_size_; ++t) {
    const auto& thread = thread_info_[t];
    if (!thread.registered) continue;
    if (!thread.res_cached && !thread.loaded_for_inference) return false;
    if (!thread.res_cached) exists_pending = true;
  }
  return exists_pending;
}

/* static */ bool NNInterface::AnyWaveReady(WaveCond* c) {
  const NNInterface* self = c->self;
  if (!self->running_.load(std::memory_order_acquire)) return true;
  for (int w = 0; w < self->num_waves_; ++w) {
    if ((c->only < 0 || c->only == w) && self->WaveShouldInfer(w)) return true;
  }
  return false;
}

void NNInterface::InferWavesLoop(int only) {
  while (running_.load(std::memory_order_acquire)) {
    InferWaves(only);
  }
}

// EXP shared interface: one inference of one wave. Picks a ready wave (with a
// single infer thread, round-robin from the wave after the last one run),
// resets its signal count in the same mu_ hold as the readiness check (see
// Infer()), snapshots its loaded rows, and runs the engine with mu_ released
// so the other wave can load and signal meanwhile. Rows loaded during the run
// are not in the snapshot and wait for the next inference of their wave.
// On timeout, a wave with loaded rows runs without its signal (as Infer()).
void NNInterface::InferWaves(int only) {
  WaveCond cond{this, only};
  bool ready = true;
  if (timeout_ > 0) {
    ready = mu_.LockWhenWithTimeout(absl::Condition(&AnyWaveReady, &cond),
                                    absl::Microseconds(timeout_));
  } else {
    mu_.LockWhen(absl::Condition(&AnyWaveReady, &cond));
  }
  if (!running_.load(std::memory_order_acquire) ||
      num_registered_threads_ == 0) {
    mu_.Unlock();
    return;
  }
  const auto wave_loaded = [&](int w) {
    for (int t = w * wave_size_; t < (w + 1) * wave_size_; ++t) {
      if (thread_info_[t].registered && thread_info_[t].loaded_for_inference) {
        return true;
      }
    }
    return false;
  };
  int wave = -1;
  for (int i = 1; i <= num_waves_ && wave < 0; ++i) {
    const int w = only >= 0 ? only : (last_wave_ + i + num_waves_) % num_waves_;
    if (ready ? WaveShouldInfer(w) : wave_loaded(w)) wave = w;
    if (only >= 0) break;
  }
  if (wave < 0) {
    mu_.Unlock();
    return;
  }
  num_signaled_search_tasks_[wave] = 0;
  const int t_begin = wave * wave_size_, t_end = (wave + 1) * wave_size_;
  // Do not overwrite results this wave's workers have not read (Infer()).
  for (int t = t_begin; t < t_end; ++t) {
    if (thread_info_[t].res_ready.load(std::memory_order_acquire)) {
      mu_.Unlock();
      return;
    }
  }
  absl::InlinedVector<int, constants::kMaxNumThreads> rows;
  for (int t = t_begin; t < t_end; ++t) {
    if (thread_info_[t].registered && thread_info_[t].loaded_for_inference) {
      rows.push_back(t);
    }
  }
  if (rows.empty()) {
    mu_.Unlock();
    return;
  }
  last_wave_ = wave;
  mu_.Unlock();

  auto inst_ns = [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  const int64_t inst_t0 = inst_ns();
  engines_[wave]->RunInference();
  const int64_t inst_t1 = inst_ns();
  inst_wave_start[wave].store(inst_t0);
  inst_wave_end[wave].store(inst_t1);
  {
    absl::MutexLock l(&inst_iv_mu_);
    if (inst_iv_.size() < (1u << 20)) inst_iv_.emplace_back(inst_t0, inst_t1);
  }

  absl::MutexLock l(&mu_);
  for (int t : rows) {
    thread_info_[t].res_ready.store(true, std::memory_order_release);
    thread_info_[t].loaded_for_inference = false;
  }
  for (int t = t_begin; t < t_end; ++t) {
    thread_info_[t].res_cached = false;
  }
  wakers_[wave]->NotifyAll();
}

}  // namespace nn
