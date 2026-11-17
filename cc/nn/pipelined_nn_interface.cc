#include "cc/nn/pipelined_nn_interface.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"

namespace nn {

namespace {
using namespace ::core;
using namespace ::game;
}  // namespace

PipelinedNNInterface::PipelinedNNInterface(
    int num_threads, int64_t timeout, size_t cache_size,
    std::vector<std::unique_ptr<Engine>> wave_engines, SignalKind signal_kind,
    int num_shared_search_tasks, WakeStrategy wake_strategy)
    : num_threads_(num_threads),
      num_waves_(static_cast<int>(wave_engines.size())),
      wave_size_(num_waves_ > 0 ? num_threads / num_waves_ : 0),
      signal_kind_(signal_kind),
      timeout_(timeout),
      thread_info_(num_threads),
      running_(true),
      engines_(std::move(wave_engines)),
      symmetries_(num_threads),
      num_shared_search_tasks_(num_shared_search_tasks) {
  CHECK(num_waves_ >= 2 && num_waves_ <= kMaxWaves) << num_waves_;
  CHECK(num_threads_ % num_waves_ == 0 && wave_size_ > 1)
      << num_threads_ << " threads, " << num_waves_ << " waves";
  for (int w = 0; w < num_waves_; ++w) {
    wakers_.push_back(MakeWaker(wake_strategy));
  }
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_caches_[thread_id] =
        LRUCache<NNKey, NNInferResult>(cache_size / num_threads_);
  }
  infer_thread_ = std::thread(&PipelinedNNInterface::InferWavesLoop, this);

  LOG(INFO) << "NNInterface Initialized (shared, " << num_waves_
            << " waves). Engine Type: `" << KindToString(engines_[0]->kind())
            << "`";
}

PipelinedNNInterface::~PipelinedNNInterface() {
  mu_.Lock();
  running_.store(false, std::memory_order_release);
  mu_.Unlock();

  if (infer_thread_.joinable()) {
    infer_thread_.join();
  }
}

NNInferResult PipelinedNNInterface::LoadAndGetInference(
    int thread_id, const Game& game, Color color_to_move,
    Probability& probability) {
  NNKey cache_key = MakeKey(game, color_to_move);

  if (thread_caches_[thread_id].Contains(cache_key)) {
    // Cached. Immediately return result.
    absl::MutexLock l(&mu_);
    thread_info_[thread_id].res_cached = true;

    return thread_caches_[thread_id].Get(cache_key).value();
  }

  // Not cached. Load for inference.
  DCHECK(game.moves().size() >= constants::kNumLastMoves);

  Symmetry sym = GetRandomSymmetry(probability.prng());
  LoadBatch(thread_id, game, color_to_move, sym);
  SignalLoadedAndBlockUntilReady(thread_id);

  NNInferResult infer_result = GetBatch(thread_id, sym);

  // Cache this result.
  thread_caches_[thread_id].Insert(cache_key, infer_result);
  return infer_result;
}

std::array<float, constants::kNumBoardLocs>
PipelinedNNInterface::LoadAndGetOwnership(int thread_id, const Game& game,
                                          Color color_to_move) {
  LoadBatch(thread_id, game, color_to_move, Symmetry::kIdentity);
  SignalLoadedAndBlockUntilReady(thread_id);

  // Inference result is now ready.
  std::array<float, constants::kNumBoardLocs> ownership;
  engines_[WaveOf(thread_id)]->GetOwnership(RowOf(thread_id), ownership);
  return ownership;
}

void PipelinedNNInterface::LoadEntry(int thread_id, int offset,
                                     const game::Game& game,
                                     game::Color color_to_move,
                                     core::Probability& probability) {
  const int canonical_tid = thread_id + offset;
  NNKey cache_key = MakeKey(game, color_to_move);

  if (thread_caches_[canonical_tid].Contains(cache_key)) {
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

NNInferResult PipelinedNNInterface::FetchEntry(int thread_id, int offset,
                                               const game::Game& game,
                                               game::Color color_to_move) {
  const int canonical_tid = thread_id + offset;
  NNKey cache_key = MakeKey(game, color_to_move);

  ThreadInfo& thread_info = thread_info_[canonical_tid];
  if (thread_caches_[canonical_tid].Contains(cache_key)) {
    // Cached. Return cache entry.
    absl::MutexLock l(&mu_);
    thread_info.res_cached = false;
    return thread_caches_[canonical_tid].Get(cache_key).value();
  }

  wakers_[WaveOf(canonical_tid)]->Wait(thread_info.res_ready, mu_);

  NNInferResult infer_result =
      GetBatch(canonical_tid, symmetries_[canonical_tid]);

  // Cache this result.
  thread_caches_[canonical_tid].Insert(cache_key, infer_result);
  return infer_result;
}

NNInferResult PipelinedNNInterface::GetBatch(int thread_id,
                                             game::Symmetry sym) {
  NNInferResult infer_result;
  engines_[WaveOf(thread_id)]->GetBatch(RowOf(thread_id), infer_result);

  // Clear res_ready without the lock. Safe because:
  // - GetBatch() completes before this store (release ordering).
  // - InferWaves() checks res_ready with acquire ordering before running the
  //   wave's engine.
  // - So if infer sees false, GetBatch has already consumed the engine
  //   buffer.
  thread_info_[thread_id].res_ready.store(false, std::memory_order_release);

  UnapplySymmetry(sym, infer_result);
  return infer_result;
}

void PipelinedNNInterface::SignalLoadedAndBlockUntilReady(int thread_id) {
  ThreadInfo& thread_info = thread_info_[thread_id];
  {
    absl::MutexLock l(&mu_);
    thread_info.loaded_for_inference = true;
    thread_info.res_ready.store(false, std::memory_order_relaxed);
    thread_info.res_cached = false;
  }
  // mu_ released before calling waker — same convention for both strategies.
  wakers_[WaveOf(thread_id)]->Wait(thread_info.res_ready, mu_);
}

// Same readiness rule as NNInterface::ShouldInfer(), per wave.
bool PipelinedNNInterface::WaveShouldInfer(int wave) const {
  if (signal_kind_ == SignalKind::kExplicit) {
    const int remaining =
        num_shared_search_tasks_ - num_exited_search_tasks_[wave];
    if (remaining <= 0) return false;
    return num_signaled_search_tasks_[wave] == remaining;
  }
  bool exists_pending = false;
  for (int t = wave * wave_size_; t < (wave + 1) * wave_size_; ++t) {
    const auto& thread = thread_info_[t];
    if (!thread.res_cached && !thread.loaded_for_inference) return false;
    if (!thread.res_cached) exists_pending = true;
  }
  return exists_pending;
}

/* static */ bool PipelinedNNInterface::AnyWaveReady(
    PipelinedNNInterface* self) {
  if (!self->running_.load(std::memory_order_acquire)) return true;
  for (int w = 0; w < self->num_waves_; ++w) {
    if (self->WaveShouldInfer(w)) return true;
  }
  return false;
}

void PipelinedNNInterface::InferWavesLoop() {
  while (running_.load(std::memory_order_acquire)) {
    InferWaves();
  }
}

// One inference of one wave. Picks a ready wave (round-robin from the wave
// after the last one run), resets its signal count in the same mu_ hold as the
// readiness check (see NNInterface::Infer()), snapshots its loaded rows, and
// runs the engine with mu_ released so the other wave can load and signal
// meanwhile. Rows loaded during the run are not in the snapshot and wait for
// the next inference of their wave. On timeout, a wave with loaded rows runs
// without its signal (as NNInterface::Infer()).
void PipelinedNNInterface::InferWaves() {
  bool ready = true;
  if (timeout_ > 0) {
    ready = mu_.LockWhenWithTimeout(absl::Condition(&AnyWaveReady, this),
                                    absl::Microseconds(timeout_));
  } else {
    mu_.LockWhen(absl::Condition(&AnyWaveReady, this));
  }
  if (!running_.load(std::memory_order_acquire)) {
    mu_.Unlock();
    return;
  }
  const auto wave_loaded = [&](int w) {
    for (int t = w * wave_size_; t < (w + 1) * wave_size_; ++t) {
      if (thread_info_[t].loaded_for_inference) return true;
    }
    return false;
  };
  int wave = -1;
  for (int i = 1; i <= num_waves_ && wave < 0; ++i) {
    const int w = (last_wave_ + i + num_waves_) % num_waves_;
    if (ready ? WaveShouldInfer(w) : wave_loaded(w)) wave = w;
  }
  if (wave < 0) {
    mu_.Unlock();
    return;
  }
  num_signaled_search_tasks_[wave] = 0;
  const int t_begin = wave * wave_size_, t_end = (wave + 1) * wave_size_;
  // Do not overwrite results this wave's workers have not read
  // (NNInterface::Infer()).
  for (int t = t_begin; t < t_end; ++t) {
    if (thread_info_[t].res_ready.load(std::memory_order_acquire)) {
      mu_.Unlock();
      return;
    }
  }
  absl::InlinedVector<int, constants::kMaxNumThreads> rows;
  for (int t = t_begin; t < t_end; ++t) {
    if (thread_info_[t].loaded_for_inference) rows.push_back(t);
  }
  if (rows.empty()) {
    mu_.Unlock();
    return;
  }
  last_wave_ = wave;
  mu_.Unlock();

  engines_[wave]->RunInference();

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
