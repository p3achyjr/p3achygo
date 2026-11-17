#ifndef NN_INTERFACE_H_
#define NN_INTERFACE_H_

#include <array>
#include <atomic>
#include <deque>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/lru_cache.h"
#include "cc/core/probability.h"
#include "cc/core/spin_mutex.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/symmetry.h"
#include "cc/nn/engine/engine.h"

namespace tensorflow {
class GraphDef;
class Tensor;
}  // namespace tensorflow

namespace nn {

// INST: steady_clock ns of the last RunInference() start/end.
inline std::atomic<int64_t> g_inst_infer_start{0}, g_inst_infer_end{0};

// Helper predicate for absl::Condition used by MutexWaker.
inline bool ResReadyTrue(std::atomic<bool>* r) {
  return r->load(std::memory_order_acquire);
}

// Abstract wakeup helper. Workers call Wait() after releasing mu_; Infer()
// calls NotifyAll() before releasing mu_.
class Waker {
 public:
  virtual ~Waker() = default;
  // Called WITHOUT mu held. Blocks until res_ready is true.
  virtual void Wait(std::atomic<bool>& res_ready, absl::Mutex& mu) = 0;
  // Called with mu HELD (before Infer() unlocks mu). Signals all waiters.
  virtual void NotifyAll() = 0;
};

// MutexWaker: uses absl::Mutex::LockWhen. Good when #threads ≤ hw_concurrency
// (absl spins rather than sleeping, ~0.6 µs/thread wakeup overhead).
class MutexWaker : public Waker {
 public:
  void Wait(std::atomic<bool>& res_ready, absl::Mutex& mu) override {
    mu.LockWhen(absl::Condition(ResReadyTrue, &res_ready));
    mu.Unlock();
  }
  // No-op: absl wakes LockWhen waiters when mu is unlocked with res_ready set.
  void NotifyAll() override {}
};

// GenCounterWaker: uses an atomic generation counter + notify_all. Good when
// #threads >> hw_concurrency (one FUTEX_WAKE unblocks all threads at once).
class GenCounterWaker : public Waker {
 public:
  void Wait(std::atomic<bool>& res_ready, absl::Mutex& /*mu*/) override {
    uint64_t gen = gen_.load(std::memory_order_acquire);
    while (!res_ready.load(std::memory_order_acquire)) {
      gen_.wait(gen);
      gen = gen_.load(std::memory_order_acquire);
    }
  }
  void NotifyAll() override {
    gen_.fetch_add(1, std::memory_order_release);
    gen_.notify_all();
  }

 private:
  std::atomic<uint64_t> gen_{0};
};

/*
 * Bridge between `Board` and other game objects to NN evaluation.
 *
 * This is an interface containing a single model. Each thread will fill its
 * slot in the neural network batch, and when the batch is full, inference will
 * be performed.
 */
class NNInterface final {
 public:
  enum class SignalKind : uint8_t {
    kAuto = 0,
    kExplicit = 1,
  };

  // How workers are notified when inference completes.
  //   kMutex:      absl::Mutex::Await — best when #threads ≤ hw_concurrency
  //                (absl spins rather than sleeping; ~0.6 µs/thread wakeup).
  //   kGenCounter: atomic generation counter + notify_all — best when
  //                #threads >> hw_concurrency (one FUTEX_WAKE for all).
  enum class WakeStrategy : uint8_t {
    kMutex = 0,
    kGenCounter = 1,
  };

  // Lightweight view of a contiguous slice of thread slots within an
  // NNInterface. Bakes in `task_offset` so callers (Search, LeafEvaluator)
  // never have to pass it explicitly.
  struct Slot {
    Slot(NNInterface* nn_interface, int task_offset)
        : nn_interface_(nn_interface),
          task_offset_(task_offset),
          wave_(nn_interface->WaveOf(task_offset)) {}

    SignalKind signal_kind() const { return nn_interface_->signal_kind(); }

    void LoadEntry(int thread_id, const game::Game& game,
                   game::Color color_to_move, core::Probability& probability) {
      nn_interface_->LoadEntry(thread_id, task_offset_, game, color_to_move,
                               probability);
    }

    NNInferResult FetchEntry(int thread_id, const game::Game& game,
                             game::Color color_to_move) {
      return nn_interface_->FetchEntry(thread_id, task_offset_, game,
                                       color_to_move);
    }

    // Routes through canonical_tid = thread_id + task_offset.
    NNInferResult LoadAndGetInference(int thread_id, const game::Game& game,
                                      game::Color color_to_move,
                                      core::Probability& probability) {
      return nn_interface_->LoadAndGetInference(thread_id + task_offset_, game,
                                                color_to_move, probability);
    }

    void SignalReadyForInference() {
      nn_interface_->SignalReadyForInference(wave_);
    }
    void UnregisterSearchTask() { nn_interface_->UnregisterSearchTask(wave_); }
    NNInterface* nn() const { return nn_interface_; }
    // EXP shared interface: the buffer set (wave) this slot's rows live in.
    int wave() const { return wave_; }

   private:
    NNInterface* nn_interface_;
    int task_offset_;
    int wave_;
  };

  Slot MakeSlot(int task_offset) { return Slot(this, task_offset); }

  NNInterface(int num_threads, std::unique_ptr<Engine> engine);
  NNInterface(int num_threads, int64_t timeout, size_t cache_size,
              std::unique_ptr<Engine> engine);
  // Convenience ctor for testing: same as above but with explicit WakeStrategy.
  NNInterface(int num_threads, int64_t timeout, size_t cache_size,
              std::unique_ptr<Engine> engine, WakeStrategy wake_strategy);

  // We only specify signal_kind during parallel search, meaning we also need to
  // specify how many search tasks share this interface.
  NNInterface(int num_threads, std::unique_ptr<Engine> engine,
              SignalKind signal_kind, int num_shared_search_tasks);
  NNInterface(int num_threads, int64_t timeout, size_t cache_size,
              std::unique_ptr<Engine> engine, SignalKind signal_kind,
              int num_shared_search_tasks,
              WakeStrategy wake_strategy = WakeStrategy::kGenCounter);
  // EXP shared interface (pipelined search): one interface, one buffer set
  // (engine of batch num_threads / W) per wave. Thread t lives in wave
  // t / (num_threads / W), row t % (num_threads / W). Each wave has its own
  // signal count, wake-up and inference; the engine runs outside mu_.
  // num_infer_threads: W (one per wave; waves' inferences may overlap on the
  // GPU) or 1 (one thread runs whichever wave is ready; never two at once).
  NNInterface(int num_threads, int64_t timeout, size_t cache_size,
              std::vector<std::unique_ptr<Engine>> wave_engines,
              SignalKind signal_kind, int num_shared_search_tasks,
              WakeStrategy wake_strategy, int num_infer_threads);
  // EXP round-free search (queue mode, KataGo-style): num_threads workers,
  // one engine of batch engine_batch. LoadEntry featurizes into the worker's
  // staging slot and queues it; one inference thread takes up to engine_batch
  // queued rows as soon as at least max(1, min_fill) are queued (with
  // min_fill > 1: or after min_fill_timeout_us with at least one queued),
  // runs the engine, and copies each row's output to the worker's result slot.
  // SignalReadyForInference() is a no-op.
  NNInterface(int num_threads, size_t cache_size,
              std::unique_ptr<Engine> engine, int engine_batch, int min_fill,
              int64_t min_fill_timeout_us, WakeStrategy wake_strategy);
  ~NNInterface();

  // Disable Copy
  NNInterface(NNInterface const&) = delete;
  NNInterface& operator=(NNInterface const&) = delete;

  SignalKind signal_kind() const { return signal_kind_; }

  // INST: this interface's last RunInference() start/end (steady_clock ns), and
  // a log of every RunInference() interval (for GPU-busy / overlap).
  std::atomic<int64_t> inst_infer_start{0}, inst_infer_end{0};
  // INST: per-wave last inference start/end (wave 0 only when not shared).
  std::array<std::atomic<int64_t>, 2> inst_wave_start{}, inst_wave_end{};
  std::vector<std::pair<int64_t, int64_t>> InstTakeIntervals() {
    absl::MutexLock l(&inst_iv_mu_);
    std::vector<std::pair<int64_t, int64_t>> out;
    out.swap(inst_iv_);
    return out;
  }

  // Controls how many of the most-recent moves are included in the NN cache
  // key. Range [0, kNumLastMoves]. Default: kNumLastMoves (5).
  void SetNumCacheLastMoves(int n) { num_cache_last_moves_ = n; }

  // Whether to include the board's simple-ko point in the NN cache key.
  void SetCacheKeyKo(bool b) { cache_key_ko_ = b; }

  // EXP (P3_SHARED_CACHE=1, queue mode only): one LRU cache of `total_size`
  // entries shared by all threads, guarded by a spin lock, instead of the
  // per-thread caches.
  void EnableSharedCache(size_t total_size);

  // EXP: NN cache lookups / hits since the last call (LoadEntry and
  // LoadAndGetInference lookups), then resets both.
  std::pair<int64_t, int64_t> TakeCacheCounters() {
    return {nnc_lookups_.exchange(0, std::memory_order_relaxed),
            nnc_hits_.exchange(0, std::memory_order_relaxed)};
  }

  // Blocks until result is ready.
  NNInferResult LoadAndGetInference(int thread_id, const game::Game& game,
                                    game::Color color_to_move,
                                    core::Probability& probability)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Just get ownership.
  std::array<float, constants::kNumBoardLocs> LoadAndGetOwnership(
      int thread_id, const game::Game& game, game::Color color_to_move)
      ABSL_LOCKS_EXCLUDED(mu_);

  void RegisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);
  void UnregisterThread(int thread_id) ABSL_LOCKS_EXCLUDED(mu_);

  // async API
  void LoadEntry(int thread_id, int offset, const game::Game& game,
                 game::Color color_to_move, core::Probability& probability)
      ABSL_LOCKS_EXCLUDED(mu_);
  NNInferResult FetchEntry(int thread_id, int offset, const game::Game& game,
                           game::Color color_to_move) ABSL_LOCKS_EXCLUDED(mu_);
  inline void SignalReadyForInference(int wave = 0) {
    if (queue_mode_) return;
    {
      absl::MutexLock l(&mu_);
      ++num_signaled_search_tasks_[wave];
    }
    if (num_threads_ == 1) {
      Infer();  // No infer_thread_ exists; run synchronously.
    }
  }
  inline void UnregisterSearchTask(int wave = 0) {
    absl::MutexLock l(&mu_);
    ++num_exited_search_tasks_[wave];
  }
  int num_waves() const { return num_waves_; }
  bool queue_mode() const { return queue_mode_; }
  // INST queue mode: batches run, rows evaluated, and time (ns) the inference
  // thread spent copying rows in / out around the engine call.
  std::atomic<int64_t> inst_q_batches{0}, inst_q_rows{0}, inst_q_in_ns{0},
      inst_q_out_ns{0};
  int wave_size() const { return wave_size_; }
  int WaveOf(int canonical_tid) const { return canonical_tid / wave_size_; }

 private:
  static constexpr int64_t kTimeoutUs = 400;
  struct NNKey {
    game::Color color_to_move;
    game::Zobrist::Hash board_hash;
    // [oldest, ..., most recent]; unused leading slots are kNoopLoc.
    std::array<game::Loc, constants::kNumLastMoves> last_moves;
    float komi;
    game::Loc ko;  // kNoopLoc unless cache_key_ko_.

    friend bool operator==(const NNKey& c0, const NNKey& c1) {
      return c0.color_to_move == c1.color_to_move &&
             c0.board_hash == c1.board_hash && c0.last_moves == c1.last_moves &&
             c0.komi == c1.komi && c0.ko == c1.ko;
    }

    template <typename H>
    friend H AbslHashValue(H h, const NNKey& c) {
      h = H::combine(std::move(h), c.color_to_move, c.board_hash);
      for (const game::Loc& loc : c.last_moves) {
        h = H::combine(std::move(h), loc);
      }
      h = H::combine(std::move(h), c.komi, c.ko);
      return h;
    }
  };

  struct ThreadInfo {
    bool registered = true;  // If this thread is registered.
    bool loaded_for_inference =
        false;  // If this thread has loaded its data for inference.
    std::atomic<bool> res_ready{false};  // Whether inference result is ready.
    bool res_cached = false;  // Whether the cache key is cached. Also toggles
                              // whether to cache the NN inference result.

    ThreadInfo() = default;
    ThreadInfo(const ThreadInfo&) = delete;
    ThreadInfo& operator=(const ThreadInfo&) = delete;
  };

  // Cache Helpers.
  void InitializeCache(size_t cache_size);
  // One lookup (counts a lookup / hit). With the shared cache another thread
  // may evict between two calls, so callers look up once and keep the result.
  std::optional<NNInferResult> CacheLookup(int thread_id, const NNKey& key);
  void CacheInsert(int thread_id, const NNKey& key,
                   const NNInferResult& result);
  void LoadBatch(int thread_id, const game::Game& game,
                 game::Color color_to_move, game::Symmetry sym);
  inline NNInferResult GetBatch(int thread_id, game::Symmetry sym)
      ABSL_LOCKS_EXCLUDED(mu_) {
    NNInferResult infer_result;
    if (queue_mode_) {
      infer_result = q_results_[thread_id];
    } else {
      engines_[WaveOf(thread_id)]->GetBatch(RowOf(thread_id), infer_result);
    }

    // Clear res_ready without the lock. Safe because:
    // - engine_->GetBatch() completes before this store (release ordering).
    // - Infer() checks res_ready with acquire ordering before RunInference().
    // - So if infer sees false, GetBatch has already consumed the engine
    // buffer.
    thread_info_[thread_id].res_ready.store(false, std::memory_order_release);

    // Unapply symmetry.
    std::array<float, constants::kNumBoardLocs> grid_logits_sym;
    std::array<float, constants::kNumBoardLocs> grid_probs_sym;
    std::array<float, constants::kNumBoardLocs> grid_probs_opt_sym;
    std::copy(infer_result.move_logits.begin(),
              infer_result.move_logits.begin() + constants::kNumBoardLocs,
              grid_logits_sym.begin());
    std::copy(infer_result.move_probs.begin(),
              infer_result.move_probs.begin() + constants::kNumBoardLocs,
              grid_probs_sym.begin());
    std::copy(infer_result.opt_move_probs.begin(),
              infer_result.opt_move_probs.begin() + constants::kNumBoardLocs,
              grid_probs_opt_sym.begin());
    std::array<float, constants::kNumBoardLocs> grid_logits =
        ApplyInverse(sym, grid_logits_sym, BOARD_LEN);
    std::array<float, constants::kNumBoardLocs> grid_probs =
        ApplyInverse(sym, grid_probs_sym, BOARD_LEN);
    std::array<float, constants::kNumBoardLocs> grid_probs_opt =
        ApplyInverse(sym, grid_probs_opt_sym, BOARD_LEN);
    std::copy(grid_logits.begin(), grid_logits.end(),
              infer_result.move_logits.begin());
    std::copy(grid_probs.begin(), grid_probs.end(),
              infer_result.move_probs.begin());
    std::copy(grid_probs_opt.begin(), grid_probs_opt.end(),
              infer_result.opt_move_probs.begin());

    return infer_result;
  }

  // Called by worker.
  inline void SignalLoadedAndBlockUntilReady(int thread_id)
      ABSL_LOCKS_EXCLUDED(mu_) {
    if (num_threads_ == 1) {
      engines_[0]->RunInference();
      return;
    }
    ThreadInfo& thread_info = thread_info_[thread_id];
    {
      absl::MutexLock l(&mu_);
      thread_info.loaded_for_inference = true;
      thread_info.res_ready.store(false, std::memory_order_relaxed);
      thread_info.res_cached = false;
      if (queue_mode_) q_.push_back(thread_id);
    }
    // mu_ released before calling waker — same convention for both strategies.
    wakers_[WaveOf(thread_id)]->Wait(thread_info.res_ready, mu_);
  }

  inline void BlockUntilReady(int thread_id) ABSL_LOCKS_EXCLUDED(mu_) {
    wakers_[WaveOf(thread_id)]->Wait(thread_info_[thread_id].res_ready, mu_);
  }

  int RowOf(int canonical_tid) const { return canonical_tid % wave_size_; }

  NNKey MakeKey(const game::Game& game, game::Color color_to_move) const;

  // Inference Loop.
  void InferLoop();
  void Infer() ABSL_LOCKS_EXCLUDED(mu_);
  bool ShouldInfer() const;

  // EXP shared interface: inference loop / step for wave `only` (-1: any).
  struct WaveCond {
    const NNInterface* self;
    int only;
  };
  static bool AnyWaveReady(WaveCond* c);
  bool WaveShouldInfer(int wave) const;
  void InferWavesLoop(int only);
  void InferWaves(int only) ABSL_LOCKS_EXCLUDED(mu_);

  // EXP queue mode.
  static bool QueueReady(NNInterface* self);
  void InferQueueLoop();
  void InferQueue() ABSL_LOCKS_EXCLUDED(mu_);
  bool queue_mode_ = false;
  int q_engine_batch_ = 0;
  int q_min_fill_ = 1;
  int64_t q_min_fill_timeout_us_ = 0;
  std::deque<int> q_ ABSL_GUARDED_BY(mu_);  // queued thread ids.
  std::vector<GoFeatures> q_staged_;        // per thread, written by owner.
  std::vector<NNInferResult> q_results_;    // per thread, written by infer.

  std::string id_;
  int num_registered_threads_ ABSL_GUARDED_BY(mu_);
  const int num_threads_;
  const SignalKind signal_kind_;
  int num_cache_last_moves_ = constants::kNumLastMoves;
  bool cache_key_ko_ = false;

  // Synchronization
  absl::Mutex mu_;
  absl::InlinedVector<ThreadInfo, constants::kMaxNumThreads> thread_info_;

  // Inference thread. Runs inference until told to stop.
  std::thread infer_thread_;
  // EXP shared interface: wave inference threads (empty when num_waves_ == 1).
  std::vector<std::thread> wave_infer_threads_;
  static constexpr int kMaxWaves = 2;
  int num_waves_ = 1;
  int wave_size_;       // num_threads_ / num_waves_.
  int last_wave_ = -1;  // last wave run by a shared infer thread.
  std::atomic<bool> running_;

  std::array<core::LRUCache<NNKey, NNInferResult>, constants::kMaxNumThreads>
      thread_caches_;  // Per-thread cache.
  // EXP shared cache (see EnableSharedCache).
  bool shared_cache_ = false;
  core::SpinMutex shared_cache_mu_;
  core::LRUCache<NNKey, NNInferResult> shared_cache_lru_;
  // Per thread: the cache hit found by LoadEntry, returned by the matching
  // FetchEntry (written and read only by the owning thread).
  std::vector<std::optional<std::pair<NNKey, NNInferResult>>> load_hits_;
  std::atomic<int64_t> nnc_lookups_{0};
  std::atomic<int64_t> nnc_hits_{0};
  const int64_t timeout_;

  // Engines: one per wave (one when not shared).
  std::vector<std::unique_ptr<Engine>> engines_;

  // Per-thread symmetries.
  absl::InlinedVector<game::Symmetry, constants::kMaxNumThreads> symmetries_;

  // Explicit signalling.
  const int num_shared_search_tasks_;
  // Per wave; index 0 only when not shared.
  std::array<int, kMaxWaves> num_signaled_search_tasks_{};
  std::array<int, kMaxWaves> num_exited_search_tasks_{};

  // Wakeup strategy: owns the waiting/notification mechanism.
  // One per wave.
  std::vector<std::unique_ptr<Waker>> wakers_;

  // INST: RunInference() intervals since the last InstTakeIntervals().
  absl::Mutex inst_iv_mu_;
  std::vector<std::pair<int64_t, int64_t>> inst_iv_
      ABSL_GUARDED_BY(inst_iv_mu_);
};

}  // namespace nn

#endif
