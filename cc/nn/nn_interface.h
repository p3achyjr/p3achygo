#ifndef NN_INTERFACE_H_
#define NN_INTERFACE_H_

#include <array>
#include <atomic>
#include <deque>
#include <optional>
#include <thread>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/symmetry.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/nn_cache.h"

namespace tensorflow {
class GraphDef;
class Tensor;
}  // namespace tensorflow

namespace nn {

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
    // Round-free search: workers queue rows at any time; the inference thread
    // runs up to `engine_batch` queued rows (FIFO) as soon as one is queued.
    // Rows are staged per thread and results copied out per thread, so a
    // thread's row is not tied to an engine row. SignalReadyForInference()
    // has no effect. Built with the round-free constructor only.
    kRoundFree = 2,
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
        : nn_interface_(nn_interface), task_offset_(task_offset) {}

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

    void SignalReadyForInference() { nn_interface_->SignalReadyForInference(); }
    void UnregisterSearchTask() { nn_interface_->UnregisterSearchTask(); }

   private:
    NNInterface* nn_interface_;
    int task_offset_;
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
  // Round-free search (SignalKind::kRoundFree). `engine` has batch size
  // `engine_batch`, at most `num_threads`.
  NNInterface(int num_threads, size_t cache_size,
              std::unique_ptr<Engine> engine, int engine_batch,
              WakeStrategy wake_strategy = WakeStrategy::kGenCounter);
  ~NNInterface();

  // Disable Copy
  NNInterface(NNInterface const&) = delete;
  NNInterface& operator=(NNInterface const&) = delete;

  SignalKind signal_kind() const { return signal_kind_; }

  // Controls how many of the most-recent moves are included in the NN cache
  // key. Range [0, kNumLastMoves]. Default: kNumLastMoves (5).
  void SetNumCacheLastMoves(int n) { num_cache_last_moves_ = n; }

  // Whether to include the board's simple-ko point in the NN cache key.
  void SetCacheKeyKo(bool b) { cache_key_ko_ = b; }

  // Degenerate-eval re-evaluation (LoadAndGetInference, and FetchEntry in
  // round-free mode). An
  // eval is degenerate if its largest policy logit exceeds `max_logit` (> 0),
  // or if |E[outcome] - E[mcts_value_dist]| (both in [-1, 1]) exceeds
  // `value_disagree` (> 0). A degenerate eval is re-run under up to
  // `max_retries` other symmetries; the first non-degenerate result is kept
  // (else the one with the smallest max logit).
  // Motivation: nbt nets give confident garbage on some symmetries
  // (2026-11-17: top logit ~20-30 vs <=15 normally, |score| > 100, outcome
  // head contradicting the saturated mcts-value head).
  void SetDegenerateReeval(float max_logit, float value_disagree,
                           int max_retries) {
    reeval_max_logit_ = max_logit;
    reeval_value_disagree_ = value_disagree;
    reeval_max_retries_ = max_retries;
    if (max_logit > 0.0f || value_disagree > 0.0f) {
      LOG(INFO) << "NN re-eval on: max_logit=" << max_logit
                << " value_disagree=" << value_disagree
                << " max_retries=" << max_retries;
    }
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
  inline void SignalReadyForInference() {
    {
      absl::MutexLock l(&mu_);
      ++num_signaled_search_tasks_;
    }
    if (num_threads_ == 1) {
      Infer();  // No infer_thread_ exists; run synchronously.
    }
  }
  inline void UnregisterSearchTask() {
    absl::MutexLock l(&mu_);
    ++num_exited_search_tasks_;
  }

 private:
  static constexpr int64_t kTimeoutUs = 400;
  struct ThreadInfo {
    bool registered = true;  // If this thread is registered.
    bool loaded_for_inference =
        false;  // If this thread has loaded its data for inference.
    std::atomic<bool> res_ready{false};  // Whether inference result is ready.
    bool res_cached = false;  // Whether the cache key is cached. Also toggles
                              // whether to cache the NN inference result.
    // Owner thread only: LoadEntry found its key in the cache and holds the
    // result in held_hits_ for the matching FetchEntry (a shared cache may
    // evict the entry in between).
    bool hit_held = false;
    // kRoundFree: copy this row's ownership out after inference.
    bool wants_ownership = false;

    ThreadInfo() = default;
    ThreadInfo(const ThreadInfo&) = delete;
    ThreadInfo& operator=(const ThreadInfo&) = delete;
  };

  void LoadBatch(int thread_id, const game::Game& game,
                 game::Color color_to_move, game::Symmetry sym);
  inline NNInferResult GetBatch(int thread_id, game::Symmetry sym)
      ABSL_LOCKS_EXCLUDED(mu_) {
    NNInferResult infer_result;
    if (signal_kind_ == SignalKind::kRoundFree) {
      infer_result = rf_results_[thread_id];
    } else {
      engine_->GetBatch(thread_id, infer_result);
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
      engine_->RunInference();
      return;
    }
    ThreadInfo& thread_info = thread_info_[thread_id];
    {
      absl::MutexLock l(&mu_);
      thread_info.loaded_for_inference = true;
      thread_info.res_ready.store(false, std::memory_order_relaxed);
      thread_info.res_cached = false;
      if (signal_kind_ == SignalKind::kRoundFree)
        rf_queue_.push_back(thread_id);
    }
    // mu_ released before calling waker — same convention for both strategies.
    waker_->Wait(thread_info.res_ready, mu_);
  }

  inline void BlockUntilReady(int thread_id) ABSL_LOCKS_EXCLUDED(mu_) {
    waker_->Wait(thread_info_[thread_id].res_ready, mu_);
  }

  NNKey MakeKey(const game::Game& game, game::Color color_to_move) const;

  // Inference Loop.
  void InferLoop();
  void Infer() ABSL_LOCKS_EXCLUDED(mu_);
  bool ShouldInfer() const;

  // Round-free inference loop: one batch per call.
  void InferRoundFreeLoop();
  void InferRoundFree() ABSL_LOCKS_EXCLUDED(mu_);
  static bool RoundFreeReady(NNInterface* self);

  std::string id_;
  int num_registered_threads_ ABSL_GUARDED_BY(mu_);
  const int num_threads_;
  const SignalKind signal_kind_;
  int num_cache_last_moves_ = constants::kNumLastMoves;
  bool cache_key_ko_ = false;
  bool IsDegenerate(const NNInferResult& r, float* max_logit) const;
  // If re-eval is on and `infer_result` is degenerate, re-runs the position
  // under up to reeval_max_retries_ other symmetries (blocking) and replaces
  // `infer_result` with the first non-degenerate one (else the smallest max
  // logit). `offset` picks the cyclic start among the 7 other symmetries.
  void MaybeReeval(int thread_id, const game::Game& game,
                   game::Color color_to_move, game::Symmetry sym, int offset,
                   NNInferResult& infer_result);
  float reeval_max_logit_ = 0.0f;       // 0 = off.
  float reeval_value_disagree_ = 0.0f;  // 0 = off.
  int reeval_max_retries_ = 0;
  std::atomic<int64_t> num_evals_ = 0;
  std::atomic<int64_t> num_degenerate_evals_ = 0;
  std::atomic<int64_t> num_reeval_extra_evals_ = 0;
  std::atomic<int64_t> num_reeval_rescued_ = 0;

  // Synchronization
  absl::Mutex mu_;
  absl::InlinedVector<ThreadInfo, constants::kMaxNumThreads> thread_info_;

  // Inference thread. Runs inference until told to stop.
  std::thread infer_thread_;
  std::atomic<bool> running_;

  // Per-thread LRU caches, or one shared cache in round-free mode.
  std::unique_ptr<NNCache> cache_;
  // Per-thread cache hits held between LoadEntry and FetchEntry.
  std::vector<NNInferResult> held_hits_;
  const int64_t timeout_;

  // Engine.
  std::unique_ptr<Engine> engine_;

  // Per-thread symmetries.
  absl::InlinedVector<game::Symmetry, constants::kMaxNumThreads> symmetries_;

  // Explicit signalling.
  const int num_shared_search_tasks_;
  int num_signaled_search_tasks_ = 0;
  int num_exited_search_tasks_ = 0;

  // Wakeup strategy: owns the waiting/notification mechanism.
  std::unique_ptr<Waker> waker_;

  // kRoundFree only.
  int rf_engine_batch_ = 0;
  std::deque<int> rf_queue_ ABSL_GUARDED_BY(mu_);  // Queued thread ids.
  std::vector<GoFeatures> rf_staged_;              // Per-thread input.
  std::vector<NNInferResult> rf_results_;          // Per-thread output.
  std::vector<std::array<float, constants::kNumBoardLocs>> rf_ownership_;
};

}  // namespace nn

#endif
