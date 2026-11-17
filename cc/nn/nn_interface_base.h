#ifndef NN_NN_INTERFACE_BASE_H_
#define NN_NN_INTERFACE_BASE_H_

#include <array>
#include <atomic>
#include <memory>

#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/symmetry.h"
#include "cc/game/zobrist.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/engine/go_features.h"

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

// NN cache key.
struct NNKey {
  game::Color color_to_move;
  game::Zobrist::Hash board_hash;
  // [oldest, ..., most recent]; unused leading slots are kNoopLoc.
  std::array<game::Loc, constants::kNumLastMoves> last_moves;
  float komi;
  game::Loc ko;  // kNoopLoc unless the interface keys on ko.

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

// Shared helpers for NNInterface and PipelinedNNInterface.
//
// Cache key: the last `num_cache_last_moves` moves (range [0,
// kNumLastMoves]), plus the simple-ko point if `cache_key_ko`.
NNKey MakeNNKey(const game::Game& game, game::Color color_to_move,
                int num_cache_last_moves, bool cache_key_ko);
// Network input for `game` under symmetry `sym`.
GoFeatures MakeGoFeatures(const game::Game& game, game::Color color_to_move,
                          game::Symmetry sym);
// Maps the policy outputs of a result computed under `sym` back to the
// original orientation.
void UnapplySymmetry(game::Symmetry sym, NNInferResult& result);

/*
 * What a search thread needs from an NN interface: NNInterface (one engine)
 * and PipelinedNNInterface (one engine per wave) both implement it.
 */
class NNInterfaceBase {
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
  // interface. Bakes in `task_offset` so callers (Search, LeafEvaluator)
  // never have to pass it explicitly.
  struct Slot {
    Slot(NNInterfaceBase* nn_interface, int task_offset)
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

   private:
    NNInterfaceBase* nn_interface_;
    int task_offset_;
    int wave_;  // The wave (engine) this slot's rows live in.
  };

  virtual ~NNInterfaceBase() = default;

  Slot MakeSlot(int task_offset) { return Slot(this, task_offset); }

  virtual SignalKind signal_kind() const = 0;

  // Blocks until result is ready.
  virtual NNInferResult LoadAndGetInference(int thread_id,
                                            const game::Game& game,
                                            game::Color color_to_move,
                                            core::Probability& probability) = 0;
  // Just get ownership.
  virtual std::array<float, constants::kNumBoardLocs> LoadAndGetOwnership(
      int thread_id, const game::Game& game, game::Color color_to_move) = 0;

  // async API
  virtual void LoadEntry(int thread_id, int offset, const game::Game& game,
                         game::Color color_to_move,
                         core::Probability& probability) = 0;
  virtual NNInferResult FetchEntry(int thread_id, int offset,
                                   const game::Game& game,
                                   game::Color color_to_move) = 0;
  virtual void SignalReadyForInference(int wave) = 0;
  virtual void UnregisterSearchTask(int wave) = 0;

  // Threads [w * wave_size(), (w + 1) * wave_size()) live in wave w.
  virtual int num_waves() const = 0;
  virtual int wave_size() const = 0;
  int WaveOf(int canonical_tid) const { return canonical_tid / wave_size(); }

 protected:
  NNInterfaceBase() = default;
};

std::unique_ptr<Waker> MakeWaker(NNInterfaceBase::WakeStrategy strategy);

}  // namespace nn

#endif
