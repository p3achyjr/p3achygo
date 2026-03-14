#pragma once

#include <algorithm>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/heap.h"
#include "cc/core/ring_buffer.h"
#include "cc/game/game.h"

namespace selfplay {

struct InitState {
  game::Board board;
  absl::InlinedVector<game::Move, constants::kMaxGameLen> last_moves;
  game::Color color_to_move;
  int move_num;
  // If true, the game started from this state should use full search params
  // and force all moves to be trainable.
  bool force_full_search = false;
};

enum class BufferType { kGoExploit, kRegret };

// Abstract interface for state reuse buffers.
class ReuseBuffer {
 public:
  virtual ~ReuseBuffer() = default;
  virtual void Add(InitState state, float regret) = 0;
  virtual std::optional<InitState> Get() = 0;
  virtual BufferType GetType() const = 0;
};

// Ignores regret, uniform random sampling.
class GoExploitReuseBuffer final : public ReuseBuffer {
 public:
  void Add(InitState state, float /*regret*/) override {
    absl::MutexLock l(&mu_);
    buffer_.Append(state);
  }

  std::optional<InitState> Get() override {
    absl::MutexLock l(&mu_);
    return buffer_.PopRandom();
  }

  BufferType GetType() const override { return BufferType::kGoExploit; }

 private:
  core::RingBuffer<InitState, constants::kGoExploitBufferSize> buffer_
      ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
};

// Priority queue: highest-regret states are returned first from Get().
// Evicts lowest-regret entry on overflow.
class RegretGuidedBuffer final : public ReuseBuffer {
 public:
  explicit RegretGuidedBuffer(int capacity = constants::kGoExploitBufferSize)
      : regret_pq_(kMaxCmp, capacity) {}

  void Add(InitState state, float regret) override {
    absl::MutexLock l(&mu_);
    regret_pq_.PushHeap(Entry{state, regret});
  }

  // Returns and removes the highest-regret state, with force_full_search set.
  std::optional<InitState> Get() override {
    absl::MutexLock l(&mu_);
    if (regret_pq_.Size() <= 0) return std::nullopt;
    InitState state = regret_pq_.PopHeap().state;
    state.force_full_search = true;
    return state;
  }

  BufferType GetType() const override { return BufferType::kRegret; }

 private:
  struct Entry {
    InitState state;
    float regret;
  };

  struct MaxCmp {
    bool operator()(const Entry& a, const Entry& b) const {
      return a.regret < b.regret;
    }
  };
  static constexpr MaxCmp kMaxCmp{};

  core::Heap<Entry, MaxCmp> regret_pq_;
  absl::Mutex mu_;
};

}  // namespace selfplay
