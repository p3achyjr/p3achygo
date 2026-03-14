#pragma once

#include <algorithm>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/ring_buffer.h"
#include "cc/game/game.h"

namespace selfplay {

struct InitState {
  game::Board board;
  absl::InlinedVector<game::Move, constants::kMaxGameLen> last_moves;
  game::Color color_to_move;
  int move_num;
};

// Abstract interface for state reuse buffers.
class ReuseBuffer {
 public:
  virtual ~ReuseBuffer() = default;
  virtual void Add(InitState state, float regret) = 0;
  virtual std::optional<InitState> Get() = 0;
  // Whether to add states mid-game (via AddNewInitState).
  virtual bool AllowsMidGameAdd() const { return true; }
};

// Ignores regret, uniform random sampling.
class GoExploitReuseBuffer final : public ReuseBuffer {
 public:
  void Add(InitState state, float /*regret*/) {
    absl::MutexLock l(&mu_);
    buffer_.Append(state);
  }

  std::optional<InitState> Get() {
    absl::MutexLock l(&mu_);
    return buffer_.PopRandom();
  }

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
      : capacity_(capacity) {}

  void Add(InitState state, float regret) override {
    absl::MutexLock l(&mu_);
    if ((int)entries_.size() < capacity_) {
      entries_.push_back({std::move(state), regret});
      std::push_heap(entries_.begin(), entries_.end(), kMaxCmp);
    } else {
      // Evict lowest-regret entry if new entry has higher regret.
      int min_idx = MinRegretIndexLocked();
      if (regret > entries_[min_idx].regret) {
        entries_[min_idx] = {std::move(state), regret};
        std::make_heap(entries_.begin(), entries_.end(), kMaxCmp);
      }
    }
  }

  // Returns and removes the highest-regret state.
  std::optional<InitState> Get() override {
    absl::MutexLock l(&mu_);
    if (entries_.empty()) return std::nullopt;
    std::pop_heap(entries_.begin(), entries_.end(), kMaxCmp);
    InitState result = std::move(entries_.back().state);
    entries_.pop_back();
    return result;
  }

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

  int MinRegretIndexLocked() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int min_idx = 0;
    for (int i = 1; i < (int)entries_.size(); ++i) {
      if (entries_[i].regret < entries_[min_idx].regret) min_idx = i;
    }
    return min_idx;
  }

  bool AllowsMidGameAdd() const override { return false; }

  const int capacity_;
  std::vector<Entry> entries_ ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
};

}  // namespace selfplay
