#ifndef MCTS_VLOSS_H_
#define MCTS_VLOSS_H_

#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/game/loc.h"

namespace mcts {

class VirtualLossTable final {
  absl::Mutex mu_;
  std::array<uint8_t, constants::kMaxMovesPerPosition> vloss_{};

 public:
  // Atomically marks `loc` as a loss. Returns whether `loc` was successfully
  // marked.
  bool MarkIfAvailable(game::Loc loc) {
    absl::MutexLock l(&mu_);
    uint8_t prev_value = vloss_[loc];
    vloss_[loc] = 1;
    return prev_value == 0;
  }

  void Unmark(game::Loc loc) {
    absl::MutexLock l(&mu_);
    vloss_[loc] = 0;
  }
};

class VLossLock final {
  VirtualLossTable* vlosses_;
  game::Loc loc_ = game::kNoopLoc;

 public:
  VLossLock(VirtualLossTable* vlosses) : vlosses_(vlosses) {}
  ~VLossLock() {
    if (loc_ != game::kNoopLoc) {
      vlosses_->Unmark(loc_);
    }
  }

  VLossLock(const VLossLock&) = delete;
  VLossLock(VLossLock&&) = delete;
  auto operator=(const VLossLock&) = delete;
  auto operator=(VLossLock&&) = delete;

  bool Acquire(game::Loc loc) {
    if (vlosses_->MarkIfAvailable(loc)) {
      loc_ = loc;
      return true;
    }

    return false;
  }
};
}  // namespace mcts

#endif  // MCTS_VLOSS_H_
