#ifndef MCTS_GLOBAL_TTABLE_H
#define MCTS_GLOBAL_TTABLE_H

#include "absl/synchronization/mutex.h"
#include "cc/core/lru_cache.h"
#include "cc/game/board.h"
#include "cc/game/zobrist.h"

namespace mcts {

class PositionCache final {
 public:
  PositionCache(int max_size)
      : cache_(max_size),
        num_queries_(0),
        num_hits_(0),
        num_unique_positions_approx_(0) {}
  ~PositionCache() = default;

  inline int num_queries() {
    absl::MutexLock l(&mu_);
    return num_queries_;
  }

  inline int num_hits() {
    absl::MutexLock l(&mu_);
    return num_hits_;
  }

  inline int num_unique_positions_approx() {
    absl::MutexLock l(&mu_);
    return num_unique_positions_approx_;
  }

  // Inserts key, or if the key is present, increments the counter.
  // Using this operation counts as a query.
  int Insert(game::Zobrist::Hash board_hash) {
    absl::MutexLock l(&mu_);

    ++num_queries_;
    std::optional<int> count = cache_.Get(board_hash);
    if (!count) {
      ++num_unique_positions_approx_;
      cache_.Insert(board_hash, 1);
      return 1;
    }

    ++num_hits_;
    int new_count = *count + 1;
    cache_.Insert(board_hash, new_count);
    return new_count;
  }

 private:
  core::LRUCache<game::Zobrist::Hash, int> cache_ ABSL_GUARDED_BY(mu_);
  int num_queries_ ABSL_GUARDED_BY(mu_);
  int num_hits_ ABSL_GUARDED_BY(mu_);
  int num_unique_positions_approx_ ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
};
}  // namespace mcts

#endif
