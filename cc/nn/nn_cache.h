#ifndef NN_NN_CACHE_H_
#define NN_NN_CACHE_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "absl/hash/hash.h"
#include "cc/constants/constants.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"
#include "cc/game/zobrist.h"
#include "cc/nn/engine/engine.h"

namespace nn {

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

/*
 * Thread-safe NN result cache, shared by every thread of an NNInterface.
 *
 * KataGo's NNCacheTable design (neuralnet/nneval.cpp): a direct-mapped table
 * of 2^k slots, each holding a shared_ptr to an immutable (key, result) entry.
 * slot = hash(key) & mask; an insert overwrites the slot (no LRU). A pool of
 * mutexes guards the slots (mutex = slot & pool mask). A lookup compares the
 * full stored key.
 *
 * Critical sections only copy or swap a shared_ptr: lookups copy the result
 * after unlocking, and inserts build the entry before locking and free the
 * replaced entry after unlocking.
 */
class NNCache final {
 public:
  static constexpr int kDefaultLog2Mutexes = 16;

  // `num_entries` is rounded down to a power of two; 0 disables the cache.
  // The mutex pool has 2^log2_mutexes mutexes, at most one per slot.
  explicit NNCache(size_t num_entries, int log2_mutexes = kDefaultLog2Mutexes);
  ~NNCache() = default;

  NNCache(NNCache const&) = delete;
  NNCache& operator=(NNCache const&) = delete;

  std::optional<NNInferResult> Get(const NNKey& key) const;
  void Insert(const NNKey& key, const NNInferResult& result);

  // Lookup / hit counts since the last call (relaxed; for stats only), then
  // resets them.
  struct Counters {
    int64_t lookups = 0;
    int64_t hits = 0;
  };
  Counters TakeCounters();

  size_t num_entries() const { return entries_ == nullptr ? 0 : mask_ + 1; }
  size_t num_mutexes() const {
    return mutexes_ == nullptr ? 0 : mutex_mask_ + 1;
  }

 private:
  using Entry = std::shared_ptr<const std::pair<NNKey, NNInferResult>>;

  uint64_t SlotOf(const NNKey& key) const { return absl::HashOf(key) & mask_; }

  std::unique_ptr<Entry[]> entries_;
  uint64_t mask_ = 0;
  mutable std::unique_ptr<std::mutex[]> mutexes_;
  uint64_t mutex_mask_ = 0;
  mutable std::atomic<int64_t> lookups_{0};
  mutable std::atomic<int64_t> hits_{0};
};

}  // namespace nn

#endif
