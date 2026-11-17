#ifndef NN_NN_CACHE_H_
#define NN_NN_CACHE_H_

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "cc/constants/constants.h"
#include "cc/core/lru_cache.h"
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
 * NN result cache used by NNInterface. `thread_id` is the caller's canonical
 * thread id; implementations may ignore it.
 */
class NNCache {
 public:
  virtual ~NNCache() = default;

  virtual std::optional<NNInferResult> Get(int thread_id, const NNKey& key) = 0;
  virtual void Insert(int thread_id, const NNKey& key,
                      const NNInferResult& result) = 0;
};

/*
 * One LRU cache per thread, each of `total_size / num_threads` entries. A
 * thread only sees its own entries; thread `t` must only be used by one thread
 * at a time.
 */
class PerThreadNNCache final : public NNCache {
 public:
  PerThreadNNCache(int num_threads, size_t total_size);

  std::optional<NNInferResult> Get(int thread_id, const NNKey& key) override;
  void Insert(int thread_id, const NNKey& key,
              const NNInferResult& result) override;

 private:
  std::vector<core::LRUCache<NNKey, NNInferResult>> caches_;
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
class SharedNNCache final : public NNCache {
 public:
  static constexpr int kDefaultLog2Mutexes = 16;

  // `num_entries` is rounded down to a power of two; 0 disables the cache.
  // The mutex pool has 2^log2_mutexes mutexes, at most one per slot.
  explicit SharedNNCache(size_t num_entries,
                         int log2_mutexes = kDefaultLog2Mutexes);

  SharedNNCache(SharedNNCache const&) = delete;
  SharedNNCache& operator=(SharedNNCache const&) = delete;

  std::optional<NNInferResult> Get(int thread_id, const NNKey& key) override;
  void Insert(int thread_id, const NNKey& key,
              const NNInferResult& result) override;

  size_t num_entries() const { return entries_ == nullptr ? 0 : mask_ + 1; }
  size_t num_mutexes() const {
    return mutexes_ == nullptr ? 0 : mutex_mask_ + 1;
  }

 private:
  using Entry = std::shared_ptr<const std::pair<NNKey, NNInferResult>>;

  uint64_t SlotOf(const NNKey& key) const { return absl::HashOf(key) & mask_; }

  std::unique_ptr<Entry[]> entries_;
  uint64_t mask_ = 0;
  std::unique_ptr<std::mutex[]> mutexes_;
  uint64_t mutex_mask_ = 0;
};

}  // namespace nn

#endif
