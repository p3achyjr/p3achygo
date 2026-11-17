#include "cc/nn/nn_cache.h"

#include <atomic>
#include <thread>
#include <vector>

#include "cc/core/doctest_include.h"

namespace nn {
namespace {

NNKey Key(uint64_t hash, game::Color color = BLACK, float komi = 7.5f) {
  NNKey key{};
  key.color_to_move = color;
  key.board_hash = hash;
  key.last_moves.fill(game::kNoopLoc);
  key.komi = komi;
  key.ko = game::kNoopLoc;
  return key;
}

// A result whose every field encodes `v`, so a torn or mixed-up entry shows.
NNInferResult Result(float v) {
  NNInferResult r{};
  r.move_logits.fill(v);
  r.move_probs.fill(v);
  r.value_probs.fill(v);
  r.score_probs.fill(v);
  r.opt_move_probs.fill(v);
  r.err2_outcome = v;
  r.mcts_value_dist.fill(v);
  return r;
}

bool IsResult(const NNInferResult& r, float v) {
  for (float x : r.move_logits)
    if (x != v) return false;
  for (float x : r.move_probs)
    if (x != v) return false;
  for (float x : r.value_probs)
    if (x != v) return false;
  for (float x : r.score_probs)
    if (x != v) return false;
  for (float x : r.opt_move_probs)
    if (x != v) return false;
  for (float x : r.mcts_value_dist)
    if (x != v) return false;
  return r.err2_outcome == v;
}

}  // namespace

TEST_CASE("SharedNNCache: sizes round down to a power of two") {
  CHECK(SharedNNCache(0).num_entries() == 0);
  CHECK(SharedNNCache(1).num_entries() == 1);
  CHECK(SharedNNCache(3).num_entries() == 2);
  CHECK(SharedNNCache(1024).num_entries() == 1024);
  CHECK(SharedNNCache(1500).num_entries() == 1024);
  CHECK(SharedNNCache(size_t{1} << 20).num_entries() == (size_t{1} << 20));
  // Mutex pool: 2^16 by default, at most one per slot.
  CHECK(SharedNNCache(size_t{1} << 20).num_mutexes() == (size_t{1} << 16));
  CHECK(SharedNNCache(1024).num_mutexes() == 1024);
  CHECK(SharedNNCache(1024, /*log2_mutexes=*/4).num_mutexes() == 16);
}

TEST_CASE("SharedNNCache: disabled cache never hits") {
  SharedNNCache cache(0);
  cache.Insert(0, Key(1), Result(1.0f));
  CHECK(!cache.Get(0, Key(1)).has_value());
}

TEST_CASE("SharedNNCache: get after insert, full-key compare") {
  SharedNNCache cache(1024);
  CHECK(!cache.Get(0, Key(7)).has_value());
  cache.Insert(0, Key(7), Result(3.0f));
  std::optional<NNInferResult> hit = cache.Get(0, Key(7));
  REQUIRE(hit.has_value());
  CHECK(IsResult(*hit, 3.0f));
  // Same board hash, different key fields: misses.
  CHECK(!cache.Get(0, Key(7, WHITE)).has_value());
  CHECK(!cache.Get(0, Key(7, BLACK, 6.5f)).has_value());
  NNKey ko = Key(7);
  ko.ko = game::Loc{3, 3};
  CHECK(!cache.Get(0, ko).has_value());
  // Overwrite.
  cache.Insert(0, Key(7), Result(4.0f));
  hit = cache.Get(0, Key(7));
  REQUIRE(hit.has_value());
  CHECK(IsResult(*hit, 4.0f));
}

TEST_CASE("SharedNNCache: lookup / hit counters") {
  SharedNNCache cache(1024);
  CHECK(!cache.Get(0, Key(1)).has_value());
  cache.Insert(0, Key(1), Result(1.0f));
  CHECK(cache.Get(0, Key(1)).has_value());
  CHECK(cache.Get(0, Key(1)).has_value());
  NNCache::Counters c = cache.TakeCounters();
  CHECK(c.lookups == 3);
  CHECK(c.hits == 2);
  c = cache.TakeCounters();
  CHECK(c.lookups == 0);
  CHECK(c.hits == 0);
}

TEST_CASE("SharedNNCache: one slot, inserts overwrite") {
  SharedNNCache cache(1);
  cache.Insert(0, Key(1), Result(1.0f));
  cache.Insert(0, Key(2), Result(2.0f));
  CHECK(!cache.Get(0, Key(1)).has_value());
  std::optional<NNInferResult> hit = cache.Get(0, Key(2));
  REQUIRE(hit.has_value());
  CHECK(IsResult(*hit, 2.0f));
}

// Many threads insert and look up overlapping keys in a small table (heavy
// slot contention and eviction). Every hit must be the result for its key.
TEST_CASE("SharedNNCache: concurrent get / insert stress") {
  static constexpr int kThreads = 32;
  static constexpr int kKeys = 512;
  static constexpr int kIters = 20000;
  SharedNNCache cache(64, /*log2_mutexes=*/3);
  std::atomic<int> bad{0};
  std::atomic<int64_t> hits{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      uint64_t s = 0x9E3779B97F4A7C15ull * (t + 1);
      for (int i = 0; i < kIters; ++i) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        const int k = static_cast<int>(s % kKeys);
        if (std::optional<NNInferResult> hit = cache.Get(0, Key(k))) {
          hits.fetch_add(1, std::memory_order_relaxed);
          if (!IsResult(*hit, static_cast<float>(k))) bad.fetch_add(1);
        } else {
          cache.Insert(0, Key(k), Result(static_cast<float>(k)));
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  CHECK(bad.load() == 0);
  CHECK(hits.load() > 0);
}

TEST_CASE("PerThreadNNCache: per-thread entries, LRU eviction, counters") {
  PerThreadNNCache cache(/*num_threads=*/2, /*total_size=*/4);  // 2 per thread
  cache.Insert(0, Key(1), Result(1.0f));
  CHECK(cache.Get(0, Key(1)).has_value());
  CHECK(!cache.Get(1, Key(1)).has_value());  // thread 1 has its own cache.
  cache.Insert(0, Key(2), Result(2.0f));
  CHECK(cache.Get(0, Key(1)).has_value());  // 1 is now most recently used.
  cache.Insert(0, Key(3), Result(3.0f));    // evicts 2.
  CHECK(cache.Get(0, Key(1)).has_value());
  CHECK(!cache.Get(0, Key(2)).has_value());
  CHECK(cache.Get(0, Key(3)).has_value());
  NNCache::Counters c = cache.TakeCounters();
  CHECK(c.lookups == 6);
  CHECK(c.hits == 4);
}

}  // namespace nn
