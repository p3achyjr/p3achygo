#include "cc/nn/nn_cache.h"

#include <atomic>
#include <random>
#include <thread>
#include <vector>

#include "cc/core/doctest_include.h"
#include "cc/game/loc.h"

namespace nn {
namespace {

NNKey MakeTestKey(uint64_t h, game::Loc ko = game::kNoopLoc) {
  NNKey key{};
  key.color_to_move = BLACK;
  key.board_hash = game::Zobrist::Hash(h);
  key.last_moves.fill(game::kNoopLoc);
  key.komi = 7.5f;
  key.ko = ko;
  return key;
}

NNInferResult MakeTestResult(float v) {
  NNInferResult r{};
  r.move_logits.fill(v);
  r.err2_outcome = v;
  return r;
}

}  // namespace

TEST_CASE("NNCache - miss, insert, hit") {
  NNCache cache(/*log2_size=*/10, /*log2_mutexes=*/4);
  const NNKey key = MakeTestKey(42);
  CHECK(cache.Get(key) == nullptr);
  cache.Insert(key, MakeTestResult(3.0f));
  std::shared_ptr<const NNCache::Entry> hit = cache.Get(key);
  REQUIRE(hit != nullptr);
  CHECK(hit->first == key);
  CHECK(hit->second.err2_outcome == 3.0f);
  CHECK(hit->second.move_logits[17] == 3.0f);
}

TEST_CASE("NNCache - full key compared; insert overwrites the slot") {
  // One slot: every key maps to it.
  NNCache cache(/*log2_size=*/0, /*log2_mutexes=*/16);
  const NNKey k1 = MakeTestKey(1);
  const NNKey k2 = MakeTestKey(2);
  const NNKey k1_ko = MakeTestKey(1, game::Loc{3, 3});
  cache.Insert(k1, MakeTestResult(1.0f));
  CHECK(cache.Get(k2) == nullptr);
  CHECK(cache.Get(k1_ko) == nullptr);
  std::shared_ptr<const NNCache::Entry> held = cache.Get(k1);
  REQUIRE(held != nullptr);
  cache.Insert(k2, MakeTestResult(2.0f));
  CHECK(cache.Get(k1) == nullptr);
  REQUIRE(cache.Get(k2) != nullptr);
  CHECK(cache.Get(k2)->second.err2_outcome == 2.0f);
  // A held entry stays valid after its slot is overwritten.
  CHECK(held->first == k1);
  CHECK(held->second.err2_outcome == 1.0f);
}

TEST_CASE("NNCache - concurrent inserts and gets") {
  // Small table (heavy slot collisions) and few mutexes (heavy lock sharing).
  // Every hit must carry the value written for its key.
  NNCache cache(/*log2_size=*/6, /*log2_mutexes=*/2);
  static constexpr int kThreads = 16;
  static constexpr int kOps = 50000;
  static constexpr int kKeys = 512;
  std::atomic<int64_t> hits = 0, bad = 0;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 rng(1234 + t);
      for (int i = 0; i < kOps; ++i) {
        const uint64_t k = rng() % kKeys;
        const NNKey key = MakeTestKey(k);
        if (rng() % 2) {
          cache.Insert(key, MakeTestResult(float(k)));
        } else if (std::shared_ptr<const NNCache::Entry> e = cache.Get(key)) {
          hits++;
          if (!(e->first == key) || e->second.err2_outcome != float(k) ||
              e->second.move_logits[100] != float(k)) {
            bad++;
          }
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  CHECK(hits.load() > 0);
  CHECK(bad.load() == 0);
}

}  // namespace nn
