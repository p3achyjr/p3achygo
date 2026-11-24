#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "cc/mcts/bias_cache.h"

#include <atomic>
#include <cmath>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "cc/mcts/tree.h"
#include "doctest/doctest.h"

namespace mcts {
namespace {

LocalPattern Pattern(uint64_t id) {
  LocalPattern
      p;  // Move/Loc have no default initializers; set every key field.
  p.last_move = game::Move{BLACK, game::Loc{3, 3}};
  p.two_moves_ago = game::Move{WHITE, game::Loc{15, 15}};
  p.grid_hash = id;
  return p;
}

// Many threads update nodes that share a few entries while nodes are destroyed
// (TreeNode::~TreeNode removes 80% of the node's contribution) and another
// thread creates and prunes unrelated entries. Each entry must end up holding
// exactly the sum of its nodes' last contributions.
TEST_CASE("BiasCache - concurrent updates, node destruction and pruning") {
  constexpr int kEntries = 8, kThreads = 16, kNodesPerThread = 64,
                kIters = 3000;
  constexpr float kForget = 0.8f;
  BiasCache cache(/*alpha=*/1.0f, /*lambda=*/0.5f);
  std::vector<std::shared_ptr<BiasCache::Entry>> entries;
  for (int k = 0; k < kEntries; ++k)
    entries.push_back(cache.GetOrCreate(Pattern(k)));

  struct Contribution {
    int entry;
    double obs, weight;  // the node's last terms, scaled by what remains
  };
  std::vector<std::vector<Contribution>> contributions(kThreads);
  std::atomic<bool> stop_pruner = false;
  std::thread pruner([&]() {
    uint64_t id = kEntries;
    while (!stop_pruner.load()) {
      {
        auto unused = cache.GetOrCreate(Pattern(id++));
      }
      cache.PruneUnused();
    }
  });

  std::vector<std::thread> workers;
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&, t]() {
      std::mt19937 rng(t);
      std::uniform_real_distribution<float> util(-1.0f, 1.0f);
      std::vector<std::unique_ptr<TreeNode>> nodes, children;
      std::vector<int> lifetime;
      for (int i = 0; i < kNodesPerThread; ++i) {
        nodes.push_back(std::make_unique<TreeNode>(i));
        children.push_back(std::make_unique<TreeNode>(i));
        nodes[i]->children[0].store(children[i].get());
        nodes[i]->bias_cache_entry = cache.GetOrCreate(Pattern(i % kEntries));
        nodes[i]->init_util_est = util(rng);
        // Half the nodes are destroyed partway through.
        lifetime.push_back(i % 2 == 0 ? kIters
                                      : std::uniform_int_distribution<int>(
                                            1, kIters - 1)(rng));
      }
      for (int it = 0; it < kIters; ++it) {
        for (int i = 0; i < kNodesPerThread; ++i) {
          if (!nodes[i]) continue;
          nodes[i]->child_visits[0] = it + 1;
          nodes[i]->n = it + 2;
          children[i]->v = util(rng);
          cache.UpdateAndFetch(nodes[i].get());
          if (it + 1 == lifetime[i]) {
            contributions[t].push_back(
                {i % kEntries, (1.0 - kForget) * nodes[i]->last_obs_bias_term,
                 (1.0 - kForget) * nodes[i]->last_weight_term});
            nodes[i].reset();
          }
        }
      }
      for (int i = 0; i < kNodesPerThread; ++i) {
        if (!nodes[i]) continue;
        contributions[t].push_back({i % kEntries, nodes[i]->last_obs_bias_term,
                                    nodes[i]->last_weight_term});
        nodes[i]->bias_cache_entry.reset();  // keep the contribution in place
      }
    });
  }
  for (auto& w : workers) w.join();
  stop_pruner = true;
  pruner.join();

  std::vector<double> obs(kEntries, 0.0), weight(kEntries, 0.0),
      obs_scale(kEntries, 0.0);
  for (const auto& per_thread : contributions) {
    for (const auto& c : per_thread) {
      obs[c.entry] += c.obs;
      weight[c.entry] += c.weight;
      obs_scale[c.entry] += std::abs(c.obs);
    }
  }
  for (int k = 0; k < kEntries; ++k) {
    // Float accumulation over ~10^5 deltas per entry; a lost update would be
    // off by a whole term (weight terms here are ~10^3).
    CHECK(std::abs(entries[k]->weight_sum - weight[k]) <= 1e-4 * weight[k] + 1);
    CHECK(std::abs(entries[k]->obs_bias_sum - obs[k]) <=
          1e-4 * obs_scale[k] + 1);
  }
}

}  // namespace
}  // namespace mcts
