#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "cc/constants/constants.h"
#include "cc/game/color.h"
#include "cc/mcts/leaf_evaluator.h"
#include "cc/mcts/search.h"
#include "cc/mcts/tree.h"
#include "doctest/doctest.h"

namespace mcts {
namespace {

constexpr float kVwe = 0.25f;

struct RandomTree {
  std::vector<std::unique_ptr<TreeNode>> nodes;  // nodes[0] is the root
  TreeNode* root() const { return nodes[0].get(); }
};

// Subtree of exactly `size` nodes; n == subtree size, as in a searched tree.
TreeNode* Grow(RandomTree& tree, std::mt19937& rng, int size,
               game::Color color) {
  tree.nodes.push_back(std::make_unique<TreeNode>(tree.nodes.size()));
  TreeNode* node = tree.nodes.back().get();
  node->color_to_move = color;
  node->n = size;
  node->init_outcome_est = std::uniform_real_distribution<float>(-1, 1)(rng);
  node->init_score_est = std::uniform_real_distribution<float>(-40, 40)(rng);
  node->init_score_var = std::uniform_real_distribution<float>(0, 400)(rng);
  int remaining = size - 1;
  if (remaining == 0) {
    node->is_terminal = std::uniform_int_distribution<int>(0, 9)(rng) == 0;
    return node;
  }
  std::vector<int> actions(constants::kMaxMovesPerPosition);
  for (int i = 0; i < (int)actions.size(); ++i) actions[i] = i;
  std::shuffle(actions.begin(), actions.end(), rng);
  const int k =
      std::min(remaining, std::uniform_int_distribution<int>(1, 6)(rng));
  for (int i = 0; i < k; ++i) {
    // Skewed split: the first child takes most of the visits, like a PV.
    const int child_size =
        i == k - 1 ? remaining
                   : std::max(1, std::uniform_int_distribution<int>(
                                     1, std::max(1, remaining * 3 / 4))(rng));
    remaining -= child_size;
    TreeNode* child = Grow(tree, rng, child_size, game::OppositeColor(color));
    node->children[actions[i]].store(child);
    node->child_visits[actions[i]] = child_size;
    if (remaining == 0) break;
  }
  return node;
}

// Independent reference: utilities and stats from scratch against `center`.
void BuildStats(TreeNode* node, TreeNode* root, game::Color root_color,
                float center, const ScoreUtilityParams& params) {
  bool has_children = false;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (node->child(a) == nullptr) continue;
    has_children = true;
    BuildStats(node->child(a), root, root_color, center, params);
  }
  const float c = node->color_to_move == root_color ? center : -center;
  node->init_util_est =
      node == root
          ? node->init_outcome_est
          : node->init_outcome_est +
                ScoreUtility(
                    params, node->init_score_est,
                    node->is_terminal ? 0.0f : std::sqrt(node->init_score_var),
                    c);
  if (has_children) {
    RecomputeNodeStats(node, 0.0f, kVwe);
  } else {
    node->w = node->init_util_est;
    node->v = node->init_util_est;
  }
}

struct Snapshot {
  std::vector<float> w, v, v_var, score, util;
  std::vector<double> v_m3;
};

Snapshot Snap(const RandomTree& tree) {
  Snapshot s;
  for (const auto& node : tree.nodes) {
    s.w.push_back(node->w);
    s.v.push_back(node->v);
    s.v_var.push_back(node->v_var);
    s.v_m3.push_back(node->v_m3);
    s.score.push_back(node->score);
    s.util.push_back(node->init_util_est);
  }
  return s;
}

void CheckEqual(const Snapshot& a, const Snapshot& b) {
  REQUIRE(a.v.size() == b.v.size());
  for (size_t i = 0; i < a.v.size(); ++i) {
    CHECK(a.w[i] == b.w[i]);
    CHECK(a.v[i] == b.v[i]);
    CHECK(a.v_var[i] == b.v_var[i]);
    CHECK(a.v_m3[i] == b.v_m3[i]);
    CHECK(a.score[i] == b.score[i]);
    CHECK(a.util[i] == b.util[i]);
  }
}

TEST_CASE("RecenterTree matches a from-scratch build, in parallel") {
  for (const auto mode :
       {ScoreUtilityMode::kDirect, ScoreUtilityMode::kIntegral}) {
    for (const int seed : {1, 2, 3}) {
      const ScoreUtilityParams params{.score_weight = 0.5f, .mode = mode};
      std::mt19937 rng(seed);
      RandomTree tree;
      Grow(tree, rng, 3000, BLACK);
      const float c0 = 3.5f, c1 = -12.25f;

      // Same center: recentering is a no-op (bitwise).
      BuildStats(tree.root(), tree.root(), BLACK, c0, params);
      const Snapshot built_c0 = Snap(tree);
      const size_t count = RecenterTree(tree.root(), BLACK, c0, params,
                                        /*bias_cache=*/nullptr, kVwe,
                                        /*num_threads=*/32);
      CHECK(count == tree.nodes.size());
      CheckEqual(Snap(tree), built_c0);

      // New center: same as building from scratch with it.
      RecenterTree(tree.root(), BLACK, c1, params, nullptr, kVwe, 32);
      const Snapshot recentered_c1 = Snap(tree);
      BuildStats(tree.root(), tree.root(), BLACK, c1, params);
      CheckEqual(recentered_c1, Snap(tree));
      CHECK(recentered_c1.v[0] != built_c0.v[0]);
    }
  }
}

}  // namespace
}  // namespace mcts
