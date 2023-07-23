#include "cc/mcts/gumbel.h"

#include "cc/core/doctest_include.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/mcts/constants.h"
#include "cc/mcts/gumbel.h"
#include "cc/nn/nn_interface.h"

namespace mcts {
namespace {
using core::Probability;
using game::Color;
using game::Game;
using game::Loc;
using nn::NNInterface;
static constexpr int kGumbelN = 8;
static constexpr int kGumbelK = 4;

// The tree will also assign all subchildren under child 0 with q=-.5, child 1
// with q=-.5 + 1/3, child 2 with q=-.5 + 2/3, and child 3 with q=.5. MCTS
// should conclude that each subtree has the given qs, and select child `3` as
// the best move.
void InitValueFields(TreeNode* node, int child_index, Color color_to_move,
                     float root_score_est, int depth) {
  node->color_to_move = color_to_move;

  float q_mult = color_to_move == BLACK ? 1.0 : -1.0;
  float q = depth == 0 ? 0 : q_mult * (-0.5 + child_index * (1.0 / 3.0));
  node->outcome_est = q;
  node->score_est = 0.0;

  node->w = q;
  node->v = node->w;
  node->init_util_est = node->w;
}

// This builds a tree that consistently assigns logit 2 to child 0, and logit 1
// to all others.
// Customize the field initialization logic with `Func`.
template <typename Fn>
void BuildMctsTree(Fn init_fn, TreeNode* node, int child_index,
                   Color color_to_move, int depth, int max_depth,
                   float root_score_est) {
  init_fn(node, child_index, color_to_move, root_score_est, depth);

  // Our prior move is 0, but we should choose 3.
  for (int i = 0; i < kGumbelK; ++i) {
    node->move_logits[i] = 1;
    node->move_probs[i] = 0.2;
  }

  node->move_logits[0] = 2;
  node->move_probs[0] = 0.4;

  if (depth == max_depth) {
    return;
  }

  for (int i = 0; i < kGumbelK; ++i) {
    node->children[i] = std::make_unique<mcts::TreeNode>();
    auto child = node->children[i].get();

    // All nodes below the first layer of the tree should have the same value.
    auto next_child_index = depth == 0 ? i : child_index;
    BuildMctsTree(init_fn, child, next_child_index,
                  game::OppositeColor(color_to_move), depth + 1, max_depth,
                  root_score_est);
  }
}

}  // namespace

TEST_CASE("GumbelTest") {
  NNInterface nn_interface(0);
  GumbelEvaluator gumbel_evaluator(&nn_interface, 0);
  Probability probability(0);
  Game game;

  game::Loc expected_nn_move = Loc{0, 0};
  game::Loc expected_gumbel_move = Loc{0, 3};

  std::unique_ptr<mcts::TreeNode> root_node =
      std::make_unique<mcts::TreeNode>();
  BuildMctsTree(InitValueFields, root_node.get(),
                -1 /* hack to initialize root q to 0. */, BLACK, 0 /* depth */,
                3 /* max_depth */, 5);

  // Build an MCTS tree but do not advance node states. This will "trick" the
  // evaluator into believing the nodes are new. We link dummy implementations
  // for gumbel and leaf evaluation for testing.
  GumbelResult gumbel_result = gumbel_evaluator.SearchRoot(
      probability, game, root_node.get(), BLACK, kGumbelN, kGumbelK);

  // Check move predictions are correct.
  CHECK(gumbel_result.nn_move == expected_nn_move);
  CHECK(gumbel_result.mcts_move == expected_gumbel_move);

  // Sanity check Q values and visit counts at children.
  int expected_root_n = 9;
  CHECK(root_node->n == expected_root_n);
  CHECK(root_node->max_child_n == 3);
  CHECK(root_node->children[0]->n == 1);
  CHECK(root_node->children[1]->n == 1);
  CHECK(root_node->children[2]->n == 3);
  CHECK(root_node->children[3]->n == 3);

  float q0 = -.5;
  float q1 = -.5 + 1. / 3;
  float q2 = -.5 + 2. / 3;
  float q3 = .5;
  float expected_root_q =
      (q0 + q1 + 3 * q2 + 3 * q3) / static_cast<float>(expected_root_n);

  CHECK(root_node->v == doctest::Approx(expected_root_q));

  // Flip q-values because these are from the perspective of the opponent.
  CHECK(-root_node->children[0]->v == doctest::Approx(q0));
  CHECK(-root_node->children[1]->v == doctest::Approx(q1));
  CHECK(-root_node->children[2]->v == doctest::Approx(q2));
  CHECK(-root_node->children[3]->v == doctest::Approx(q3));
}
}  // namespace mcts
