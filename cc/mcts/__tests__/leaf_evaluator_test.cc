#include "cc/mcts/leaf_evaluator.h"

#include "cc/core/doctest_include.h"
#include "cc/nn/nn_interface.h"

namespace mcts {
namespace {
using namespace ::nn;
using namespace ::game;
using namespace ::core;

static constexpr float kScoreWeight = 0.5f;
static constexpr float kRootScore = 15;
}  // namespace

TEST_CASE("LeafEvaluatorTest") {
  NNInterface nn_interface(0);  // Should link a dummy implementation
  LeafEvaluator leaf_evaluator(&nn_interface, 0, kScoreWeight);
  Probability probability(0);
  Game game;

  std::unique_ptr<mcts::TreeNode> node = std::make_unique<mcts::TreeNode>();
  leaf_evaluator.EvaluateLeaf(probability, game, node.get(),
                              BLACK /* color_to_move */, BLACK /* root_color */,
                              kRootScore /* root_score_est */);

  CHECK(node->init_util_est ==
        doctest::Approx(ScoreTransform(kScoreWeight, 0, kRootScore)));

  node = std::make_unique<mcts::TreeNode>();
  leaf_evaluator.EvaluateLeaf(probability, game, node.get(),
                              BLACK /* color_to_move */, WHITE /* root_color */,
                              kRootScore /* root_score_est */);
  CHECK(node->init_util_est ==
        doctest::Approx(ScoreTransform(kScoreWeight, 0, -kRootScore)));

  node = std::make_unique<mcts::TreeNode>();
  leaf_evaluator.EvaluateLeaf(probability, game, node.get(),
                              WHITE /* color_to_move */, WHITE /* root_color */,
                              kRootScore /* root_score_est */);
  CHECK(node->init_util_est ==
        doctest::Approx(ScoreTransform(kScoreWeight, 0, kRootScore)));

  node = std::make_unique<mcts::TreeNode>();
  leaf_evaluator.EvaluateLeaf(probability, game, node.get(),
                              WHITE /* color_to_move */, BLACK /* root_color */,
                              kRootScore /* root_score_est */);
  CHECK(node->init_util_est ==
        doctest::Approx(ScoreTransform(kScoreWeight, 0, -kRootScore)));
}

}  // namespace mcts
