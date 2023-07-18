#include "cc/game/board.h"
#include "cc/mcts/constants.h"
#include "cc/mcts/leaf_evaluator.h"

namespace mcts {
using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::game::Scores;

LeafEvaluator::LeafEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : nn_interface_(nn_interface),
      thread_id_(thread_id),
      score_weight_(kDefaultScoreWeight) {}

void LeafEvaluator::EvaluateRoot(core::Probability& probability,
                                 const game::Game& game, TreeNode* node,
                                 game::Color color_to_move) {
  // Call for any not-yet-evaluated root nodes.
  InitTreeNode(probability, node, game, color_to_move);
}

void LeafEvaluator::EvaluateLeaf(core::Probability& probability,
                                 const game::Game& game, TreeNode* node,
                                 game::Color color_to_move,
                                 game::Color root_color,
                                 float root_score_estimate) {
  InitTreeNode(probability, node, game, color_to_move);
}

void LeafEvaluator::EvaluateTerminal(const Scores& scores,
                                     TreeNode* terminal_node,
                                     Color color_to_move, Color root_color,
                                     float root_score_estimate) {}

void LeafEvaluator::InitTreeNode(core::Probability& probability, TreeNode* node,
                                 const game::Game& game,
                                 game::Color color_to_move) {
  node->n = 1;
  AdvanceState(node);
}
}  // namespace mcts
