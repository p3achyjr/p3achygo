#include "cc/mcts/leaf_evaluator.h"

namespace mcts {
using ::game::Color;
using ::game::Game;
using ::game::Loc;

LeafEvaluator::LeafEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : nn_interface_(nn_interface), thread_id_(thread_id) {}

void LeafEvaluator::EvaluateLeaf(Game& game, TreeNode* node,
                                 Color color_to_move, float root_score_est) {
  InitTreeNode(node, game, color_to_move);
}

void LeafEvaluator::InitTreeNode(TreeNode* node, const Game& game,
                                 Color color_to_move) {
  node->n = 1;
  AdvanceState(node);
}
}  // namespace mcts
