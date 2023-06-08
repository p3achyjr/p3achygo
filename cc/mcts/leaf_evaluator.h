#ifndef __CC_LEAF_EVALUATOR_H_
#define __CC_LEAF_EVALUATOR_H_

#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {

/*
 * Wrapper Class for Leaf Evaluation. Mainly useful for testing.
 */
class LeafEvaluator final {
 public:
  LeafEvaluator(nn::NNInterface* nn_interface, int thread_id);
  ~LeafEvaluator() = default;

  // Disable Copy and Move.
  LeafEvaluator(LeafEvaluator const&) = delete;
  LeafEvaluator& operator=(LeafEvaluator const&) = delete;
  LeafEvaluator(LeafEvaluator&&) = delete;
  LeafEvaluator& operator=(LeafEvaluator&&) = delete;

  // Calls `InitTreeNode` and fills initial stats.
  void EvaluateLeaf(game::Game& game, TreeNode* node, game::Color color_to_move,
                    float root_score_estimate);

  // Evaluates a leaf node using the neural net.
  void InitTreeNode(TreeNode* node, const game::Game& game,
                    game::Color color_to_move);

 private:
  nn::NNInterface* nn_interface_;
  int thread_id_;
};
}  // namespace mcts

#endif
