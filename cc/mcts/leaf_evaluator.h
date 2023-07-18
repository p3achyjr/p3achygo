#ifndef MCTS_LEAF_EVALUATOR_H_
#define MCTS_LEAF_EVALUATOR_H_

#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {

inline float ScoreTransform(float c_score, float score_est,
                            float root_score_est) {
  return c_score * M_2_PI * std::atan((score_est - root_score_est) / BOARD_LEN);
}

/*
 * Wrapper Class for Leaf Evaluation. Mainly useful for testing.
 */
class LeafEvaluator final {
 public:
  LeafEvaluator(nn::NNInterface* nn_interface, int thread_id);
  LeafEvaluator(nn::NNInterface* nn_interface, int thread_id,
                float score_weight);
  ~LeafEvaluator() = default;

  // Disable Copy and Move.
  LeafEvaluator(LeafEvaluator const&) = delete;
  LeafEvaluator& operator=(LeafEvaluator const&) = delete;
  LeafEvaluator(LeafEvaluator&&) = delete;
  LeafEvaluator& operator=(LeafEvaluator&&) = delete;

  // Calls `InitTreeNode` and fills initial stats for unevaluated root nodes.
  void EvaluateRoot(core::Probability& probability, const game::Game& game,
                    TreeNode* node, game::Color color_to_move);

  // Calls `InitTreeNode` and fills initial stats.
  void EvaluateLeaf(core::Probability& probability, const game::Game& game,
                    TreeNode* node, game::Color color_to_move,
                    game::Color root_color, float root_score_estimate);

  // Populates a terminal node.
  void EvaluateTerminal(const game::Scores& scores, TreeNode* terminal_node,
                        game::Color color_to_move, game::Color root_color,
                        float root_score_estimate);

  // Evaluates a leaf node using the neural net.
  void InitTreeNode(core::Probability& probability, TreeNode* node,
                    const game::Game& game, game::Color color_to_move);

 private:
  // Populates initial fields _after_ a call to InitTreeNode.
  void InitFields(TreeNode* node, float score_utility);

  nn::NNInterface* nn_interface_;
  int thread_id_;
  const float score_weight_;
};
}  // namespace mcts

#endif
