#ifndef __MCTS_GUMBEL_H_
#define __MCTS_GUMBEL_H_

#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/mcts/leaf_evaluator.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {

/*
 * Class responsible for executing gumbel search.
 */
class GumbelEvaluator final {
 public:
  GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id);
  ~GumbelEvaluator() = default;

  // Disable Copy and Move.
  GumbelEvaluator(GumbelEvaluator const&) = delete;
  GumbelEvaluator& operator=(GumbelEvaluator const&) = delete;
  GumbelEvaluator(GumbelEvaluator&&) = delete;
  GumbelEvaluator& operator=(GumbelEvaluator&&) = delete;

  // Performs a full Gumbel root search. Returns a pair of the original move,
  // and the selected move.
  std::pair<game::Loc, game::Loc> SearchRoot(core::Probability& probability,
                                             game::Game& game, TreeNode* node,
                                             game::Color color_to_move, int n,
                                             int k);

 private:
  // Runs Gumbel non-root search path until leaf, and returns the search path
  // excluding root.
  std::vector<TreeNode*> SearchNonRoot(game::Game& game, TreeNode* node,
                                       game::Color color_to_move,
                                       float root_score_est);

  // Calls `InitTreeNode` and fills initial stats.
  void EvaluateLeaf(game::Game& game, TreeNode* node, game::Color color_to_move,
                    float root_score_est);

  // Evaluates a leaf node using the neural net.
  void InitTreeNode(TreeNode* node, const game::Game& game,
                    game::Color color_to_move);

  // Updates all nodes in tree, based on leaf evaluation.
  void Backward(std::vector<TreeNode*>& path);

  std::unique_ptr<LeafEvaluator> leaf_evaluator_;
};

}  // namespace mcts

#endif
