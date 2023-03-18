#ifndef __MCTS_GUMBEL_H_
#define __MCTS_GUMBEL_H_

#include "cc/core/probability.h"
#include "cc/game/board.h"
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
                                             game::Board& board, TreeNode* node,
                                             std::vector<game::Loc>& moves,
                                             int color_to_move, int n, int k);

 private:
  // Runs Gumbel non-root search path until leaf, and returns the search path
  // excluding root.
  std::vector<TreeNode*> SearchNonRoot(game::Board& board, TreeNode* node,
                                       std::vector<game::Loc>& moves,
                                       int color_to_move,
                                       float root_score_estimate);

  // Calls `InitTreeNode` and fills initial stats.
  void EvaluateLeaf(game::Board& board, TreeNode* node,
                    std::vector<game::Loc>& moves, int color_to_move,
                    float root_score_estimate);

  // Evaluates a leaf node using the neural net.
  void InitTreeNode(TreeNode* node, const game::Board& board,
                    const std::vector<game::Loc>& moves, int color_to_move);

  // Updates all nodes in tree, based on leaf evaluation.
  void Backward(std::vector<TreeNode*>& path);

  nn::NNInterface* nn_interface_;
  int thread_id_;
};

}  // namespace mcts

#endif