#ifndef __MCTS_GUMBEL_H_
#define __MCTS_GUMBEL_H_

#include "cc/game/board.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {

/*
 * Class responsible for executing gumbel search.
 */
class GumbelEvaluator final {
 public:
  game::Loc SearchRoot(game::Board& board, TreeNode* node,
                       std::vector<game::Loc>& moves, int color_to_move, int n,
                       int k);
  void SearchNonRoot(game::Board& board, TreeNode* node,
                     std::vector<game::Loc>& moves, int color_to_move,
                     float root_score_estimate);

  void EvaluateLeaf(game::Board& board, TreeNode* node,
                    std::vector<game::Loc>& moves, int color_to_move,
                    float root_score_estimate);

 private:
  void InitTreeNode(TreeNode* node, const game::Board& board,
                    const std::vector<game::Loc>& moves, int color_to_move);
  nn::NNInterface nn_interface_;
};

}  // namespace mcts

#endif