#ifndef MCTS_GUMBEL_H_
#define MCTS_GUMBEL_H_

#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/mcts/leaf_evaluator.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {

struct GumbelResult {
  game::Loc nn_move;
  game::Loc mcts_move;
  std::array<float, constants::kMaxNumMoves> pi_improved;
};

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
  // If n == 1, we will sample a move directly from the policy.
  GumbelResult SearchRoot(core::Probability& probability, game::Game& game,
                          TreeNode* root, game::Color color_to_move, int n,
                          int k);

 private:
  static constexpr int kMaxPathLenEst = 128;
  // Runs Gumbel non-root search path until leaf, and returns the search path
  // including root.
  absl::InlinedVector<TreeNode*, kMaxPathLenEst> SearchNonRoot(
      core::Probability& probability, game::Game& game, TreeNode* root,
      TreeNode* node, game::Color color_to_move, game::Color root_color,
      float root_score_est);

  // Updates all nodes in tree, based on leaf evaluation.
  void Backward(absl::InlinedVector<TreeNode*, kMaxPathLenEst>& path);

  // Leaf Evaluation module.
  LeafEvaluator leaf_evaluator_;
};

}  // namespace mcts

#endif
