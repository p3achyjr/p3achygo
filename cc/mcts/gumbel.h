#ifndef MCTS_GUMBEL_H_
#define MCTS_GUMBEL_H_

#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/mcts/leaf_evaluator.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {

struct ChildStats {
  game::Loc move;
  int n;
  float q;
  float qz;
  float score;
  float prob;
  float logit;
  float gumbel_noise;
  float qtransform;
};

struct GumbelResult {
  game::Loc nn_move;
  game::Loc mcts_move;
  std::array<float, constants::kMaxMovesPerPosition> pi_improved;
  absl::InlinedVector<ChildStats, 16> child_stats;
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
                          int k, float noise_scaling);
  inline GumbelResult SearchRoot(core::Probability& probability,
                                 game::Game& game, TreeNode* root,
                                 game::Color color_to_move, int n, int k) {
    return SearchRoot(probability, game, root, color_to_move, n, k, 1.0f);
  }

  // Uses PUCT formula to select actions at the root, but uses Q-based planning
  // at non-root nodes, as described in the Gumbel paper.
  GumbelResult SearchRootPuct(core::Probability& probability, game::Game& game,
                              TreeNode* root, game::Color color_to_move, int n,
                              const float c_puct);

 private:
  static constexpr int kMaxPathLenEst = 128;
  // Runs Gumbel non-root search path until leaf, and returns the search path
  // including root.
  absl::InlinedVector<TreeNode*, kMaxPathLenEst> SearchNonRoot(
      core::Probability& probability, game::Game& game, TreeNode* node,
      game::Color color_to_move, game::Color root_color, float root_score_est);

  // Updates all nodes in tree, based on leaf evaluation.
  void Backward(absl::InlinedVector<TreeNode*, kMaxPathLenEst>& path);

  // Single Backward Step.
  void SingleBackup(TreeNode* node, int child_n, float leaf_q,
                    float leaf_q_outcome);

  // Leaf Evaluation module.
  LeafEvaluator leaf_evaluator_;
};

}  // namespace mcts

#endif
