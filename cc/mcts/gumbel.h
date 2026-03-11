#ifndef MCTS_GUMBEL_H_
#define MCTS_GUMBEL_H_

#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/mcts/leaf_evaluator.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
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
  float kld = 0;
  uint32_t visits = 0;
  // std::string dbg;
};

struct GumbelSearchParams {
  int n;
  int k;
  float noise_scaling = 1.0f;
  bool disable_pass = false;
  bool early_stopping_enabled = false;
  bool over_search_enabled = false;
};

/*
 * Class responsible for executing gumbel search.
 */
class GumbelEvaluator final {
 public:
  GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id);
  GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id,
                  ScoreUtilityParams score_params);
  ~GumbelEvaluator() = default;

  // Disable Copy and Move.
  GumbelEvaluator(GumbelEvaluator const&) = delete;
  GumbelEvaluator& operator=(GumbelEvaluator const&) = delete;
  GumbelEvaluator(GumbelEvaluator&&) = delete;
  GumbelEvaluator& operator=(GumbelEvaluator&&) = delete;

  // Performs a full Gumbel root search. Returns a pair of the original move,
  // and the selected move.
  // If params.n == 1, we will sample a move directly from the policy.
  GumbelResult SearchRoot(core::Probability& probability, game::Game& game,
                          NodeTable* node_table, TreeNode* root,
                          game::Color color_to_move, GumbelSearchParams params);

  // Performs a full PUCT search.
  GumbelResult SearchRootPuct(core::Probability& probability, game::Game& game,
                              NodeTable* node_table, TreeNode* root,
                              game::Color color_to_move, int n,
                              const PuctParams puct_params);

 private:
  static constexpr int kMaxPathLenEst = 128;
  using SearchPath =
      absl::InlinedVector<std::pair<game::Loc, TreeNode*>, kMaxPathLenEst>;

  // Runs Gumbel non-root search path until leaf, and returns the search
  // path including root.
  SearchPath Search(core::Probability& probability, game::Game& game,
                    NodeTable* node_table, TreeNode* node,
                    game::Color color_to_move, game::Color root_color,
                    float root_score_est, SearchPolicy* search_policy);

  // Updates all nodes in tree, based on leaf evaluation.
  void Backward(SearchPath& path, bool use_idempotent_updates);

  // Single Backward Step.
  void SingleBackup(TreeNode* node, game::Loc action, bool is_leaf,
                    float leaf_q, float leaf_q_outcome, float leaf_score,
                    bool is_idempotent = false);

  // Leaf Evaluation module.
  LeafEvaluator leaf_evaluator_;
};

}  // namespace mcts

#endif
