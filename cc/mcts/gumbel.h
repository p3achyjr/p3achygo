#ifndef MCTS_GUMBEL_H_
#define MCTS_GUMBEL_H_

#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/mcts/bias_cache.h"
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
  float var;
  float prob;
  float logit;
  float gumbel_noise;
  float qtransform;
  float improved_policy;  // improved policy probability for this move
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
  int n = 0;
  int k = 0;
  float noise_scaling = 1.0f;
  bool disable_pass = false;
  bool early_stopping_enabled = false;
  bool over_search_enabled = false;
  // If > 0, sample mcts_move from the improved policy with temperature tau
  // (each probability raised to 1/tau, then renormalized). tau == 0 takes the
  // sequential-halving winner directly.
  float tau = 0.0f;
  // Prior visits for nonroot var-scaling PUCT. -1 disables var scaling.
  int nonroot_var_scale_prior_visits = 10;

  // Sigma (Q transform) used for the root improved policy and, optionally,
  // the sequential-halving comparisons.
  //   kProd: current behavior. Halving uses (50 + MaxN(root)) * q; the target
  //          uses (50 + 2 ln(N_win + 1)) * q_norm(q).
  //   kSe:   sigma(q) = (c_visit + c * sqrt(N_eff) / sigma_q) * q in raw-Q
  //          units, N_eff = min visits over the live candidate set. The logit
  //          gap between two actions is then their Q gap in standard-error
  //          units (times c). See notes/claude/variance-aware-selection.md.
  enum class SigmaMode : uint8_t { kProd = 0, kSe = 1 };
  SigmaMode sigma_mode = SigmaMode::kProd;
  // kSe only: also use the SE sigma inside sequential halving. Otherwise
  // halving keeps the kProd form and only the target changes.
  bool sigma_se_halving = false;
  float sigma_se_c = 1.2f;
  // Per-visit stddev of backed-up values. <= 0 derives it from the root's
  // visited children (visit-weighted RMS of child stddev).
  float sigma_se_sigma_q = 0.4f;
  // Floor on the coefficient, raw-Q units (the paper's c_visit role).
  float sigma_se_c_visit = 0.0f;
  // kSe only: which N_eff the final target uses. kMinTopK = min visits over
  // all top-k candidates (very small after halving). kWinner = visits of the
  // selected move, the analogue of prod's theoretical_winner_visits.
  enum class SigmaSeNEff : uint8_t { kMinTopK = 0, kWinner = 1 };
  SigmaSeNEff sigma_se_n_eff = SigmaSeNEff::kMinTopK;

  // Forward-declared; defined below once GumbelSearchParams is complete.
  class Builder;
};

class GumbelSearchParams::Builder {
 public:
  Builder() = default;
  Builder& set_n(int v) {
    p_.n = v;
    return *this;
  }
  Builder& set_k(int v) {
    p_.k = v;
    return *this;
  }
  Builder& set_noise_scaling(float v) {
    p_.noise_scaling = v;
    return *this;
  }
  Builder& set_disable_pass(bool v) {
    p_.disable_pass = v;
    return *this;
  }
  Builder& set_early_stopping_enabled(bool v) {
    p_.early_stopping_enabled = v;
    return *this;
  }
  Builder& set_over_search_enabled(bool v) {
    p_.over_search_enabled = v;
    return *this;
  }
  Builder& set_tau(float v) {
    p_.tau = v;
    return *this;
  }
  Builder& set_nonroot_var_scale_prior_visits(int v) {
    p_.nonroot_var_scale_prior_visits = v;
    return *this;
  }
  Builder& set_sigma_mode(GumbelSearchParams::SigmaMode v) {
    p_.sigma_mode = v;
    return *this;
  }
  Builder& set_sigma_se_halving(bool v) {
    p_.sigma_se_halving = v;
    return *this;
  }
  Builder& set_sigma_se_c(float v) {
    p_.sigma_se_c = v;
    return *this;
  }
  Builder& set_sigma_se_sigma_q(float v) {
    p_.sigma_se_sigma_q = v;
    return *this;
  }
  Builder& set_sigma_se_c_visit(float v) {
    p_.sigma_se_c_visit = v;
    return *this;
  }
  Builder& set_sigma_se_n_eff(GumbelSearchParams::SigmaSeNEff v) {
    p_.sigma_se_n_eff = v;
    return *this;
  }
  GumbelSearchParams build() const { return p_; }

 private:
  GumbelSearchParams p_;
};

/*
 * Class responsible for executing gumbel search.
 */
class GumbelEvaluator final {
 public:
  GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id);
  GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id,
                  ScoreUtilityParams score_params);
  GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id,
                  ScoreUtilityParams score_params, BiasCache* bias_cache);
  // Convenience ctor for self-play: no custom ScoreUtilityParams needed.
  GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id,
                  BiasCache* bias_cache);
  ~GumbelEvaluator() = default;

  // Disable Copy and Move.
  GumbelEvaluator(GumbelEvaluator const&) = delete;
  GumbelEvaluator& operator=(GumbelEvaluator const&) = delete;
  GumbelEvaluator(GumbelEvaluator&&) = delete;
  GumbelEvaluator& operator=(GumbelEvaluator&&) = delete;

  // Sample raw from policy
  GumbelResult Sample(core::Probability& probability, game::Game& game,
                      TreeNode* root, game::Color color_to_move,
                      float tau = 1.0f);

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
                    float root_score_est, SearchPolicy* search_policy,
                    bool first_is_root);

  // Updates all nodes in tree, based on leaf evaluation.
  void Backward(SearchPath& path, const PuctParams puct_params,
                bool use_idempotent_updates);

  // Single Backward Step.
  void SingleBackup(TreeNode* node, game::Loc action, bool is_leaf,
                    float leaf_q, float leaf_q_outcome, float leaf_score,
                    const PuctParams puct_params, bool is_idempotent = false);

  // Assigns bias_cache_entry to node based on the current game position.
  void AssignBiasCacheEntry(const game::Game& game, TreeNode* node);

  // Leaf Evaluation module.
  LeafEvaluator leaf_evaluator_;
  BiasCache* bias_cache_ = nullptr;
};

float ComputeKLD(
    const std::array<float, constants::kMaxMovesPerPosition>& target,
    const std::array<float, constants::kMaxMovesPerPosition>& prior);

std::array<float, constants::kMaxMovesPerPosition> ComputeImprovedPolicy(
    const TreeNode* node, const int n);

}  // namespace mcts

#endif
