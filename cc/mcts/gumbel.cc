#include "cc/mcts/gumbel.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/core/util.h"
#include "cc/core/vmath.h"
#include "cc/game/board.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/bias_cache.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {
namespace {
using ::game::Color;
using ::game::Game;
using ::game::Loc;

static constexpr float kSmallLogit =
    -10000;  // Not sure how extreme values impact std::exp, so keeping it
             // relatively small.
static constexpr float kMinNonRootDisparity = -100000;
static constexpr int kVisit = 50;
static constexpr float kValueScale = 1.0;

struct GumbelMoveInfo {
  float prob = 0;
  float logit = 0;
  float gumbel_noise = 0;
  float qtransform = 0;
  int move_encoding = game::kInvalidMoveEncoding;
  Loc move_loc = game::kNoopLoc;
};

bool GumbelMoveInfoGreater(const GumbelMoveInfo& x, const GumbelMoveInfo& y) {
  return x.logit + x.gumbel_noise + x.qtransform >
         y.logit + y.gumbel_noise + y.qtransform;
}

int log2(int x) {
  // x needs to be > 0
  int i = 1;
  while (x >> i > 0) {
    ++i;
  }

  return i - 1;
}

inline int ceil_div(int x, int y) { return (x + y - 1) / y; }

// `q`: value estimate of action
// `max_b`: max visit count of all children
float QTransform(float q, float max_b) {
  return (kVisit + max_b) * kValueScale * q;
}

// interpolates NN value evaluation with empirical values derived from visits.
float VMixed(const TreeNode* node) {
  if (SumChildrenN(node) == 0) {
    return node->init_util_est;
  }

  double weighted_visited_q = 0;
  double visited_prob = 0;
  for (int action = 0; action < constants::kMaxMovesPerPosition; ++action) {
    if (NAction(node, action) > 0) {
      weighted_visited_q += (node->move_probs[action] * Q(node, action));
      visited_prob += node->move_probs[action];
    }
  }

  double interpolated_q =
      (weighted_visited_q * SumChildrenN(node) / visited_prob +
       node->init_util_est);

  return interpolated_q / (1 + SumChildrenN(node));
}

int Argmax(const std::array<float, constants::kMaxMovesPerPosition>& arr) {
  int arg_max = 0;
  float max_val = -FLT_MAX;
  for (int i = 0; i < arr.size(); ++i) {
    if (arr[i] > max_val) {
      max_val = arr[i];
      arg_max = i;
    }
  }

  return arg_max;
}

// Samples an action from `policy` (already normalized) with temperature tau.
// Each probability is raised to 1/tau, renormalized, then sampled.
// TODO: remove board param after debugging.
int SampleFromPolicy(
    const std::array<float, constants::kMaxMovesPerPosition>& policy, float tau,
    core::Probability& probability, const game::Board& board) {
  std::array<float, constants::kMaxMovesPerPosition> tempered;
  float total = 0.0f;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    tempered[a] = std::pow(policy[a], 1.0f / tau);
    total += tempered[a];
  }
  float p = probability.Uniform();
  float mass = 0.0f;
  int last_nonzero = -1;
  if (std::isfinite(total) && total > 0.0f) {
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      if (tempered[a] == 0.0f || !std::isfinite(tempered[a])) continue;
      last_nonzero = a;
      float prob = tempered[a] / total;
      if (p >= mass && p < mass + prob) return a;
      mass += prob;
    }
  }
  // Build log message for all failure modes.
  std::ostringstream dbg;
  dbg << "SampleFromPolicy: failed to sample"
      << " (p=" << p << " mass=" << mass << " total=" << total << " tau=" << tau
      << ")\n"
      << game::ToString(board.position()) << "\n";
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (policy[a] == 0.0f) continue;
    dbg << "  " << game::AsLoc(a) << "  policy=" << policy[a]
        << "  tempered=" << tempered[a]
        << "  prob=" << (total > 0.0f ? tempered[a] / total : 0.0f) << "\n";
  }
  // Rounding or nan/inf: last_nonzero is valid, recover and warn.
  if (last_nonzero >= 0) {
    LOG(WARNING) << dbg.str();
    return last_nonzero;
  }
  // Truly degenerate policy (all zeros): crash.
  CHECK(false) << dbg.str();
}

float ComputeKLD(
    const std::array<float, constants::kMaxMovesPerPosition>& target,
    const std::array<float, constants::kMaxMovesPerPosition>& prior) {
  constexpr double kEps = 1e-10;
  double kld = 0.0;
  for (size_t i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    if (target[i] == 0.0f) continue;
    kld += target[i] * std::log(target[i] / (prior[i] + kEps));
  }
  return static_cast<float>(kld);
}

// Compute completedQ = {q(a) if N(a) > 0, v_mixed otherwise }.
// Q values are normalized to [0, 1] before weighting.
std::array<float, constants::kMaxMovesPerPosition> ComputeImprovedPolicy(
    const TreeNode* node) {
  constexpr float kQRange = 3.0f;
  const auto q_norm = [](float q) { return (q + 1.5f) / kQRange; };
  float v_mix = q_norm(VMixed(node));
  const float max_b = 2 * std::log(static_cast<float>(MaxN(node)));
  alignas(MM_ALIGN) std::array<float, constants::kMaxMovesPerPosition>
      logits_improved;
  for (int action = 0; action < constants::kMaxMovesPerPosition; ++action) {
    logits_improved[action] =
        node->move_logits[action] +
        QTransform(NAction(node, action) > 0 ? q_norm(Q(node, action)) : v_mix,
                   max_b);
  }

  return core::SoftmaxV(logits_improved);
}

// Version of the above that only bumps the probabiilities of Gumbel selected
// actions. This avoids choosing moves with low visit counts and high Q-values.
// Q values are normalized from [-1.1, 1.1] to [0, 1] before weighting.
// max_n controls the Q weight: logit[a] += (kVisit + max_n) * q_norm(Q(a)).
std::array<float, constants::kMaxMovesPerPosition> ComputeRootImprovedPolicy(
    TreeNode* node,
    const std::array<float, constants::kMaxMovesPerPosition> masked_logits,
    const absl::InlinedVector<int, 16>& visited_actions, float max_n) {
  constexpr float kQRange = 2.2f;
  const auto q_norm = [](float q) { return (q + 1.1f) / kQRange; };

  float v_mix = q_norm(VMixed(node));
  alignas(MM_ALIGN) std::array<float, constants::kMaxMovesPerPosition>
      logits_improved;
  for (int action = 0; action < constants::kMaxMovesPerPosition; ++action) {
    float q = core::InlinedVecContains(visited_actions, action)
                  ? q_norm(Q(node, action))
                  : v_mix;
    logits_improved[action] =
        masked_logits[action] + (kVisit + max_n) * kValueScale * q;
  }

  return core::SoftmaxV(logits_improved);
}

}  // namespace

Loc GumbelNonRootSearchPolicy::SelectNextAction(const TreeNode* node,
                                                const Game& game,
                                                Color color_to_move,
                                                bool is_root) const {
  DCHECK(node->state != TreeNodeState::kNew);
  std::array<float, constants::kMaxMovesPerPosition> policy_improved =
      ComputeImprovedPolicy(node);

  // select node with greatest disparity between expected value and visit
  // count.
  int selected_action = 0;
  float max_disparity = kMinNonRootDisparity;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    TreeNode* child = node->children[a];
    float disparity =
        policy_improved[a] - (N(child) / (1 + SumChildrenN(node)));
    if (disparity > max_disparity && game.IsValidMove(a, color_to_move)) {
      max_disparity = disparity;
      selected_action = a;
    }
  }

  return game::AsLoc(selected_action);
}

GumbelEvaluator::GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : leaf_evaluator_(nn_interface, thread_id) {}

GumbelEvaluator::GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id,
                                 ScoreUtilityParams score_params)
    : leaf_evaluator_(nn_interface->MakeSlot(0), thread_id, score_params) {}

GumbelEvaluator::GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id,
                                 ScoreUtilityParams score_params,
                                 BiasCache* bias_cache)
    : leaf_evaluator_(nn_interface->MakeSlot(0), thread_id, score_params),
      bias_cache_(bias_cache) {}

GumbelEvaluator::GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id,
                                 BiasCache* bias_cache)
    : leaf_evaluator_(nn_interface, thread_id), bias_cache_(bias_cache) {}

// `n`: total number of simulations.
// `k`: initial number of actions selected.
// `n` must be >= `klogk`.
// !! `game` and `node` must be kept in sync with each other.
GumbelResult GumbelEvaluator::SearchRoot(core::Probability& probability,
                                         Game& game, NodeTable* node_table,
                                         TreeNode* const root,
                                         Color color_to_move,
                                         GumbelSearchParams params) {
  DCHECK(root);
  const int n = params.n;
  int k = params.k;
  const float noise_scaling = params.noise_scaling;
  const bool disable_pass = params.disable_pass;
  const bool early_stopping_enabled = params.early_stopping_enabled && false;
  const bool over_search_enabled = params.over_search_enabled && false;
  const float tau = params.tau;
  int num_rounds = std::max(log2(k), 1);

  if (root->state == TreeNodeState::kNew) {
    leaf_evaluator_.EvaluateRoot(probability, game, root, color_to_move);
    AssignBiasCacheEntry(game, root);
  }

  auto num_moves = constants::kMaxMovesPerPosition;
  auto k_valid = 0;

  std::array<float, constants::kMaxMovesPerPosition>
      masked_logits;  // Logits that zero-out probability for illegal moves.
  GumbelMoveInfo gmove_info[constants::kMaxMovesPerPosition]{};
  for (int i = 0; i < num_moves; ++i) {
    if ((disable_pass && i == constants::kPassMoveEncoding) ||
        !game.IsValidMove(i, color_to_move)) {
      // ignore move henceforth
      masked_logits[i] = kSmallLogit;
      gmove_info[i].logit = kSmallLogit;
      gmove_info[i].gumbel_noise = 0;
      gmove_info[i].qtransform = 0;
      continue;
    }

    masked_logits[i] = root->move_logits[i];

    gmove_info[i].prob = root->move_probs[i];
    gmove_info[i].logit = root->move_logits[i];
    gmove_info[i].gumbel_noise = noise_scaling * probability.GumbelSample();
    gmove_info[i].move_encoding = i;
    gmove_info[i].move_loc = game::AsLoc(i);

    ++k_valid;
  }

  k = std::min(k_valid, k);  // in case we have less valid moves than k.

  // reverse sort
  std::sort(gmove_info, gmove_info + num_moves, GumbelMoveInfoGreater);
  if (n == 1) {
    return GumbelResult{game::AsLoc(Argmax(root->move_logits)),
                        gmove_info[0].move_loc, root->move_probs};
  }

  // populate top-k moves.
  absl::InlinedVector<int, 16> top_k_actions;
  for (int i = 0; i < k; ++i) {
    top_k_actions.emplace_back(gmove_info[i].move_encoding);
  }

  // checks whether none of the moves to be eliminated are the best move with
  // 90% confidence.
  static constexpr int kMinEarlyStoppingVisits = 6;
  const auto bottom_half_ci_gating_check =
      [&gmove_info, &root](const int visits_so_far, const int k) {
        if (visits_so_far < kMinEarlyStoppingVisits) return false;
        float top_lcb_90 = -2;
        float bot_ucb_90 = -2;
        for (auto i = 0; i < k; ++i) {
          auto& move_info = gmove_info[i];
          if (move_info.move_encoding == game::kInvalidMoveEncoding) {
            continue;
          }
          const int a = move_info.move_encoding;
          const int kb = k / 2 + k % 2;
          if (i < k / 2) {
            top_lcb_90 = std::max(top_lcb_90, Lcb(root, a, .05 / kb));
          } else {
            bot_ucb_90 = std::max(bot_ucb_90, Ucb(root, a, .05 / kb));
          }
        }
        return bot_ucb_90 <= top_lcb_90;
      };

  const auto update_qtransform = [&gmove_info, &root](int k) {
    for (auto i = 0; i < k; ++i) {
      auto& move_info = gmove_info[i];
      if (move_info.move_encoding == game::kInvalidMoveEncoding) {
        continue;
      }

      TreeNode* child = root->children[move_info.move_encoding].load(
          std::memory_order_relaxed);
      move_info.qtransform = QTransform(-V(child), MaxN(root));
    }
  };

  // For each round:
  // - Select k top nodes to search.
  // - Reject half for next round.
  // - Divide k by 2
  // - Multiply visits_per_action by 2
  uint32_t visits_spent = 0;

  // Theoretical winner visits: analytically predicted visit count for the
  // winner under sequential halving with no tree reuse.
  // = sum_r round(n / (num_rounds * k_r)) for each halving round r.
  int theoretical_winner_visits = 0;
  {
    int k_tmp = k;
    while (k_tmp > 1) {
      theoretical_winner_visits +=
          std::round(float(n) / float(num_rounds * k_tmp));
      k_tmp /= 2;
    }
  }

  const int m = k;
  while (k > 1) {
    // We are supporting three search modes:
    // 1. normal
    // 2. early stopping
    // - every visits_per_action / 4 moves, check whether we hit some confidence
    // check and stop early if we do.
    // 3. over search
    // - do at least visits_per_action, and up to visits_per_action * 2.5. do
    // the same confidence check, using the base visits_per_action / 4 interval.
    const int v = std::round(float(n) / float(num_rounds * k));
    const auto [visits_per_action, early_stopping_check_interval,
                min_check_interval] = [early_stopping_enabled,
                                       over_search_enabled,
                                       v]() -> std::tuple<int, int, int> {
      if (early_stopping_enabled) {
        return {v, ceil_div(v, 4),
                std::max(v / 4, kMinEarlyStoppingVisits) - 1};
      } else if (over_search_enabled) {
        return {v * 5 / 2, ceil_div(v, 4), v - 1};
      }

      return {v, v, v};
    }();
    int visits_in_round = 0;
    for (int visit_num = 0; visit_num < visits_per_action; ++visit_num) {
      ++visits_in_round;
      for (auto i = 0; i < k; ++i) {
        auto& move_info = gmove_info[i];
        if (move_info.move_encoding == game::kInvalidMoveEncoding) {
          // We have less valid moves than k.
          continue;
        }

        Game search_game = game;
        search_game.PlayMove(move_info.move_loc, color_to_move);

        // This branch should only trigger once in this search.
        if (!root->children[move_info.move_encoding]) {
          // After the move, it's the opponent's turn.
          root->children[move_info.move_encoding] = node_table->GetOrCreate(
              search_game.board().hash(), game::OppositeColor(color_to_move),
              search_game.IsGameOver());
        }
        TreeNode* child = root->children[move_info.move_encoding];
        const bool enable_var_scaling =
            params.nonroot_var_scale_prior_visits >= 0;
        PuctSearchPolicy nonroot_policy(
            PuctParams::Builder()
                .set_enable_var_scaling(enable_var_scaling)
                .set_var_scale_prior_visits(
                    enable_var_scaling ? params.nonroot_var_scale_prior_visits
                                       : 0)
                .build());
        SearchPath search_path = Search(
            probability, search_game, node_table, child,
            game::OppositeColor(color_to_move), color_to_move,
            root->init_score_est, &nonroot_policy, /*first_is_root=*/false);

        // update tree
        const bool use_idempotent_updates =
            node_table->is_graph() || (bias_cache_ != nullptr);
        Backward(search_path, use_idempotent_updates);
        root->child_visits[move_info.move_encoding] += 1;
        // TODO: Update root max_child_n?
#if 0
        if (root->child_visits[move_info.move_encoding] > root->max_child_n) {
          root->max_child_n = root->child_visits[move_info.move_encoding];
        }
#endif
        ++visits_spent;
      }

      if (visit_num % early_stopping_check_interval ==
              (early_stopping_check_interval - 1) &&
          visit_num >= min_check_interval) {
        update_qtransform(k);
        std::sort(gmove_info, gmove_info + k, GumbelMoveInfoGreater);
        if (bottom_half_ci_gating_check(visit_num, k)) {
          break;
        }
      }
    }

    update_qtransform(k);
    std::sort(gmove_info, gmove_info + k, GumbelMoveInfoGreater);
    k /= 2;
  }

  AdvanceState(root);

  // Get improved policy from completed-Q values.
  // max_n = 2 * ln(theoretical winner visits) controls Q weight.
  const float visit_advantage =
      2 * std::log(static_cast<float>(theoretical_winner_visits + 1));
  std::array<float, constants::kMaxMovesPerPosition> pi_improved =
      ComputeRootImprovedPolicy(root, masked_logits, top_k_actions,
                                visit_advantage);

  Loc raw_nn_move = game::AsLoc(Argmax(root->move_logits));
  Loc mcts_move = tau > 0.0f ? game::AsLoc(SampleFromPolicy(
                                   pi_improved, tau, probability, game.board()))
                             : gmove_info[0].move_loc;

  GumbelResult result = {raw_nn_move, mcts_move, pi_improved};

  // Populate stats for visited children.
  for (int i = 0; i < m; ++i) {
    const GumbelMoveInfo& gmove = gmove_info[i];
    result.child_stats.emplace_back(ChildStats{
        gmove.move_loc /* move */,
        static_cast<int>(NAction(root, gmove.move_encoding)) /* n */,
        Q(root, gmove.move_encoding) /* q */,
        QOutcome(root, gmove.move_encoding) /* qz */,
        ChildScore(root, gmove.move_encoding) /* score */,
        gmove.prob /* prob */,
        gmove.logit /* logit */,
        gmove.gumbel_noise /* gumbel_noise */,
        gmove.qtransform /* qtransform */,
        pi_improved[gmove.move_encoding] /* improved_policy */,
    });
  }

  // Update root, adding all visits from children, but only q estimates from
  // the move that MCTS selects. This avoids corrupting the root value estimate
  // with moves that gumbel forces MCTS to play. Since we will not use the root
  // on subsequent time steps, we can afford to do this.
  for (const auto& child_stat : result.child_stats) {
    if (child_stat.move == result.mcts_move) {
      root->w += child_stat.n * child_stat.q;
      root->w_outcome += child_stat.n * child_stat.qz;

      int total_visits = root->n + child_stat.n;
      float root_ratio = root->n / static_cast<float>(total_visits);
      float child_ratio = child_stat.n / static_cast<float>(total_visits);
      root->v = root_ratio * root->v + child_ratio * child_stat.q;
      root->v_outcome =
          root_ratio * root->v_outcome + child_ratio * child_stat.qz;

#ifdef V_CATEGORICAL
      // Update categorical distribution.
      TreeNode* child = root->children[child_stat.move];
      if (child != nullptr) {
        for (int v_bucket = 0; v_bucket < kNumVBuckets; ++v_bucket) {
          // Mirror buckets, since we should flip signs.
          root->v_categorical[v_bucket] +=
              child->v_categorical[kNumVBuckets - v_bucket - 1];
        }
      }
#endif
    }

    root->n += child_stat.n;
  }

  result.kld = ComputeKLD(result.pi_improved, root->move_probs);
  result.visits = visits_spent;
  return result;
}

// Search using PUCT to select root actions. Still uses Gumbel planning at
// non-root nodes.
GumbelResult GumbelEvaluator::SearchRootPuct(core::Probability& probability,
                                             game::Game& game,
                                             NodeTable* node_table,
                                             TreeNode* root,
                                             game::Color color_to_move, int n,
                                             const PuctParams puct_params) {
  auto best_lcb_move = [&](const TreeNode* node) {
    std::array<std::pair<int, float>, constants::kMaxMovesPerPosition>
        move_lcbs;
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      move_lcbs[a] = {a, Lcb(node, a)};
    }

    std::sort(
        move_lcbs.begin(), move_lcbs.end(),
        [](const auto& p0, const auto& p1) { return p1.second < p0.second; });

    for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
      auto [a, _] = move_lcbs[i];
      if (game.IsValidMove(a, color_to_move)) {
        return game::AsLoc(a);
      }
    }

    return game::kPassLoc;
  };

  if (root->state == TreeNodeState::kNew) {
    leaf_evaluator_.EvaluateRoot(probability, game, root, color_to_move);
  }

  // Freeze visit counts, so we can reconstruct stats later.
  std::array<float, constants::kMaxMovesPerPosition> visit_counts_pre_search;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    visit_counts_pre_search[a] = NAction(root, a);
  }

  // Conduct search.
  PuctSearchPolicy search_policy(puct_params);
  for (size_t _ = 0; _ < n; ++_) {
    Game search_game = game;
    SearchPath search_path =
        Search(probability, search_game, node_table, root, color_to_move,
               color_to_move, root->init_score_est, &search_policy,
               /*first_is_root=*/true);
    const bool use_idempotent_updates =
        node_table->is_graph() || (bias_cache_ != nullptr);
    Backward(search_path, use_idempotent_updates);
  }

  // Build result.
  std::array<float, constants::kMaxMovesPerPosition> visit_counts;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    visit_counts[a] = NAction(root, a) - visit_counts_pre_search[a];
  }

  // Improved policy = visit counts normalized as probabilities.
  std::array<float, constants::kMaxMovesPerPosition> pi_improved;
  {
    float total = core::SumV(visit_counts);
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      pi_improved[a] = total > 0.0f ? visit_counts[a] / total : 0.0f;
    }
  }

  Loc raw_nn_move = game::AsLoc(Argmax(root->move_logits));
  Loc mcts_move = [&]() {
    switch (puct_params.kind) {
      case PuctRootSelectionPolicy::kVisitCount:
        return game::AsLoc(Argmax(visit_counts));
      case PuctRootSelectionPolicy::kLcb:
        return best_lcb_move(root);
      case PuctRootSelectionPolicy::kVisitCountSample:
        return game::AsLoc(SampleFromPolicy(pi_improved, puct_params.tau,
                                            probability, game.board()));
    }

    return game::AsLoc(Argmax(visit_counts));
  }();

  absl::InlinedVector<ChildStats, 16> child_stats;
  for (int mv = 0; mv < constants::kMaxMovesPerPosition; ++mv) {
    if (visit_counts[mv] == 0) continue;

    child_stats.emplace_back(ChildStats{
        game::AsLoc(mv),
        static_cast<int>(visit_counts[mv]),
        Q(root, mv),
        QOutcome(root, mv),
        ChildScore(root, mv),
        root->move_probs[mv],
        root->move_logits[mv],
        0.0f,
        0.0f,
        pi_improved[mv] /* improved_policy */,
    });
  }
  GumbelResult result = {raw_nn_move, mcts_move, pi_improved,
                         std::move(child_stats)};
  return result;
}

// `board`: Local search board
// `node`: Tree node corresponding to local search board
// `moves`: Local moves vector
// `color_to_move`: Color whose turn it is to move next
// `root_score_est`: Value estimate for root node. Subsequent node score
// estimates will be centered against this value.
GumbelEvaluator::SearchPath GumbelEvaluator::Search(
    core::Probability& probability, Game& game, NodeTable* node_table,
    TreeNode* node, Color color_to_move, Color root_color, float root_score_est,
    SearchPolicy* search_policy, bool first_is_root) {
  SearchPath path = {{game::kNoopLoc, node}};
  if (node->state == TreeNodeState::kNew) {
    // leaf node. evaluate and return.
    leaf_evaluator_.EvaluateLeaf(probability, game, node, color_to_move,
                                 root_color, root_score_est);
    AssignBiasCacheEntry(game, node);
    return path;
  }

  // internal node. Trace a single path until we hit a leaf.
  bool is_first_node = first_is_root;
  while (path.back().second->state != TreeNodeState::kNew &&
         !(path.back().second->is_terminal) && !game.IsGameOver()) {
    auto& [action, node] = path.back();
    Loc selected_action = search_policy->SelectNextAction(
        node, game, color_to_move, is_first_node);
    is_first_node = false;
    CHECK(game.PlayMove(selected_action, color_to_move));
    if (!node->children[selected_action]) {
      // After the move, it's the opponent's turn
      Color next_color = game::OppositeColor(color_to_move);
      TreeNode* child = node_table->GetOrCreate(game.board().hash(), next_color,
                                                game.IsGameOver());
      node->children[selected_action] = child;
    }
    action = selected_action;
    path.push_back({game::kNoopLoc, node->children[selected_action]});
    color_to_move = game::OppositeColor(color_to_move);
    AdvanceState(node);
  }

  // either we have reached a leaf node, or we have reached the end of the game,
  // or both.
  auto& [_, leaf_node] = path.back();
  if (leaf_node->state == TreeNodeState::kNew) {
    leaf_evaluator_.EvaluateLeaf(probability, game, leaf_node, color_to_move,
                                 root_color, root_score_est);
  }

  if (game.IsGameOver() && !leaf_node->is_terminal) {
    // evaluate score
    game::Scores scores = game.GetScores();
    leaf_evaluator_.EvaluateTerminal(scores, leaf_node, color_to_move,
                                     root_color, root_score_est);
  }

  AssignBiasCacheEntry(game, leaf_node);

  return path;
}

void GumbelEvaluator::AssignBiasCacheEntry(const game::Game& game,
                                           TreeNode* node) {
  if (!bias_cache_) return;
  std::optional<LocalPattern> local_pattern =
      LocalPattern::FromCurrentPosition(game);
  if (!local_pattern.has_value()) return;
  node->bias_cache_entry = bias_cache_->GetOrCreate(*local_pattern);
}

void GumbelEvaluator::Backward(SearchPath& path, bool use_idempotent_updates) {
  auto [_, leaf] = path.back();
  float leaf_q = leaf->init_util_est;
  float leaf_q_outcome = leaf->init_outcome_est;
  float leaf_score = leaf->init_score_est;

  for (int i = path.size() - 1; i >= 0; --i) {
    auto [parent_action, parent] = path[i];
    float leaf_q_mult =
        leaf->color_to_move == parent->color_to_move ? 1.0f : -1.0f;

    SingleBackup(parent, parent_action, i == path.size() - 1,
                 leaf_q_mult * leaf_q, leaf_q_mult * leaf_q_outcome,
                 leaf_q_mult * leaf_score, use_idempotent_updates);
  }
}

void GumbelEvaluator::SingleBackup(TreeNode* node, game::Loc action,
                                   bool is_leaf, float leaf_q,
                                   float leaf_q_outcome, float leaf_score,
                                   bool is_idempotent) {
  if (is_leaf) {
    node->n += 1;
    node->w = node->init_util_est;
    node->w_outcome = node->init_outcome_est;
    node->v = node->init_util_est;
    node->v_outcome = node->init_outcome_est;
    return;
  }

  const int a = game::AsIndex(action, BOARD_LEN);
  float v_old = node->v, v_outcome_old = node->v_outcome, n_old = node->n;
  node->n += 1;
  node->child_visits[a] += 1;
  if (is_idempotent) {
    // recompute.
    const float obs_bias = bias_cache_ != nullptr && node->bias_cache_entry
                               ? bias_cache_->UpdateAndFetch(node)
                               : 0.0f;
    RecomputeNodeStats(node, obs_bias);
  } else {
    // incremental
    node->w += leaf_q;
    node->w_outcome += leaf_q_outcome;
    node->v = node->w / node->n;
    node->v_outcome = node->w_outcome / node->n;
    node->score = leaf_score * (1.0f / N(node)) +
                  node->score * ((N(node) - 1.0f) / N(node));
    // Update 3rd moment.
    const auto compute_m3 = [](const double m3, const double m2, const double d,
                               const double n) {
      const double d3 = d * d * d;
      return m3 + ((n * n - 1) * d3 / (n * n)) - (3 * d * m2 / n);
    };
    const float m3 = compute_m3(node->v_m3 * n_old, node->v_var * n_old,
                                leaf_q - node->v, n_old);
    const float m3_outcome =
        compute_m3(node->v_outcome_m3 * n_old, node->v_outcome_var * n_old,
                   leaf_q_outcome - node->v_outcome, n_old);
    node->v_m3 = m3 / node->n;
    node->v_outcome_m3 = m3_outcome / node->n;
    // Update variance.
    node->v_var = n_old * node->v_var + (leaf_q - v_old) * (leaf_q - node->v);
    node->v_var /= node->n;
    node->v_outcome_var =
        n_old * node->v_outcome_var +
        (leaf_q_outcome - v_outcome_old) * (leaf_q_outcome - node->v_outcome);
    node->v_outcome_var /= node->n;
    int child_n = node->child_visits[a];
    if (child_n > node->max_child_n) {
      node->max_child_n = child_n;
    }
  }

#ifdef V_CATEGORICAL
  // Add V to bucket.
  int v_bucket =
      std::clamp(static_cast<int>((leaf_q_outcome + 1.0f) / kBucketRange), 0,
                 kNumVBuckets - 1);
  node->v_categorical[v_bucket] += 1;
#endif
}

}  // namespace mcts
