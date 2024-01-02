#include "cc/mcts/gumbel.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "absl/log/check.h"
#include "cc/constants/constants.h"
#include "cc/core/util.h"
#include "cc/core/vmath.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {
namespace {
using ::game::Color;
using ::game::Game;
using ::game::Loc;

static constexpr float kSmallLogit =
    -1000;  // Not sure how extreme values impact std::exp, so keeping it
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

// `q`: value estimate of action
// `max_b`: max visit count of all children
float QTransform(float q, int max_b) {
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

int Argmax(std::array<float, constants::kMaxMovesPerPosition>& arr) {
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

// Compute completedQ = {q(a) if N(a) > 0, v_mixed otherwise }.
std::array<float, constants::kMaxMovesPerPosition> ComputeImprovedPolicy(
    const TreeNode* node) {
  float v_mix = VMixed(node);
  int max_n = MaxN(node);
  alignas(MM_ALIGN) std::array<float, constants::kMaxMovesPerPosition>
      logits_improved;
  for (int action = 0; action < constants::kMaxMovesPerPosition; ++action) {
    logits_improved[action] =
        node->move_logits[action] +
        QTransform(NAction(node, action) > 0 ? Q(node, action) : v_mix, max_n);
  }

  return core::SoftmaxV(logits_improved);
}

// Version of the above that only bumps the probabiilities of Gumbel selected
// actions. This avoids choosing moves with low visit counts and high Q-values.
std::array<float, constants::kMaxMovesPerPosition> ComputeRootImprovedPolicy(
    TreeNode* node,
    const std::array<float, constants::kMaxMovesPerPosition> masked_logits,
    const absl::InlinedVector<int, 16>& visited_actions) {
  float v_mix = VMixed(node);
  int max_n = MaxN(node);
  alignas(MM_ALIGN) std::array<float, constants::kMaxMovesPerPosition>
      logits_improved;
  for (int action = 0; action < constants::kMaxMovesPerPosition; ++action) {
    if (core::InlinedVecContains(visited_actions, action)) {
      logits_improved[action] =
          masked_logits[action] + QTransform(Q(node, action), max_n);
    } else {
      logits_improved[action] =
          masked_logits[action] + QTransform(v_mix, max_n);
    }
  }

  return core::SoftmaxV(logits_improved);
}

}  // namespace

Loc GumbelNonRootSearchPolicy::SelectNextAction(const TreeNode* node,
                                                const Game& game,
                                                Color color_to_move) const {
  DCHECK(node->state != TreeNodeState::kNew);
  std::array<float, constants::kMaxMovesPerPosition> policy_improved =
      ComputeImprovedPolicy(node);

  // select node with greatest disparity between expected value and visit
  // count.
  int selected_action = 0;
  float max_disparity = kMinNonRootDisparity;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    TreeNode* child = node->children[a].get();
    float disparity =
        policy_improved[a] - (N(child) / (1 + SumChildrenN(node)));
    if (disparity > max_disparity && game.IsValidMove(a, color_to_move)) {
      max_disparity = disparity;
      selected_action = a;
    }
  }

  return game::AsLoc(selected_action);
}

PuctSearchPolicy::PuctSearchPolicy(float c_puct, float c_puct_visit_scaling)
    : c_puct_(c_puct), c_puct_visit_scaling_(c_puct_visit_scaling) {}

Loc PuctSearchPolicy::SelectNextAction(const TreeNode* node,
                                       const game::Game& game,
                                       game::Color color_to_move) const {
  static constexpr float kFPU = 0.2f;
  DCHECK(node->state != TreeNodeState::kNew);

  // Scale c_puct according to visit count.
  const float c_puct =
      c_puct_ + c_puct_visit_scaling_ * std::log((N(node) + 500.0f) / 500.0f);

  // Find total explored policy.
  float p_explored = 0.0f;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (N(node->children[a].get()) > 0) {
      p_explored += node->move_probs[a];
    }
  }

  // Fallback V for unexplored children.
  float v_fpu = V(node) - kFPU * std::sqrt(p_explored);

  // Compute PUCT values.
  std::array<std::pair<int, float>, constants::kMaxMovesPerPosition> pucts{};
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    TreeNode* child = node->children[a].get();
    float u = c_puct * node->move_probs[a] *
              (std::sqrt(N(node)) / (1 + NAction(node, a)));
    float v = child == nullptr ? v_fpu : Q(node, a);
    pucts[a] = {a, u + v};
  }

  // Reverse sort so the highest puct value is in front.
  std::sort(pucts.begin(), pucts.end(), [](const auto& p0, const auto& p1) {
    return p0.second > p1.second;
  });

  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    auto [a, _] = pucts[i];
    if (game.IsValidMove(a, color_to_move)) {
      return game::AsLoc(a);
    }
  }

  // No valid moves. Should never get here.
  return game::kNoopLoc;
}

GumbelEvaluator::GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : leaf_evaluator_(nn_interface, thread_id) {}

// `n`: total number of simulations.
// `k`: initial number of actions selected.
// `n` must be >= `klogk`.
// !! `game` and `node` must be kept in sync with each other.
GumbelResult GumbelEvaluator::SearchRoot(core::Probability& probability,
                                         Game& game, TreeNode* const root,
                                         Color color_to_move, int n, int k,
                                         float noise_scaling) {
  DCHECK(root);
  int num_rounds = std::max(log2(k), 1);

  if (root->state == TreeNodeState::kNew) {
    leaf_evaluator_.EvaluateRoot(probability, game, root, color_to_move);
  }

  auto num_moves = constants::kMaxMovesPerPosition;
  auto k_valid = 0;

  std::array<float, constants::kMaxMovesPerPosition>
      masked_logits;  // Logits that zero-out probability for illegal moves.
  GumbelMoveInfo gmove_info[num_moves];
  for (int i = 0; i < num_moves; ++i) {
    if (!game.IsValidMove(i, color_to_move)) {
      // ignore move henceforth
      masked_logits[i] = kSmallLogit;
      gmove_info[i].logit = kSmallLogit;
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

  // For each round:
  // - Select k top nodes to search.
  // - Reject half for next round.
  // - Divide k by 2
  // - Multiply visits_per_action by 2
  int n_remaining = n;
  int m = k;
  while (k > 1) {
    // Visits for this round: floor(n / (num_rounds * k)). If k ~ {2, 3}, then
    // this is the last round.
    int visits_per_action = k < 4 ? (n_remaining / k) : (n / (num_rounds * k));
    for (auto i = 0; i < k; ++i) {
      auto& move_info = gmove_info[i];
      if (move_info.move_encoding == game::kInvalidMoveEncoding) {
        // We have less valid moves than k.
        continue;
      }

      if (!root->children[move_info.move_encoding]) {
        root->children[move_info.move_encoding] = std::make_unique<TreeNode>();
      }

      TreeNode* child = root->children[move_info.move_encoding].get();
      for (int _ = 0; _ < visits_per_action; ++_) {
        Game search_game = game;
        search_game.PlayMove(move_info.move_loc, color_to_move);

        GumbelNonRootSearchPolicy search_policy;
        absl::InlinedVector<TreeNode*, kMaxPathLenEst> search_path = Search(
            probability, search_game, child, game::OppositeColor(color_to_move),
            color_to_move, root->score_est, &search_policy);

        // update tree
        Backward(search_path);

        // update budget
        --n_remaining;
      }

      // update qvalue
      auto child_q = -V(child);
      move_info.qtransform = QTransform(child_q, MaxN(root));
    }

    std::sort(gmove_info, gmove_info + k, GumbelMoveInfoGreater);
    k /= 2;
  }

  AdvanceState(root);

  Loc raw_nn_move = game::AsLoc(Argmax(root->move_logits));
  GumbelResult result = {raw_nn_move, gmove_info[0].move_loc, {}};

  // Get improved policy from completed-Q values.
  result.pi_improved =
      ComputeRootImprovedPolicy(root, masked_logits, top_k_actions);

  // result.pi_improved[gmove_info[0].move_encoding] = 1.0f;  // one-hot.

  // Populate stats for visited children.
  for (int i = 0; i < m; ++i) {
    const GumbelMoveInfo& gmove = gmove_info[i];
    result.child_stats.emplace_back(
        ChildStats{gmove.move_loc /* move */,
                   static_cast<int>(NAction(root, gmove.move_encoding)) /* n */,
                   Q(root, gmove.move_encoding) /* q */,
                   QOutcome(root, gmove.move_encoding) /* qz */,
                   ChildScore(root, gmove.move_encoding) /* score */,
                   gmove.prob /* prob */, gmove.logit /* logit */,
                   gmove.gumbel_noise /* gumbel_noise */,
                   gmove.qtransform /* qtransform */});
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
      TreeNode* child = root->children[child_stat.move].get();
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

  return result;
}

// Search using PUCT to select root actions. Still uses Gumbel planning at
// non-root nodes.
GumbelResult GumbelEvaluator::SearchRootPuct(core::Probability& probability,
                                             game::Game& game, TreeNode* root,
                                             game::Color color_to_move, int n,
                                             const float c_puct) {
  if (root->state == TreeNodeState::kNew) {
    leaf_evaluator_.EvaluateRoot(probability, game, root, color_to_move);
  }

  // Freeze visit counts, so we can reconstruct stats later.
  std::array<float, constants::kMaxMovesPerPosition> visit_counts_pre_search;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    visit_counts_pre_search[a] = NAction(root, a);
  }

  // Conduct search.
  PuctSearchPolicy search_policy(c_puct, 0.45);
  for (size_t _ = 0; _ < n; ++_) {
    Game search_game = game;
    absl::InlinedVector<TreeNode*, kMaxPathLenEst> search_path =
        Search(probability, search_game, root, color_to_move, color_to_move,
               root->score_est, &search_policy);
    Backward(search_path);
  }

  // Build result.
  std::array<float, constants::kMaxMovesPerPosition> visit_counts;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    visit_counts[a] = NAction(root, a) - visit_counts_pre_search[a];
  }

  Loc raw_nn_move = game::AsLoc(Argmax(root->move_logits));
  Loc mcts_move = game::AsLoc(Argmax(visit_counts));
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
    });
  }
  GumbelResult result = {raw_nn_move, mcts_move, visit_counts,
                         std::move(child_stats)};

  return result;
}

// `board`: Local search board
// `node`: Tree node corresponding to local search board
// `moves`: Local moves vector
// `color_to_move`: Color whose turn it is to move next
// `root_score_est`: Value estimate for root node. Subsequent node score
// estimates will be centered against this value.
absl::InlinedVector<TreeNode*, GumbelEvaluator::kMaxPathLenEst>
GumbelEvaluator::Search(core::Probability& probability, Game& game,
                        TreeNode* node, Color color_to_move, Color root_color,
                        float root_score_est, SearchPolicy* search_policy) {
  absl::InlinedVector<TreeNode*, kMaxPathLenEst> path = {node};
  if (node->state == TreeNodeState::kNew) {
    // leaf node. evaluate and return.
    leaf_evaluator_.EvaluateLeaf(probability, game, node, color_to_move,
                                 root_color, root_score_est);
    return path;
  }

  // internal node. Trace a single path until we hit a leaf.
  while (path.back()->state != TreeNodeState::kNew &&
         !(path.back()->is_terminal) && !game.IsGameOver()) {
    TreeNode* node = path.back();
    Loc selected_action =
        search_policy->SelectNextAction(node, game, color_to_move);

    if (!node->children[selected_action]) {
      node->children[selected_action] = std::make_unique<TreeNode>();
    }

    Loc move_loc = game::AsLoc(selected_action);
    game.PlayMove(move_loc, color_to_move);
    path.emplace_back(node->children[selected_action].get());
    color_to_move = game::OppositeColor(color_to_move);
    AdvanceState(node);
  }

  // either we have reached a leaf node, or we have reached the end of the game,
  // or both.
  TreeNode* leaf_node = path.back();
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

  return path;
}

void GumbelEvaluator::Backward(
    absl::InlinedVector<TreeNode*, kMaxPathLenEst>& path) {
  TreeNode* leaf = path.back();
  float leaf_q = leaf->v;
  float leaf_q_outcome = leaf->v_outcome;

  for (int i = path.size() - 2; i >= 0; --i) {
    TreeNode* parent = path[i];
    TreeNode* child = path[i + 1];
    float leaf_q_mult =
        leaf->color_to_move == parent->color_to_move ? 1.0f : -1.0f;

    SingleBackup(parent, child->n, leaf_q_mult * leaf_q,
                 leaf_q_mult * leaf_q_outcome);
  }
}

void GumbelEvaluator::SingleBackup(TreeNode* node, int child_n, float leaf_q,
                                   float leaf_q_outcome) {
  float v_old = node->v, v_outcome_old = node->v_outcome, n_old = node->n;
  node->n += 1;
  node->w += leaf_q;
  node->w_outcome += leaf_q_outcome;
  node->v = node->w / node->n;
  node->v_outcome = node->w_outcome / node->n;
  if (child_n > node->max_child_n) {
    node->max_child_n = child_n;
  }

  // Update variance.
  node->v_var = n_old * node->v_var + (leaf_q - v_old) * (leaf_q - node->v);
  node->v_var /= node->n;
  node->v_outcome_var = n_old * node->v_outcome_var +
                        (leaf_q - v_outcome_old) * (leaf_q - node->v_outcome);
  node->v_outcome_var /= node->n;

#ifdef V_CATEGORICAL
  // Add V to bucket.
  int v_bucket =
      std::clamp(static_cast<int>((leaf_q_outcome + 1.0f) / kBucketRange), 0,
                 kNumVBuckets - 1);
  node->v_categorical[v_bucket] += 1;
#endif
}

}  // namespace mcts
