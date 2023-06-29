#include "cc/mcts/gumbel.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "cc/constants/constants.h"
#include "cc/core/util.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {
namespace {
using ::game::Color;
using ::game::Game;
using ::game::Loc;

static constexpr float kSmallLogit = -100000;
static constexpr float kMinNonRootDisparity = -100000;
static constexpr int kVisit = 50;
static constexpr float kValueScale = 1.0;

struct GumbelMoveInfo {
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
float VMixed(TreeNode* node) {
  if (SumChildrenN(node) == 0) {
    return node->init_util_est;
  }

  double weighted_visited_q = 0;
  double visited_prob = 0;
  for (int action = 0; action < constants::kMaxNumMoves; ++action) {
    if (NAction(node, action) > 0) {
      weighted_visited_q += (node->move_probs[action] * QAction(node, action));
      visited_prob += node->move_probs[action];
    }
  }

  double interpolated_q =
      (weighted_visited_q * SumChildrenN(node) / visited_prob +
       node->init_util_est);

  return interpolated_q / (1 + SumChildrenN(node));
}

std::array<float, constants::kMaxNumMoves> Softmax(
    const std::array<float, constants::kMaxNumMoves>& logits) {
  double max = *std::max_element(logits.begin(), logits.end());

  std::array<double, constants::kMaxNumMoves> norm_logits;
  std::transform(logits.begin(), logits.end(), norm_logits.begin(),
                 [&](float x) { return x - max; });
  std::array<double, constants::kMaxNumMoves> exps;
  std::transform(norm_logits.begin(), norm_logits.end(), exps.begin(),
                 [&](double x) { return std::exp(x); });
  double total = std::accumulate(exps.begin(), exps.end(), 0.0);

  std::array<float, constants::kMaxNumMoves> softmax;
  std::transform(exps.begin(), exps.end(), softmax.begin(),
                 [&](double x) { return x / total; });
  return softmax;
}

int Argmax(std::array<float, constants::kMaxNumMoves>& logits) {
  int arg_max = 0;
  float max_logit = kSmallLogit;
  for (int i = 0; i < logits.size(); ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
      arg_max = i;
    }
  }

  return arg_max;
}

std::array<float, constants::kMaxNumMoves> ComputeImprovedPolicy(
    TreeNode* node) {
  float v_mix = VMixed(node);
  int max_n = MaxN(node);
  std::array<float, constants::kMaxNumMoves> logits_improved;
  for (int action = 0; action < constants::kMaxNumMoves; ++action) {
    logits_improved[action] =
        node->move_logits[action] +
        QTransform(NAction(node, action) > 0 ? QAction(node, action) : v_mix,
                   max_n);
  }

  return Softmax(logits_improved);
}

}  // namespace

GumbelEvaluator::GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : leaf_evaluator_(nn_interface, thread_id) {}

// `n`: total number of simulations.
// `k`: initial number of actions selected.
// `n` must be >= `klogk`.
// !! `game` and `node` must be kept in sync with each other.
GumbelResult GumbelEvaluator::SearchRoot(core::Probability& probability,
                                         Game& game, TreeNode* const root,
                                         Color color_to_move, int n, int k) {
  CHECK(root);
  int num_rounds = std::max(log2(k), 1);
  int visits_per_action = n / (k * num_rounds);

  if (root->state == TreeNodeState::kNew) {
    EvaluateRoot(game, root, color_to_move);
  }

  auto num_moves = constants::kMaxNumMoves;
  auto k_valid = 0;
  GumbelMoveInfo gmove_info[num_moves];
  for (int i = 0; i < num_moves; ++i) {
    if (!game.IsValidMove(i, color_to_move)) {
      // ignore move henceforth
      gmove_info[i].logit = kSmallLogit;
      continue;
    }

    gmove_info[i].logit = root->move_logits[i];
    gmove_info[i].gumbel_noise = probability.GumbelSample();
    gmove_info[i].move_encoding = i;
    gmove_info[i].move_loc = game::AsLoc(i);

    ++k_valid;
  }

  k = std::min(k_valid, k);  // in case we have less valid moves than k.

  // reverse sort
  std::sort(gmove_info, gmove_info + num_moves, GumbelMoveInfoGreater);

  // For each round:
  // - Select k top nodes to search.
  // - Reject half for next round.
  // - Divide k by 2
  // - Multiply visits_per_action by 2
  while (k > 1) {
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

        absl::InlinedVector<TreeNode*, kMaxPathLenEst> search_path =
            SearchNonRoot(search_game, root, child,
                          game::OppositeColor(color_to_move), root->score_est);

        // update tree
        Backward(search_path);
      }

      // update qvalue
      auto child_q = -Q(child);
      move_info.qtransform = QTransform(child_q, MaxN(root));
    }

    std::sort(gmove_info, gmove_info + k, GumbelMoveInfoGreater);
    k /= 2;
    visits_per_action *= 2;
  }

  AdvanceState(root);

  Loc raw_nn_move = game::AsLoc(Argmax(root->move_logits));
  GumbelResult result = {raw_nn_move, gmove_info[0].move_loc};

  // Get improved policy from completed-Q values.
  result.pi_improved = ComputeImprovedPolicy(root);

  return result;
}

// `board`: Local search board
// `node`: Tree node corresponding to local search board
// `moves`: Local moves vector
// `color_to_move`: Color whose turn it is to move next
// `root_score_est`: Value estimate for root node. Subsequent node score
// estimates will be centered against this value.
absl::InlinedVector<TreeNode*, GumbelEvaluator::kMaxPathLenEst>
GumbelEvaluator::SearchNonRoot(Game& game, TreeNode* root, TreeNode* node,
                               Color color_to_move, float root_score_est) {
  absl::InlinedVector<TreeNode*, kMaxPathLenEst> path = {root, node};
  if (node->state == TreeNodeState::kNew) {
    // leaf node. evaluate and return.
    EvaluateLeaf(game, node, color_to_move, root_score_est);
    return path;
  }

  // internal node. Trace a single path until we hit a leaf, using a
  // deterministic paradigm.
  while (path.back()->state != TreeNodeState::kNew &&
         !(path.back()->is_terminal) && !game.IsGameOver()) {
    auto node = path.back();
    std::array<float, constants::kMaxNumMoves> policy_improved =
        ComputeImprovedPolicy(node);

    // select node with greatest disparity between expected value and visit
    // count.
    int selected_action = 0;
    float max_disparity = kMinNonRootDisparity;
    for (int i = 0; i < constants::kMaxNumMoves; ++i) {
      TreeNode* child = node->children[i].get();
      float disparity =
          policy_improved[i] - (N(child) / (1 + SumChildrenN(node)));
      if (disparity > max_disparity && game.IsValidMove(i, color_to_move)) {
        max_disparity = disparity;
        selected_action = i;
      }
    }

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
    EvaluateLeaf(game, leaf_node, color_to_move, root_score_est);
  }

  if (game.IsGameOver() && !leaf_node->is_terminal) {
    // evaluate score
    game::Scores scores = game.GetScores();
    float player_score =
        color_to_move == BLACK ? scores.black_score : scores.white_score;
    float opp_score =
        color_to_move == BLACK ? scores.white_score : scores.black_score;
    // float final_score =
    //     player_score - opp_score + constants::kScoreInflectionPoint;
    // float empirical_q =
    //     (player_score > opp_score ? 1.0 : -1.0) +
    //     ScoreTransform(final_score, root_score_est, BOARD_LEN);

    // TODO: Experiment with this.
    float empirical_q = player_score > opp_score ? 1.5 : -1.5;

    leaf_node->is_terminal = true;
    leaf_node->q = empirical_q;
  }

  return path;
}

void GumbelEvaluator::EvaluateRoot(const Game& game, TreeNode* node,
                                   Color color_to_move) {
  leaf_evaluator_.EvaluateRoot(game, node, color_to_move);
}

void GumbelEvaluator::EvaluateLeaf(const Game& game, TreeNode* node,
                                   Color color_to_move, float root_score_est) {
  leaf_evaluator_.EvaluateLeaf(game, node, color_to_move, root_score_est);
}

void GumbelEvaluator::Backward(
    absl::InlinedVector<TreeNode*, kMaxPathLenEst>& path) {
  float leaf_util = path[path.size() - 1]->q;

  for (int i = path.size() - 2; i >= 0; --i) {
    TreeNode* parent = path[i];
    TreeNode* child = path[i + 1];
    parent->n += 1;
    parent->w += -leaf_util;
    parent->q = parent->w / parent->n;
    if (child->n > parent->max_child_n) {
      parent->max_child_n = child->n;
    }

    leaf_util *= -1;
  }
}

}  // namespace mcts
