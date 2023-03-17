#include "cc/mcts/gumbel.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/board.h"
#include "cc/mcts/tree.h"

using ::game::Board;
using ::game::Loc;

namespace mcts {
namespace {

static constexpr int kVisit = 50;
static constexpr float kValueScale = 1.0;
static constexpr float kScoreScale = .5;

struct GumbelMoveInfo {
  float logit = 0;
  float gumbel_noise = 0;
  float qtransformed = 0;
  int move_encoding = -1;
  Loc move_loc = {-1, -1};
};

bool GumbelMoveInfoLess(const GumbelMoveInfo& x, const GumbelMoveInfo& y) {
  return x.logit + x.gumbel_noise + x.qtransformed <
         y.logit + y.gumbel_noise + y.qtransformed;
}

bool GumbelMoveInfoGreater(const GumbelMoveInfo& x, const GumbelMoveInfo& y) {
  return x.logit + x.gumbel_noise + x.qtransformed >
         y.logit + y.gumbel_noise + y.qtransformed;
}

int log2(int x) {
  // x needs to be > 0
  int i = 1;
  while (x >> i > 0) {
    i++;
  }

  return i - 1;
}

float ScoreTransform(float score_estimate, float root_score_estimate,
                     int board_length) {
  return 2 / M_PI *
         std::atan((score_estimate - root_score_estimate) / board_length);
}

// `q`: value estimate of action
// `max_b`: max visit count of all children
float QTransform(float q, int max_b) {
  return (kVisit + max_b) * kValueScale * q;
}

// interpolates NN value evaluation with empirical values derived from visits.
float VMixed(TreeNode* node) {
  double weighted_visited_qvalues = 0;
  double total_visited_probability = 0;
  for (auto i = 0; i < constants::kMaxNumMoves; ++i) {
    if (N(node->children[i].get()) > 0) {
      weighted_visited_qvalues +=
          (node->move_probabilities[i] * Q(node->children[i].get()));
      total_visited_probability += node->move_probabilities[i];
    }
  }

  double interpolated_qvalue =
      (weighted_visited_qvalues * N(node) / total_visited_probability +
       node->init_utility_estimate);

  return interpolated_qvalue / (1 + N(node));
}

void Softmax(float* logits, float* result, int n) {
  double max = *std::max_element(logits, logits + n);

  double normalized_logits[n];
  std::transform(logits, logits + n, normalized_logits,
                 [&](float x) { return x - max; });
  double exponentiated[n];
  std::transform(normalized_logits, normalized_logits + n, exponentiated,
                 [&](double x) { return std::exp(x); });
  double total = std::accumulate(exponentiated, exponentiated + n, 0.0);
  std::transform(exponentiated, exponentiated + n, result,
                 [&](double x) { return x / total; });
}

}  // namespace

// `n`: total number of simulations.
// `k`: initial number of actions selected.
// `n` must be >= `klogk`.
// !! `board`, `node`, and `moves` must be kept in sync with each other.
Loc GumbelEvaluator::SearchRoot(Board& board, TreeNode* node,
                                std::vector<Loc>& moves, int color_to_move,
                                int n, int k) {
  CHECK(node);
  core::Probability probability;
  int num_rounds = log2(k);
  int visits_per_action = n / k * num_rounds;

  if (node->state == TreeNodeState::kNew) {
    InitTreeNode(node, board, moves, color_to_move);
  }

  auto num_moves = constants::kMaxNumMoves;
  GumbelMoveInfo gmove_info[num_moves];
  for (auto i = 0; i < num_moves; ++i) {
    gmove_info[i].logit = node->move_logits[i];
    gmove_info[i].gumbel_noise = probability.GumbelSample();
    gmove_info[i].move_encoding = i;
    gmove_info[i].move_loc = board.MoveAsLoc(i);
  }

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
      if (!board.IsValidMove(move_info.move_loc, color_to_move)) {
        // ignore move henceforth
        move_info.logit = -10000;
        continue;
      }

      if (!node->children[move_info.move_encoding]) {
        node->children[move_info.move_encoding] = std::make_unique<TreeNode>();
      }

      TreeNode* child = node->children[move_info.move_encoding].get();
      game::Board search_board = board;
      std::vector<Loc> search_moves = moves;
      search_board.Move(move_info.move_loc, color_to_move);
      search_moves.emplace_back(move_info.move_loc);

      for (auto _ = 0; _ < visits_per_action; ++_) {
        SearchNonRoot(search_board, child, search_moves,
                      game::OppositeColor(color_to_move), node->score_estimate);
      }

      // update qvalue
      auto child_q = Q(child);
      move_info.qtransformed = QTransform(child_q, MaxN(node));

      // update tree
      UpdateParentFromChild(node, child);
    }

    std::sort(gmove_info, gmove_info + k, GumbelMoveInfoGreater);
    k /= 2;
    visits_per_action *= 2;
  }

  AdvanceState(node);
  return gmove_info[0].move_loc;
}

// `board`: Local search board
// `node`: Tree node corresponding to local search board
// `moves`: Local moves vector
// `color_to_move`: Color whose turn it is to move next
// `root_score_estimate`: Value estimate for root node. Subsequent node score
// estimates will be centered against this value.
void GumbelEvaluator::SearchNonRoot(game::Board& board, TreeNode* node,
                                    std::vector<game::Loc>& moves,
                                    int color_to_move,
                                    float root_score_estimate) {
  if (node->state == TreeNodeState::kNew) {
    // leaf node. evaluate and return.
    EvaluateLeaf(board, node, moves, color_to_move, root_score_estimate);
    return;
  }

  // internal node. Trace a single path until we hit a leaf, using a
  // deterministic paradigm.
  std::vector<TreeNode*> path = {node};
  while (path.back()->state != TreeNodeState::kNew &&
         !(path.back()->is_terminal) && !board.IsGameOver()) {
    auto node = path.back();
    float v_mix = VMixed(node);
    float logits_improved[constants::kMaxNumMoves];
    int max_n = MaxN(node);
    for (auto i = 0; i < constants::kMaxNumMoves; ++i) {
      TreeNode* child = node->children[i].get();
      logits_improved[i] = node->move_logits[i] +
                           QTransform(N(child) > 0 ? Q(child) : v_mix, max_n);
    }

    float policy_improved[constants::kMaxNumMoves];
    Softmax(logits_improved, policy_improved, constants::kMaxNumMoves);

    // select node with greatest disparity between expected value and visit
    // count.
    int selected_action = 0;
    float max_disparity = -10000;
    for (auto i = 0; i < constants::kMaxNumMoves; ++i) {
      if (!board.IsValidMove(board.MoveAsLoc(i), color_to_move)) {
        continue;
      }

      TreeNode* child = node->children[i].get();
      float disparity = policy_improved[i] - (N(child) / (1 + node->n));
      if (disparity > max_disparity) {
        max_disparity = disparity;
        selected_action = i;
      }
    }

    if (!node->children[selected_action]) {
      node->children[selected_action] = std::make_unique<TreeNode>();
    }

    Loc move_loc = board.MoveAsLoc(selected_action);
    board.Move(move_loc, color_to_move);
    moves.emplace_back(move_loc);
    path.emplace_back(node->children[selected_action].get());
    color_to_move = game::OppositeColor(color_to_move);
    AdvanceState(node);
  }

  // either we have reached a leaf node, or we have reached the end of the game,
  // or both.
  TreeNode* leaf_node = path.back();
  if (leaf_node->state == TreeNodeState::kNew) {
    EvaluateLeaf(board, node, moves, color_to_move, root_score_estimate);
  }

  if (board.IsGameOver() && !leaf_node->is_terminal) {
    // evaluate score
    float player_score = board.Score(color_to_move);
    float opp_score = board.Score(game::OppositeColor(color_to_move));
    float final_score =
        player_score - opp_score + constants::kScoreInflectionPoint;
    float empirical_q =
        (player_score > opp_score ? 1.0 : -1.0) +
        ScoreTransform(final_score, root_score_estimate, board.length());
    leaf_node->is_terminal = true;
    leaf_node->q = empirical_q;
  }

  // traverse path backwards, updating all N/Q/W values along the way.
  for (auto i = path.size() - 2; i >= 0; --i) {
    UpdateParentFromChild(path[i], path[i + 1]);
  }
}

void GumbelEvaluator::EvaluateLeaf(game::Board& board, TreeNode* node,
                                   std::vector<game::Loc>& moves,
                                   int color_to_move,
                                   float root_score_estimate) {
  InitTreeNode(node, board, moves, color_to_move);
  float score_utility =
      kScoreScale *
      ScoreTransform(node->score_estimate, root_score_estimate, board.length());
  node->n = 1;
  node->w = node->value_estimate + score_utility;
  node->q = node->w;
  node->init_utility_estimate = node->w;
}

void GumbelEvaluator::InitTreeNode(TreeNode* node, const game::Board& board,
                                   const std::vector<game::Loc>& moves,
                                   int color_to_move) {
  DCHECK(node->state == TreeNodeState::kNew);
  absl::StatusOr<nn::NNInferResult> infer_result =
      nn_interface_.GetInferenceResult(board, moves, color_to_move);

  CHECK_OK(infer_result);

  std::copy(infer_result->move_logits,
            infer_result->move_logits + constants::kMaxNumMoves,
            node->move_logits);
  std::copy(infer_result->move_probabilities,
            infer_result->move_probabilities + constants::kMaxNumMoves,
            node->move_probabilities);

  float value_estimate = infer_result->value_probability[0] * -1 +
                         infer_result->value_probability[1] * 1;

  float score_estimate = 0.0;
  for (auto i = 0; i < constants::kNumScoreLogits; ++i) {
    score_estimate += (infer_result->score_probabilities[i] * i);
  }

  node->color_to_move = color_to_move;
  node->score_estimate = score_estimate;
  AdvanceState(node);
}

}  // namespace mcts
