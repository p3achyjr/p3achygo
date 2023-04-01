#include "cc/mcts/gumbel.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "cc/constants/constants.h"
#include "cc/core/util.h"
#include "cc/game/board.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

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
  float qtransform = 0;
  int move_encoding = -1;
  Loc move_loc = {-1, -1};
};

bool GumbelMoveInfoLess(const GumbelMoveInfo& x, const GumbelMoveInfo& y) {
  return x.logit + x.gumbel_noise + x.qtransform <
         y.logit + y.gumbel_noise + y.qtransform;
}

bool GumbelMoveInfoGreater(const GumbelMoveInfo& x, const GumbelMoveInfo& y) {
  return x.logit + x.gumbel_noise + x.qtransform >
         y.logit + y.gumbel_noise + y.qtransform;
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
  if (SumChildrenN(node) == 0) {
    return node->init_utility_estimate;
  }

  double weighted_visited_qvalues = 0;
  double total_visited_probability = 0;
  for (auto i = 0; i < constants::kMaxNumMoves; ++i) {
    if (N(node->children[i].get()) > 0) {
      weighted_visited_qvalues +=
          (node->move_probabilities[i] * Q(node->children[i].get()));
      total_visited_probability += node->move_probabilities[i];
    }
  }

  double interpolated_qvalue = (weighted_visited_qvalues * SumChildrenN(node) /
                                    total_visited_probability +
                                node->init_utility_estimate);

  return interpolated_qvalue / (1 + SumChildrenN(node));
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

int Argmax(float logits[constants::kMaxNumMoves]) {
  int arg_max = 0;
  float max_logit = -1000000;
  for (int i = 0; i < constants::kMaxNumMoves; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
      arg_max = i;
    }
  }

  return arg_max;
}

}  // namespace

GumbelEvaluator::GumbelEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : nn_interface_(nn_interface), thread_id_(thread_id) {}

// `n`: total number of simulations.
// `k`: initial number of actions selected.
// `n` must be >= `klogk`.
// !! `board`, `node`, and `moves` must be kept in sync with each other.
std::pair<Loc, Loc> GumbelEvaluator::SearchRoot(core::Probability& probability,
                                                Board& board, TreeNode* node,
                                                std::vector<Loc>& moves,
                                                int color_to_move, int n,
                                                int k) {
  CHECK(node);
  int num_rounds = log2(k);
  int visits_per_action = n / (k * num_rounds);

  if (node->state == TreeNodeState::kNew) {
    InitTreeNode(node, board, moves, color_to_move);
  }

  auto num_moves = constants::kMaxNumMoves;
  GumbelMoveInfo gmove_info[num_moves];
  for (auto i = 0; i < num_moves; ++i) {
    if (!board.IsValidMove(board.MoveAsLoc(i), color_to_move)) {
      // ignore move henceforth
      gmove_info[i].logit = -10000;
      continue;
    }

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
      if (!node->children[move_info.move_encoding]) {
        node->children[move_info.move_encoding] = std::make_unique<TreeNode>();
      }

      TreeNode* child = node->children[move_info.move_encoding].get();
      for (auto _ = 0; _ < visits_per_action; ++_) {
        game::Board search_board = board;
        std::vector<Loc> search_moves = moves;
        search_board.Move(move_info.move_loc, color_to_move);
        search_moves.emplace_back(move_info.move_loc);

        std::vector<TreeNode*> search_path = SearchNonRoot(
            search_board, child, search_moves,
            game::OppositeColor(color_to_move), node->score_estimate);

        // update tree
        Backward(search_path);
      }

      // update qvalue
      auto child_q = -Q(child);
      move_info.qtransform = QTransform(child_q, MaxN(node));
    }

    std::sort(gmove_info, gmove_info + k, GumbelMoveInfoGreater);
    k /= 2;
    visits_per_action *= 2;
  }

  AdvanceState(node);

  game::Loc raw_nn_move = board.MoveAsLoc(Argmax(node->move_logits));
  return {raw_nn_move, gmove_info[0].move_loc};
}

// `board`: Local search board
// `node`: Tree node corresponding to local search board
// `moves`: Local moves vector
// `color_to_move`: Color whose turn it is to move next
// `root_score_estimate`: Value estimate for root node. Subsequent node score
// estimates will be centered against this value.
std::vector<TreeNode*> GumbelEvaluator::SearchNonRoot(
    game::Board& board, TreeNode* node, std::vector<game::Loc>& moves,
    int color_to_move, float root_score_estimate) {
  std::vector<TreeNode*> path = {node};
  if (node->state == TreeNodeState::kNew) {
    // leaf node. evaluate and return.
    EvaluateLeaf(board, node, moves, color_to_move, root_score_estimate);
    return path;
  }

  // internal node. Trace a single path until we hit a leaf, using a
  // deterministic paradigm.
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
      float disparity =
          policy_improved[i] - (N(child) / (1 + SumChildrenN(node)));
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
    EvaluateLeaf(board, leaf_node, moves, color_to_move, root_score_estimate);
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

  return path;
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
  CHECK_OK(nn_interface_->LoadBatch(thread_id_, board, moves, color_to_move));
  nn::NNInferResult infer_result =
      nn_interface_->GetInferenceResult(thread_id_);

  std::copy(infer_result.move_logits,
            infer_result.move_logits + constants::kMaxNumMoves,
            node->move_logits);
  std::copy(infer_result.move_probabilities,
            infer_result.move_probabilities + constants::kMaxNumMoves,
            node->move_probabilities);

  float value_estimate = infer_result.value_probability[0] * -1 +
                         infer_result.value_probability[1] * 1;

  float score_estimate = 0.0;
  for (auto i = 0; i < constants::kNumScoreLogits; ++i) {
    score_estimate += (infer_result.score_probabilities[i] * i);
  }

  node->color_to_move = color_to_move;
  node->value_estimate = value_estimate;
  node->score_estimate = score_estimate;

  // arbitrary scale to discourage passing, since the current probability is
  // high.
  node->move_logits[constants::kPassMoveEncoding] -= 4;

  AdvanceState(node);
}

void GumbelEvaluator::Backward(std::vector<TreeNode*>& path) {
  float leaf_utility = path[path.size() - 1]->q;

  for (int i = path.size() - 2; i >= 0; --i) {
    TreeNode* parent = path[i];
    TreeNode* child = path[i + 1];
    parent->n += 1;
    parent->w += -leaf_utility;
    parent->q = parent->w / parent->n;
    if (child->n > parent->max_child_n) {
      parent->max_child_n = child->n;
    }

    leaf_utility *= -1;
  }
}

}  // namespace mcts
