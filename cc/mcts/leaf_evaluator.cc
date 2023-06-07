#include "cc/mcts/leaf_evaluator.h"

#include "absl/log/check.h"

namespace mcts {
namespace {
using ::game::Color;
using ::game::Game;
using ::game::Loc;

static constexpr float kScoreScale = .5;

float ScoreTransform(float score_est, float root_score_est, int board_length) {
  return 2 / M_PI * std::atan((score_est - root_score_est) / board_length);
}
}  // namespace

LeafEvaluator::LeafEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : nn_interface_(nn_interface), thread_id_(thread_id) {}

void LeafEvaluator::EvaluateLeaf(Game& game, TreeNode* node,
                                 Color color_to_move, float root_score_est) {
  InitTreeNode(node, game, color_to_move);
  float score_utility =
      kScoreScale *
      ScoreTransform(node->score_est, root_score_est, game.board_len());

  node->n = 1;
  node->w = node->value_est + score_utility;
  node->q = node->w;
  node->init_util_est = node->w;
}

void LeafEvaluator::InitTreeNode(TreeNode* node, const Game& game,
                                 Color color_to_move) {
  DCHECK(node->state == TreeNodeState::kNew);
  CHECK_OK(nn_interface_->LoadBatch(thread_id_, game, color_to_move));
  nn::NNInferResult infer_result =
      nn_interface_->GetInferenceResult(thread_id_);

  std::copy(infer_result.move_logits,
            infer_result.move_logits + constants::kMaxNumMoves,
            node->move_logits);
  std::copy(infer_result.move_probs,
            infer_result.move_probs + constants::kMaxNumMoves,
            node->move_probs);

  float value_est =
      infer_result.value_probs[0] * -1 + infer_result.value_probs[1] * 1;

  float score_est = 0.0;
  for (auto i = 0; i < constants::kNumScoreLogits; ++i) {
    score_est += (infer_result.score_probs[i] * i);
  }

  node->color_to_move = color_to_move;
  node->value_est = value_est;
  node->score_est = score_est;

  // arbitrary scale to discourage passing, since the current probability is
  // high.
  node->move_logits[constants::kPassMoveEncoding] -= 4;

  AdvanceState(node);
}
}  // namespace mcts
