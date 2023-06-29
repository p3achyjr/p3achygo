#include "cc/mcts/leaf_evaluator.h"

#include "absl/log/check.h"

namespace mcts {
namespace {
using ::game::Color;
using ::game::Game;
using ::game::Loc;

static constexpr float kScoreScale = .5;

float ScoreTransform(float score_est, float root_score_est) {
  return kScoreScale * M_2_PI *
         std::atan((score_est - root_score_est) / BOARD_LEN);
}
}  // namespace

LeafEvaluator::LeafEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : nn_interface_(nn_interface), thread_id_(thread_id) {}

void LeafEvaluator::EvaluateRoot(const Game& game, TreeNode* node,
                                 Color color_to_move) {
  // Call for any not-yet-evaluated root nodes.
  InitTreeNode(node, game, color_to_move);
  node->n = 1;
  node->w = node->value_est;
  node->q = node->w;
  node->init_util_est = node->w;
}

void LeafEvaluator::EvaluateLeaf(const Game& game, TreeNode* node,
                                 Color color_to_move, float root_score_est) {
  InitTreeNode(node, game, color_to_move);
  float score_utility = ScoreTransform(node->score_est, root_score_est);

  node->n = 1;
  node->w = node->value_est + score_utility;
  node->q = node->w;
  node->init_util_est = node->w;
}

void LeafEvaluator::InitTreeNode(TreeNode* node, const Game& game,
                                 Color color_to_move) {
  DCHECK(node->state == TreeNodeState::kNew);
  nn::NNInferResult infer_result =
      nn_interface_->LoadAndGetInference(thread_id_, game, color_to_move);

  std::copy(infer_result.move_logits.begin(), infer_result.move_logits.end(),
            node->move_logits.begin());
  std::copy(infer_result.move_probs.begin(), infer_result.move_probs.end(),
            node->move_probs.begin());

  float value_est =
      infer_result.value_probs[0] * -1 + infer_result.value_probs[1] * 1;

  float score_est = 0.0;
  for (auto i = 0; i < constants::kNumScoreLogits; ++i) {
    score_est += (infer_result.score_probs[i] * i);
  }

  node->color_to_move = color_to_move;
  node->value_est = value_est;
  node->score_est = score_est;

  AdvanceState(node);
}
}  // namespace mcts
