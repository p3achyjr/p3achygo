#include "cc/mcts/leaf_evaluator.h"

#include "absl/log/check.h"
#include "cc/constants/constants.h"
#include "cc/mcts/constants.h"
#include "cc/nn/engine/engine.h"

namespace mcts {
namespace {
using namespace ::game;

inline float ScoreTransform(float c_score, float score_est,
                            float root_score_est) {
  return c_score * M_2_PI * std::atan((score_est - root_score_est) / BOARD_LEN);
}

inline void InitFields(const nn::NNInferResult& infer_result, TreeNode* node,
                       Color color_to_move) {
  std::copy(infer_result.move_logits.begin(), infer_result.move_logits.end(),
            node->move_logits.begin());
  std::copy(infer_result.move_probs.begin(), infer_result.move_probs.end(),
            node->move_probs.begin());

  float value_est =
      infer_result.value_probs[0] * -1 + infer_result.value_probs[1] * 1;

  float score_est = 0.0;
  for (auto i = 0; i < constants::kNumScoreLogits; ++i) {
    float score_normalized = i - constants::kScoreInflectionPoint + .5;
    score_est += (infer_result.score_probs[i] * score_normalized);
  }

  node->color_to_move = color_to_move;
  node->init_outcome_est = value_est;
  node->init_score_est = score_est;

  AdvanceState(node);
}
}  // namespace

LeafEvaluator::LeafEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : LeafEvaluator(nn_interface->MakeSlot(0), thread_id) {}

LeafEvaluator::LeafEvaluator(nn::NNInterface::Slot slot, int thread_id)
    : slot_(slot), thread_id_(thread_id), score_weight_(kDefaultScoreWeight) {}

void LeafEvaluator::EvaluateRoot(core::Probability& probability,
                                 const Game& game, TreeNode* node,
                                 Color color_to_move) {
  // Call for any not-yet-evaluated root nodes.
  InitTreeNode(probability, node, game, color_to_move);
  node->init_util_est = node->init_outcome_est;

  // Root node does not backprop.
  node->n = 1;
  node->w = node->init_util_est;
  node->w_outcome = node->init_outcome_est;
  node->v = node->init_util_est;
  node->v_outcome = node->init_outcome_est;
}

void LeafEvaluator::EvaluateLeaf(core::Probability& probability,
                                 const Game& game, TreeNode* node,
                                 Color color_to_move, Color root_color,
                                 float root_score_est) {
  InitTreeNode(probability, node, game, color_to_move);

  root_score_est *= color_to_move == root_color ? 1.0f : -1.0f;
  float score_utility =
      ScoreTransform(score_weight_, node->init_score_est, root_score_est);
  node->init_util_est = node->init_outcome_est + score_utility;
}

void LeafEvaluator::EvaluateTerminal(const Scores& scores,
                                     TreeNode* terminal_node,
                                     Color color_to_move, Color root_color,
                                     float root_score_est) {
  float player_score =
      color_to_move == BLACK ? scores.black_score : scores.white_score;
  float opp_score =
      color_to_move == BLACK ? scores.white_score : scores.black_score;
  float final_score = player_score - opp_score;

  // Using the actual score, instead of {-1.5, 1.5}, incentivizes search to
  // pass if all other moves just lead to worse outcomes.
  float empirical_q =
      (player_score > opp_score ? 1.0 : -1.0) +
      ScoreTransform(score_weight_, final_score, root_score_est);
  float empirical_outcome = player_score > opp_score ? 1.0 : -1.0;

  terminal_node->color_to_move = color_to_move;
  terminal_node->is_terminal = true;
  terminal_node->init_util_est = empirical_q;
  terminal_node->init_outcome_est = empirical_outcome;
  terminal_node->init_score_est = final_score;
}

void LeafEvaluator::InitTreeNode(core::Probability& probability, TreeNode* node,
                                 const Game& game, Color color_to_move) {
  nn::NNInferResult infer_result =
      slot_.LoadAndGetInference(thread_id_, game, color_to_move, probability);
  InitFields(infer_result, node, color_to_move);
}

void LeafEvaluator::FetchLeafEval(TreeNode* node, const Game& game,
                                  Color color_to_move, Color root_color,
                                  float root_score_est) {
  nn::NNInferResult infer_result =
      slot_.FetchEntry(thread_id_, game, color_to_move);
  InitFields(infer_result, node, color_to_move);
  root_score_est *= color_to_move == root_color ? 1.0f : -1.0f;
  float score_utility =
      ScoreTransform(score_weight_, node->init_score_est, root_score_est);
  node->init_util_est = node->init_outcome_est + score_utility;
}

void LeafEvaluator::FetchRootEval(TreeNode* node, const game::Game& game,
                                  game::Color color_to_move) {
  nn::NNInferResult infer_result =
      slot_.FetchEntry(thread_id_, game, color_to_move);
  InitFields(infer_result, node, color_to_move);
  node->init_util_est = node->init_outcome_est;
  node->w = node->init_util_est;
  node->v = node->init_util_est;
  node->w_outcome = node->init_outcome_est;
  node->v_outcome = node->init_outcome_est;
  node->n = 1;
}
}  // namespace mcts
