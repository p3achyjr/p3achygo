#include "cc/mcts/leaf_evaluator.h"

#include "absl/log/check.h"
#include "cc/mcts/constants.h"

namespace mcts {
namespace {
using namespace ::game;
}  // namespace

LeafEvaluator::LeafEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : LeafEvaluator(nn_interface, thread_id, kDefaultScoreWeight) {}

LeafEvaluator::LeafEvaluator(nn::NNInterface* nn_interface, int thread_id,
                             float score_weight)
    : nn_interface_(nn_interface),
      thread_id_(thread_id),
      score_weight_(score_weight) {}

void LeafEvaluator::EvaluateRoot(core::Probability& probability,
                                 const Game& game, TreeNode* node,
                                 Color color_to_move) {
  // Call for any not-yet-evaluated root nodes.
  InitTreeNode(probability, node, game, color_to_move);
  InitFields(node, 0);
}

void LeafEvaluator::EvaluateLeaf(core::Probability& probability,
                                 const Game& game, TreeNode* node,
                                 Color color_to_move, Color root_color,
                                 float root_score_est) {
  InitTreeNode(probability, node, game, color_to_move);

  root_score_est *= color_to_move == root_color ? 1.0f : -1.0f;
  float score_utility =
      ScoreTransform(score_weight_, node->score_est, root_score_est);

  InitFields(node, score_utility);
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

  terminal_node->is_terminal = true;
  terminal_node->v = empirical_q;
  terminal_node->v_outcome = empirical_outcome;
}

void LeafEvaluator::InitTreeNode(core::Probability& probability, TreeNode* node,
                                 const Game& game, Color color_to_move) {
  DCHECK(node->state == TreeNodeState::kNew);
  nn::NNInferResult infer_result = nn_interface_->LoadAndGetInference(
      thread_id_, game, color_to_move, probability);

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
  node->outcome_est = value_est;
  node->score_est = score_est;

  AdvanceState(node);
}

inline void LeafEvaluator::InitFields(TreeNode* node, float score_utility) {
  node->n = 1;
  node->w = node->outcome_est + score_utility;
  node->w_outcome = node->outcome_est;
  node->v = node->w;
  node->v_outcome = node->w_outcome;
  node->init_util_est = node->w;

#ifdef V_CATEGORICAL
  // Add V to bucket.
  int v_bucket = std::clamp(static_cast<int>((node->v + 1.0f) / kBucketRange),
                            0, kNumVBuckets - 1);
  node->v_categorical[v_bucket] += 1;
#endif
}
}  // namespace mcts
