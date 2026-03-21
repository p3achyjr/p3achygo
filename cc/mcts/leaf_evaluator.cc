#include "cc/mcts/leaf_evaluator.h"

#include "absl/log/check.h"
#include "cc/constants/constants.h"
#include "cc/mcts/constants.h"
#include "cc/nn/engine/engine.h"

namespace mcts {
namespace {
using namespace ::game;

static constexpr int kNumScoreMeans = constants::kNumScoreLogits;
static constexpr int kNumScoreStddevs = constants::kNumScoreLogits / 2;
static constexpr int kScoreTableSize = kNumScoreMeans * kNumScoreStddevs;

// Interpret table as 2D array of dimensions [kNumScoreMeans, kNumScoreStddevs]
inline int table_index(const int score_idx, const int stddev) {
  return score_idx * kNumScoreStddevs + stddev;
}
static const std::array<float, kScoreTableSize> kScoreTransformTable = []() {
  constexpr float kZStep = 0.1f;
  constexpr float kZBound = 5.0f;
  std::array<float, kScoreTableSize> score_table;
  for (int score_idx = 0; score_idx < kNumScoreMeans; ++score_idx) {
    const float score_mean =
        score_idx - constants::kScoreInflectionPoint + 0.5f;
    for (int stddev = 0; stddev < kNumScoreStddevs; ++stddev) {
      // compute weighted sum of score_transform values.
      float total_pdf_mass = 0.0f;
      float score_transform_integral_unnormalized = 0.0f;
      for (float z = -kZBound; z <= kZBound; z += kZStep) {
        const float pdf_scaled = std::exp(-0.5 * z * z);
        const float score_transform =
            M_2_PI * std::atan((score_mean + z * stddev) / BOARD_LEN);
        total_pdf_mass += pdf_scaled;
        score_transform_integral_unnormalized += score_transform * pdf_scaled;
      }

      // then normalize away the weight.
      const float score_transform_integral =
          score_transform_integral_unnormalized / total_pdf_mass;
      score_table[table_index(score_idx, stddev)] = score_transform_integral;
    }
  }
  return score_table;
}();

inline float ScoreTransformIntegral(float c_score, float score_est,
                                    float score_stddev, float root_score_est) {
  // interpolate between absolute score and advantage.
  const float root_score_normalized = 0.75f * root_score_est;
  const float score_mean = score_est - root_score_normalized;
  const int score_floored = std::floor(score_mean - 0.5f);
  const int stddev_floored = std::floor(score_stddev);

  // interpolate between 4 integral values between nearest scores and nearest
  // stddevs.
  const int score_idx = std::clamp(
      score_floored + constants::kScoreInflectionPoint, 0, kNumScoreMeans - 2);
  const int stddev_idx = std::clamp(stddev_floored, 0, kNumScoreStddevs - 2);
  const int x0 = score_idx, x1 = score_idx + 1;
  const int y0 = stddev_idx, y1 = stddev_idx + 1;
  const float mean_delta = (score_mean - 0.5f) - float(score_floored);
  const float stddev_delta = score_stddev - float(stddev_floored);

  // grid coordinates (aij -> i = score_idx, j = stddev_idx)
  const float a00 = kScoreTransformTable[table_index(x0, y0)];
  const float a01 = kScoreTransformTable[table_index(x0, y1)];
  const float a10 = kScoreTransformTable[table_index(x1, y0)];
  const float a11 = kScoreTransformTable[table_index(x1, y1)];

  const float b0 = a00 + stddev_delta * (a01 - a00);
  const float b1 = a10 + stddev_delta * (a11 - a10);
  const float integral = b0 + mean_delta * (b1 - b0);
  return c_score * integral;
}

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
  std::copy(infer_result.opt_move_probs.begin(),
            infer_result.opt_move_probs.end(), node->opt_probs.begin());

  float value_est =
      infer_result.value_probs[0] * -1 + infer_result.value_probs[1] * 1;

  // Compute E[score] and Var[score] = E[score^2] - E[score]^2 in one pass.
  float score_est = 0.0;
  float score_sq_est = 0.0;
  for (auto i = 0; i < constants::kNumScoreLogits; ++i) {
    float score_normalized = i - constants::kScoreInflectionPoint + .5f;
    float p = infer_result.score_probs[i];
    score_est += p * score_normalized;
    score_sq_est += p * score_normalized * score_normalized;
  }

  node->color_to_move = color_to_move;
  node->init_outcome_est = value_est;
  node->init_score_est = score_est;
  node->init_score_var = score_sq_est - score_est * score_est;

  AdvanceState(node);
}
}  // namespace

LeafEvaluator::LeafEvaluator(nn::NNInterface* nn_interface, int thread_id)
    : LeafEvaluator(nn_interface->MakeSlot(0), thread_id) {}

LeafEvaluator::LeafEvaluator(nn::NNInterface::Slot slot, int thread_id)
    : LeafEvaluator(slot, thread_id, ScoreUtilityParams{}) {}

LeafEvaluator::LeafEvaluator(nn::NNInterface::Slot slot, int thread_id,
                             ScoreUtilityParams score_params)
    : slot_(slot), thread_id_(thread_id), score_params_(score_params) {}

float LeafEvaluator::ScoreUtility(float score_est, float score_stddev,
                                  float root_score_est) const {
  if (score_params_.mode == ScoreUtilityMode::kIntegral) {
    return ScoreTransformIntegral(score_params_.score_weight, score_est,
                                  score_stddev, root_score_est);
  }
  return ScoreTransform(score_params_.score_weight, score_est, root_score_est);
}

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
  float score_utility = ScoreUtility(
      node->init_score_est, std::sqrt(node->init_score_var), root_score_est);
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
  float score_utility = ScoreUtility(final_score, 0.0f, root_score_est);
  float empirical_q = (player_score > opp_score ? 1.0 : -1.0) + score_utility;
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
  float score_utility = ScoreUtility(
      node->init_score_est, std::sqrt(node->init_score_var), root_score_est);
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
