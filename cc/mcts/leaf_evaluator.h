#ifndef MCTS_LEAF_EVALUATOR_H_
#define MCTS_LEAF_EVALUATOR_H_

#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace mcts {

enum class ScoreUtilityMode : uint8_t { kDirect = 0, kIntegral = 1 };

struct ScoreUtilityParams {
  float score_weight = kDefaultScoreWeight;
  ScoreUtilityMode mode = ScoreUtilityMode::kDirect;
};

/*
 * Wrapper Class for Leaf Evaluation. Mainly useful for testing.
 */
class LeafEvaluator final {
 public:
  // Convenience constructors use default ScoreUtilityParams.
  LeafEvaluator(nn::NNInterface* nn_interface, int thread_id);
  LeafEvaluator(nn::NNInterface::Slot slot, int thread_id);
  LeafEvaluator(nn::NNInterface::Slot slot, int thread_id,
                ScoreUtilityParams score_params);
  ~LeafEvaluator() = default;

  // Disable Copy and Move.
  LeafEvaluator(LeafEvaluator const&) = delete;
  LeafEvaluator& operator=(LeafEvaluator const&) = delete;
  LeafEvaluator(LeafEvaluator&&) = delete;
  LeafEvaluator& operator=(LeafEvaluator&&) = delete;

  // Calls `InitTreeNode` and fills initial stats for unevaluated root nodes.
  void EvaluateRoot(core::Probability& probability, const game::Game& game,
                    TreeNode* node, game::Color color_to_move);

  // Calls `InitTreeNode` and fills initial stats.
  void EvaluateLeaf(core::Probability& probability, const game::Game& game,
                    TreeNode* node, game::Color color_to_move,
                    game::Color root_color, float root_score_est);

  // Populates a terminal node.
  void EvaluateTerminal(const game::Scores& scores, TreeNode* terminal_node,
                        game::Color color_to_move, game::Color root_color,
                        float root_score_est);

  // Evaluates a leaf node using the neural net.
  void InitTreeNode(core::Probability& probability, TreeNode* node,
                    const game::Game& game, game::Color color_to_move);

  int thread_id() const { return thread_id_; }

  // Queues an example for evaluation.
  inline void QueueEval(core::Probability& probability, const game::Game& game,
                        game::Color color_to_move) {
    slot_.LoadEntry(thread_id_, game, color_to_move, probability);
  }

  // Fetches queued example.
  void FetchLeafEval(TreeNode* node, const game::Game& game,
                     game::Color color_to_move, game::Color root_color,
                     float root_score_est);
  void FetchRootEval(TreeNode* node, const game::Game& game,
                     game::Color color_to_move);

 private:
  float ScoreUtility(float score_est, float score_stddev,
                     float root_score_est) const;

  nn::NNInterface::Slot slot_;
  int thread_id_;
  const ScoreUtilityParams score_params_;
};
}  // namespace mcts

#endif
