#ifndef __MCTS_TREE_H_
#define __MCTS_TREE_H_

#include <cstddef>
#include <memory>

#include "cc/constants/constants.h"
#include "cc/game/board.h"

namespace mcts {

enum class TreeNodeState {
  kNew = 0,          // new node. No evaluation has been performed.
  kNnEvaluated = 1,  // initial NN evaluation has been done.
  kExpanded = 2,     // at least one child has been visited.
};

struct TreeNode final {
  TreeNode() = default;
  ~TreeNode() = default;

  TreeNodeState state = TreeNodeState::kNew;
  bool is_terminal = false;
  int color_to_move;

  // change throughout search
  int n = 0;
  float w = 0;
  float q = 0;

  float w_outcome = 0;
  float q_outcome = 0;

  int max_child_n = 0;

  std::array<std::unique_ptr<TreeNode>, constants::kMaxNumMoves> children{};

  // write-once
  std::array<float, constants::kMaxNumMoves> move_logits{};
  std::array<float, constants::kMaxNumMoves> move_probs{};
  float outcome_est = 0;
  float score_est = 0;
  float init_util_est = 0;  // mix value estimate and score estimate.
};

inline float N(TreeNode* node) { return node == nullptr ? 0 : node->n; }
inline float NAction(TreeNode* node, int action) {
  return N(node->children[action].get());
}

inline float Q(TreeNode* node) {
  // return minimum value if node is null (init-to-loss).
  return node == nullptr ? -1.5 : node->q;
}

inline float QOutcome(TreeNode* node) {
  // return minimum value if node is null (init-to-loss).
  return node == nullptr ? -1.0 : node->q_outcome;
}

inline float QAction(TreeNode* node, int action) {
  // remember to flip sign. In bare MCTS, this will also cause MCTS to make deep
  // reads.
  return !node->children[action] ? -1.5 : -node->children[action]->q;
}

inline float QOutcomeAction(TreeNode* node, int action) {
  // remember to flip sign. In bare MCTS, this will also cause MCTS to make deep
  // reads.
  return !node->children[action] ? -1.0 : -node->children[action]->q_outcome;
}

inline float MaxN(TreeNode* node) {
  if (node == nullptr) return 0;

  return node->max_child_n;
}

// Each node is visited once when expanded, and once per search path. Thus, the
// total visit count of its children should be N(node) - 1
inline float SumChildrenN(TreeNode* node) {
  return node == nullptr ? 0 : node->n - 1;
}

void AdvanceState(TreeNode* node);

}  // namespace mcts

#endif  // __MCTS_TREE_H_
