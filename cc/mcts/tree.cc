#include "cc/mcts/tree.h"

#include <iostream>

namespace mcts {

void AdvanceState(TreeNode* node) {
  if (node == nullptr) return;
  switch (node->state) {
    case TreeNodeState::kNew:
      node->state = TreeNodeState::kNnEvaluated;
      break;
    case TreeNodeState::kNnEvaluated:
      node->state = TreeNodeState::kExpanded;
      break;
    case TreeNodeState::kExpanded:
      break;
  }
}

float N(TreeNode* node) { return node == nullptr ? 0 : node->n; }

float Q(TreeNode* node) {
  // return minimum value if node is null
  return node == nullptr ? -1.5 : node->q;
}

float MaxN(TreeNode* node) {
  if (node == nullptr) return 0;

  return node->max_child_n;
}

// Each node is visited once when expanded, and once per search path. Thus, the
// total visit count of its children should be N(node) - 1
float SumChildrenN(TreeNode* node) { return node == nullptr ? 0 : node->n - 1; }

}  // namespace mcts
