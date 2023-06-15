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

float NAction(TreeNode* node, int action) {
  return N(node->children[action].get());
}

float Q(TreeNode* node) {
  // return minimum value if node is null (init-to-loss).
  return node == nullptr ? -1.5 : node->q;
}

float QAction(TreeNode* node, int action) {
  // remember to flip sign. In bare MCTS, this will also cause MCTS to make deep
  // reads.
  return !node->children[action] ? -1.5 : -node->children[action]->q;
}

float MaxN(TreeNode* node) {
  if (node == nullptr) return 0;

  return node->max_child_n;
}

// Each node is visited once when expanded, and once per search path. Thus, the
// total visit count of its children should be N(node) - 1
float SumChildrenN(TreeNode* node) { return node == nullptr ? 0 : node->n - 1; }

}  // namespace mcts
