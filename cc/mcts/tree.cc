#include "cc/mcts/tree.h"

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

int MaxN(TreeNode* node) {
  if (node == nullptr) return 0;

  return node->max_child_n;
}

void UpdateParentFromChild(TreeNode* parent, TreeNode* child) {
  parent->n += 1;
  parent->w += child->q;
  parent->q = parent->w / parent->n;
  if (child->n > parent->max_child_n) {
    parent->max_child_n = child->n;
  }
}

}  // namespace mcts
