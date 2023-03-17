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

struct TreeNode {
  // immutable
  TreeNodeState state = TreeNodeState::kNew;
  bool is_terminal = false;
  int color_to_move;

  // change throughout search
  int n = 0;
  float w = 0;
  float q = 0;

  int max_child_n = 0;

  std::unique_ptr<TreeNode> children[constants::kMaxNumMoves]{};

  // write-once
  float move_logits[constants::kMaxNumMoves]{};
  float move_probabilities[constants::kMaxNumMoves]{};
  float value_estimate = 0;
  float score_estimate = 0;
  float init_utility_estimate = 0;  // mix value estimate and score estimate.
};

void AdvanceState(TreeNode* node);
float N(TreeNode* node);
float Q(TreeNode* node);
int MaxN(TreeNode* node);

// backward pass for a single parent and child
// Prerequisite: `child` is a valid child of `parent`
void UpdateParentFromChild(TreeNode* parent, TreeNode* child);

// void IncrementChildN(TreeNode* node, int child);

}  // namespace mcts

#endif  // __MCTS_TREE_H_