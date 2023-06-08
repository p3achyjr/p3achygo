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

  int max_child_n = 0;

  std::array<std::unique_ptr<TreeNode>, constants::kMaxNumMoves> children{};

  // write-once
  float move_logits[constants::kMaxNumMoves]{};
  float move_probs[constants::kMaxNumMoves]{};
  float value_est = 0;
  float score_est = 0;
  float init_util_est = 0;  // mix value estimate and score estimate.
};

void AdvanceState(TreeNode* node);
float N(TreeNode* node);
float NAction(TreeNode* node, int action);
float Q(TreeNode* node);
float QAction(TreeNode* node, int action);
float MaxN(TreeNode* node);
float SumChildrenN(TreeNode* node);

// void IncrementChildN(TreeNode* node, int child);

}  // namespace mcts

#endif  // __MCTS_TREE_H_
