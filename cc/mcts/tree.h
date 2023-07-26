#ifndef MCTS_TREE_H_
#define MCTS_TREE_H_

#include <cstddef>
#include <memory>

#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/mcts/constants.h"

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
  game::Color color_to_move;

  // change throughout search
  int n = 0;
  float w = 0;
  float v = 0;

  float w_outcome = 0;
  float v_outcome = 0;

  int max_child_n = 0;

  std::array<std::unique_ptr<TreeNode>, constants::kMaxMovesPerPosition>
      children{};

  // write-once
  std::array<float, constants::kMaxMovesPerPosition> move_logits{};
  std::array<float, constants::kMaxMovesPerPosition> move_probs{};
  float outcome_est = 0;
  float score_est = 0;
  float init_util_est = 0;  // mix value estimate and score estimate.
};

inline float N(TreeNode* node) { return node == nullptr ? 0 : node->n; }
inline float NAction(TreeNode* node, int action) {
  return N(node->children[action].get());
}

inline float V(TreeNode* node) {
  // return minimum value if node is null (init-to-loss).
  return node == nullptr ? kMinQ : node->v;
}

inline float VOutcome(TreeNode* node) {
  // return minimum value if node is null (init-to-loss).
  return node == nullptr ? -1.0 : node->v_outcome;
}

inline float Q(TreeNode* node, int action) {
  // remember to flip sign. In bare MCTS, this will also cause MCTS to make deep
  // reads.
  return !node->children[action] ? kMinQ : -node->children[action]->v;
}

inline float QOutcome(TreeNode* node, int action) {
  // remember to flip sign. In bare MCTS, this will also cause MCTS to make deep
  // reads.
  return !node->children[action] ? -1.0 : -node->children[action]->v_outcome;
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

inline float ChildScore(TreeNode* node, int action) {
  return !node->children[action] ? node->score_est
                                 : -node->children[action]->score_est;
}

void AdvanceState(TreeNode* node);

}  // namespace mcts

#endif  // MCTS_TREE_H_
