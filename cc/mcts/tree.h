#ifndef MCTS_TREE_H_
#define MCTS_TREE_H_

#include <cmath>
#include <memory>

#include "cc/constants/constants.h"
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
  float v_var = 0;

#ifdef V_CATEGORICAL
  std::array<int16_t, kNumVBuckets> v_categorical{};
#endif

  float w_outcome = 0;
  float v_outcome = 0;
  float v_outcome_var = 0;

  int max_child_n = 0;

  std::array<std::unique_ptr<TreeNode>, constants::kMaxMovesPerPosition>
      children{};

  // write-once
  std::array<float, constants::kMaxMovesPerPosition> move_logits{};
  std::array<float, constants::kMaxMovesPerPosition> move_probs{};
  float outcome_est = 0;
  float score_est = 0;
  float init_util_est = 0;  // mix value estimate and score estimate.

  inline TreeNode* child(int a) const {
    if (a < 0 || a >= constants::kMaxMovesPerPosition) {
      return nullptr;
    }

    return children[a].get();
  }
};

inline float N(const TreeNode* node) { return node == nullptr ? 0 : node->n; }
inline float NAction(const TreeNode* node, int action) {
  return node->children[action] ? N(node->children[action].get()) : 0;
}

inline float V(const TreeNode* node) {
  // return minimum value if node is null (init-to-loss).
  return node == nullptr ? kMinQ : node->v;
}

inline float VOutcome(const TreeNode* node) {
  // return minimum value if node is null (init-to-loss).
  return node == nullptr ? -1.0 : node->v_outcome;
}

inline float VVar(const TreeNode* node) {
  return node == nullptr || node->n < 3 ? kMaxQ : node->v_var;
}

inline float VOutcomeVar(const TreeNode* node) {
  return node == nullptr || node->n < 3 ? 1.0f : node->v_outcome_var;
}

inline float Q(const TreeNode* node, int action) {
  // remember to flip sign.
  return !node->children[action] ? kMinQ : -node->children[action]->v;
}

inline float QOutcome(const TreeNode* node, int action) {
  // remember to flip sign.
  return !node->children[action] ? -1.0 : -node->children[action]->v_outcome;
}

inline float QVar(const TreeNode* node, int action) {
  return node == nullptr ? kMaxQ : VVar(node->children[action].get());
}

inline float QOutcomeVar(const TreeNode* node, int action) {
  return node == nullptr ? 1.0f : VOutcomeVar(node->children[action].get());
}

inline float MaxN(const TreeNode* node) {
  if (node == nullptr) return 0;

  return node->max_child_n;
}

// Each node is visited once when expanded, and once per search path. Thus, the
// total visit count of its children should be N(node) - 1
inline float SumChildrenN(const TreeNode* node) {
  return node == nullptr ? 0 : node->n - 1;
}

inline float ChildScore(const TreeNode* node, int action) {
  return !node->children[action] ? node->score_est
                                 : -node->children[action]->score_est;
}

// Returns LCB for the value of a child node.
float Lcb(const TreeNode* node, int action);

void AdvanceState(TreeNode* node);

#ifdef V_CATEGORICAL
std::string VCategoricalHistogram(TreeNode* node);
#endif

}  // namespace mcts

#endif  // MCTS_TREE_H_
