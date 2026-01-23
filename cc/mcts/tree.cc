#include "cc/mcts/tree.h"

#include <iostream>
#include <sstream>

#include "absl/strings/str_format.h"
#include "boost/math/distributions/students_t.hpp"

namespace mcts {
namespace {

// https://github.com/leela-zero/leela-zero/blob/next/src/Utils.cpp#L56
static constexpr size_t kNumZEntries = 1000;
static constexpr float kAlpha = 0.05f;
static const std::array<float, kNumZEntries> kZTable = []() {
  std::array<float, kNumZEntries> z_table;
  for (int i = 1; i < kNumZEntries + 1; ++i) {
    boost::math::students_t dist(i);
    auto z = boost::math::quantile(boost::math::complement(
        dist, kAlpha / 2));  // Divide by 2 for double-sided lookup.
    z_table[i - 1] = z;
  }

  return z_table;
}();

float CachedQuantile(const int v) {
  if (v < 1) return kZTable[0];
  if (v < kNumZEntries) return kZTable[v - 1];
  return kZTable.back();
}

}  // namespace

float Lcb(const TreeNode* node, int action) {
  static constexpr float kMinLcb = -1e6f;
  float n = NAction(node, action);
  if (!node->child(action) || n < 2) return kMinLcb + n;

  float stddev = std::sqrt(QVar(node, action) / n);
  float z = CachedQuantile(n - 1);
  return Q(node, action) - z * stddev;
}

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

std::unique_ptr<TreeNode> CloneTree(const TreeNode* node) {
  if (node == nullptr) {
    return nullptr;
  }

  // Create a new node with the same board hash
  auto cloned = std::make_unique<TreeNode>(node->board_hash);

  // Copy all scalar fields
  cloned->state = node->state;
  cloned->is_terminal = node->is_terminal;
  cloned->color_to_move = node->color_to_move;
  cloned->n = node->n;
  cloned->w = node->w;
  cloned->v = node->v;
#ifdef V_CATEGORICAL
  cloned->v_categorical = node->v_categorical;
#endif
  cloned->w_outcome = node->w_outcome;
  cloned->v_outcome = node->v_outcome;
  cloned->v_outcome_var = node->v_outcome_var;
  cloned->max_child_n = node->max_child_n;
  cloned->child_visits = node->child_visits;
  cloned->move_logits = node->move_logits;
  cloned->move_probs = node->move_probs;
  cloned->outcome_est = node->outcome_est;
  cloned->score_est = node->score_est;
  cloned->init_util_est = node->init_util_est;

  // Recursively clone children that have been visited (n > 0)
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    if (node->children[i] != nullptr && node->child_visits[i] > 0) {
      cloned->children[i] = CloneTree(node->children[i]).release();
    }
  }

  return cloned;
}

void DeleteClonedTree(TreeNode* node) {
  if (node == nullptr) {
    return;
  }
  for (int a = 0; a < node->children.size(); ++a) {
    TreeNode* child = node->children[a];
    DeleteClonedTree(child);
  }
  delete node;
}

#ifdef V_CATEGORICAL
std::string VCategoricalHistogram(TreeNode* node) {
  if (node == nullptr) return "";

  std::stringstream ss;
  for (int i = 0; i < kNumVBuckets; ++i) {
    float lb = i * kBucketRange - 1.0f;
    float ub = (i + 1) * kBucketRange - 1.0f;
    ss << absl::StrFormat("%.4f", (lb + 1.0f) / 2);
    ss << " - ";
    ss << absl::StrFormat("%.4f", (ub + 1.0f) / 2);
    ss << ": ";

    for (int _ = 0; _ < node->v_categorical[i]; ++_) {
      ss << "#";
    }
    ss << "\n";
  }

  return ss.str();
}
#endif

}  // namespace mcts
