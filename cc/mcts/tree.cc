#include "cc/mcts/tree.h"

#include <iostream>
#include <sstream>

#include "absl/strings/str_format.h"

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

}  // namespace mcts
