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
