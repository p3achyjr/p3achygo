#include "cc/mcts/tree.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

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

float ZScore(float alpha, int n) {
  boost::math::students_t dist(n);
  return boost::math::quantile(boost::math::complement(dist, alpha / 2));
}

std::pair<float, float> ConfidenceInterval(const TreeNode* node, int action,
                                           float alpha) {
  static constexpr float kMinLcb = -1e6f;
  static constexpr float kMaxUcb = 1e6f;
  float n = NAction(node, action);
  if (!node->child(action) || n < 2) return {kMinLcb + n, kMaxUcb - n};

  float stddev = std::sqrt(QVar(node, action) / n);
  float z = std::abs(alpha - kAlpha) < 1e-5 ? CachedQuantile(n - 1)
                                            : ZScore(alpha, n - 1);
  return {Q(node, action) - z * stddev, Q(node, action) + z * stddev};
}

}  // namespace

float Lcb(const TreeNode* node, int action) {
  return ConfidenceInterval(node, action, kAlpha).first;
}

float Ucb(const TreeNode* node, int action) {
  return ConfidenceInterval(node, action, kAlpha).second;
}

float Lcb(const TreeNode* node, int action, float alpha) {
  return ConfidenceInterval(node, action, alpha).first;
}

float Ucb(const TreeNode* node, int action, float alpha) {
  return ConfidenceInterval(node, action, alpha).second;
}

std::string VCategoricalHistogram(TreeNode* node, int granularity) {
  if (node == nullptr) return "";

  // Aggregate raw buckets into display buckets.
  std::vector<uint64_t> display(granularity, 0);
  for (int i = 0; i < kNumVBuckets; ++i) {
    float center = (i + 0.5f) * kBucketRange - 1.0f;
    int j = static_cast<int>((center + 1.0f) / 2.0f * granularity);
    j = std::clamp(j, 0, granularity - 1);
    display[j] += node->v_categorical[i];
  }

  // Find first/last occupied bucket to bound the display range.
  int lo = granularity, hi = -1;
  for (int j = 0; j < granularity; ++j) {
    if (display[j] > 0) {
      if (j < lo) lo = j;
      if (j > hi) hi = j;
    }
  }
  if (hi < 0) return "";

  uint64_t max_count = 0;
  for (int j = lo; j <= hi; ++j) {
    max_count = std::max(max_count, display[j]);
  }

  static constexpr int kMaxBarWidth = 24;
  float bucket_width = 2.0f / granularity;
  float v_min = lo * bucket_width - 1.0f;
  float v_max = (hi + 1) * bucket_width - 1.0f;

  std::stringstream ss;
  ss << absl::StrFormat("  V [%+.2f .. %+.2f]\n", v_min, v_max);
  for (int j = lo; j <= hi; ++j) {
    float center = (j + 0.5f) * bucket_width - 1.0f;
    int bar_len = max_count > 0
                      ? static_cast<int>(static_cast<float>(display[j]) /
                                         max_count * kMaxBarWidth)
                      : 0;
    ss << absl::StrFormat("  %+5.2f │", center);
    for (int k = 0; k < bar_len; ++k) ss << "█";
    for (int k = bar_len; k < kMaxBarWidth; ++k) ss << " ";
    ss << absl::StrFormat("│ %lu\n", display[j]);
  }

  return ss.str();
}

}  // namespace mcts
