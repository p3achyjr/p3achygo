#ifndef MCTS_TREE_H_
#define MCTS_TREE_H_

#include <cmath>
#include <functional>
#include <memory>

#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/relaxed_atomic.h"
#include "cc/game/color.h"
#include "cc/game/zobrist.h"
#include "cc/mcts/constants.h"

namespace mcts {

enum class TreeNodeState {
  kNew = 0,          // new node. No evaluation has been performed.
  kPending = 1,      // Some thread has claimed evaluation of this node.
  kNnEvaluated = 2,  // NN evaluation has been done.
};

struct TreeNode final {
  TreeNode(game::Zobrist::Hash board_hash) : board_hash(board_hash) {};
  ~TreeNode() {
    constexpr float kBiasCacheForgetWeight = 0.8f;
    if (!bias_cache_entry) {
      return;
    }

    // TODO: make this thread-safe with BiasCache updates.
    bias_cache_entry->first -= (kBiasCacheForgetWeight * last_obs_bias_term);
    bias_cache_entry->second -= (kBiasCacheForgetWeight * last_weight_term);
  }

  const game::Zobrist::Hash board_hash;
  std::atomic<TreeNodeState> state = TreeNodeState::kNew;
  bool is_terminal = false;
  game::Color color_to_move;

  // change throughout search
  // Written under `mu` (or by a single thread); read by other search threads
  // without the lock, so they are relaxed atomics (plain loads/stores on
  // x86-64). Writers must still be serialized.
  core::RelaxedAtomic<int> n{0};
  core::RelaxedAtomic<float> w{0};
  core::RelaxedAtomic<float> v{0};
  core::RelaxedAtomic<float> v_var{0};
  core::RelaxedAtomic<double> v_m3{
      0};  // non-standardized skewness (3rd moment)

  std::array<float, kNumVBuckets> v_categorical_prior{};
  std::array<core::RelaxedAtomic<uint32_t>, kNumVBuckets> v_categorical{};
  // combined upside/downside var. The weight of v_categorical_prior is
  // specified by the user.
  core::RelaxedAtomic<float> v_cat_var_up{0};
  core::RelaxedAtomic<float> v_cat_var_down{0};

  core::RelaxedAtomic<float> w_outcome{0};
  core::RelaxedAtomic<float> v_outcome{0};
  core::RelaxedAtomic<float> v_outcome_var{0};
  core::RelaxedAtomic<double> v_outcome_m3{
      0};  // non-standardized skewness (3rd moment)
  core::RelaxedAtomic<float> score{0};
  core::RelaxedAtomic<float> v_err{0};

  core::RelaxedAtomic<int> max_child_n{0};

  std::array<std::atomic<TreeNode*>, constants::kMaxMovesPerPosition>
      children{};
  std::array<core::RelaxedAtomic<int>, constants::kMaxMovesPerPosition>
      child_visits{};

  // initial evaluation (conceptual write-once)
  std::array<float, constants::kMaxMovesPerPosition> move_logits{};
  std::array<float, constants::kMaxMovesPerPosition> move_probs{};
  std::array<float, constants::kMaxMovesPerPosition> opt_probs{};
  float init_outcome_est = 0;
  float init_score_est = 0;
  float init_score_var =
      0;                    // Var[score] under the NN's predicted distribution.
  float init_util_est = 0;  // mix value estimate and score estimate.
  float init_err_est = 0;   // initial NN estimate of err.

  inline TreeNode* child(int a) const {
    if (a < 0 || a >= constants::kMaxMovesPerPosition) {
      return nullptr;
    }

    return children[a].load(std::memory_order_relaxed);
  }

  // Bias cache stuff.
  using BiasCacheEntry = std::pair<float, float>;
  std::shared_ptr<BiasCacheEntry> bias_cache_entry;
  float last_obs_bias_term = 0;
  float last_weight_term = 0;

  // Parallel search stuff.
  absl::Mutex mu;
  std::atomic<int> n_in_flight = 0;
  std::atomic<bool> is_pending_update = false;
  std::atomic<int> sum_n_in_flights = 0;
};

inline float N(const TreeNode* node) { return node == nullptr ? 0 : node->n; }
inline float NAction(const TreeNode* node, int action) {
  return node->child_visits[action];
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
  return !node->children[action]
             ? kMinQ
             : -node->children[action].load(std::memory_order_relaxed)->v;
}

inline float QOutcome(const TreeNode* node, int action) {
  // remember to flip sign.
  return !node->children[action] ? -1.0
                                 : -node->children[action]
                                        .load(std::memory_order_relaxed)
                                        ->v_outcome;
}

inline float QVar(const TreeNode* node, int action) {
  return node == nullptr ? kMaxQ : VVar(node->children[action]);
}

inline float QOutcomeVar(const TreeNode* node, int action) {
  return node == nullptr ? 1.0f : VOutcomeVar(node->children[action]);
}

inline float Score(const TreeNode* node) {
  // 0.0f, not 0: RelaxedAtomic's ctor is explicit, so `? 0 :` makes the
  // conditional int and truncates the score.
  return node == nullptr ? 0.0f : node->score;
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
  return !node->children[action] ? node->init_score_est
                                 : -node->children[action]
                                        .load(std::memory_order_relaxed)
                                        ->init_score_est;
}

// Returns LCB/UCB for the value of a child node.
float Lcb(const TreeNode* node, int action);
float Ucb(const TreeNode* node, int action);
float Lcb(const TreeNode* node, int action, float alpha);
// Lower 95% (two-sided t) bound of the outcome winrate in [0, 1] of child
// `action`, parent's perspective: (QOutcome - t * sqrt(QOutcomeVar / N) + 1)
// / 2. 0 when the child has fewer than 2 visits.
float OutcomeWinrateLcb(const TreeNode* node, int action);
// Root move pick by LCB: the action with the highest Lcb() among those with at
// least `min_visit_frac` of the most-visited action's visits; actions below
// that rank after every action above it, by visits. Exact ties go to the higher
// prior (move_probs). Returns the best-ranked action for which `is_valid`
// holds, or -1 if none does.
int BestLcbAction(const TreeNode* node, float min_visit_frac,
                  const std::function<bool(int)>& is_valid);
float Ucb(const TreeNode* node, int action, float alpha);

inline void AdvanceState(TreeNode* node) {
  if (node == nullptr) return;
  node->state.store(TreeNodeState::kNnEvaluated, std::memory_order_release);
}

// CDF of Student's t with 3 degrees of freedom.
inline double TDist3Cdf(const double t) {
  const double x = t / std::sqrt(3.0);
  return 0.5 + (x / (1.0 + x * x) + std::atan(x)) / M_PI;
}

// KataGo's value weighting (Search::downweightBadChildrenAndNormalizeWeight):
// a child's weight is its edge visits scaled by p^exponent, where p is the t3
// CDF of how far the child's utility (parent's perspective) sits from
// `mean_util`, the visit-weighted mean child utility, in units of a visit-based
// stdev. The caller rescales the weights to the visits' total.
inline float ValueWeight(const float visits, const float child_util,
                         const double mean_util, const float exponent) {
  const double precision = 1.5 * std::sqrt(visits);
  const double stdev = std::sqrt(0.00000001 + 1.0 / precision);
  const double z = (child_util - mean_util) / stdev;
  return visits * std::pow(TDist3Cdf(z) + 0.0001, exponent);
}

// Visit-weighted mean utility of the visited children, parent's perspective.
inline double MeanChildUtility(const TreeNode* node) {
  double total_visits = 0.0, weighted_utility = 0.0;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    const TreeNode* child = node->child(a);
    if (child == nullptr || node->child_visits[a] == 0) continue;
    total_visits += node->child_visits[a];
    weighted_utility += node->child_visits[a] * -child->v;
  }
  return total_visits > 0.0 ? weighted_utility / total_visits : 0.0;
}

inline void RecomputeNodeStats(TreeNode* node, const float obs_bias = 0.0f,
                               const float value_weight_exponent = 0.0f) {
  // Children are weighted by their edge visits. With value_weight_exponent > 0
  // they are weighted by ValueWeight() instead, rescaled to the same total.
  // That needs the mean child utility first (one extra pass); the weights are
  // computed in the accumulation loop and the rescale applied to the sums.
  const bool value_weighted = value_weight_exponent > 0.0f;
  const double mean_child_util = value_weighted ? MeanChildUtility(node) : 0.0;
  std::array<float, constants::kMaxMovesPerPosition> weights;  // if weighted
  float visit_sum = 0.0f, weight_sum = 0.0f;

  const float adj_init_util_est = node->init_util_est - obs_bias;
  float w = adj_init_util_est, w_outcome = node->init_outcome_est,
        total_score = node->init_score_est, w_err = node->init_err_est;
  int max_child_n = 0;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    // flip signs.
    const TreeNode* child = node->child(a);
    if (child == nullptr) {
      continue;
    }
    float n_a = node->child_visits[a];
    if (value_weighted && n_a > 0) {
      visit_sum += n_a;
      n_a = ValueWeight(n_a, -child->v, mean_child_util, value_weight_exponent);
      weight_sum += n_a;
      weights[a] = n_a;
    }
    w -= (n_a * child->v);
    w_outcome -= (n_a * child->v_outcome);
    total_score -= (n_a * child->score);
    max_child_n = node->child_visits[a] > max_child_n ? node->child_visits[a]
                                                      : max_child_n;
    w_err += n_a * child->v_err;
  }
  // Rescale the children's part of each sum so the weights total the visits.
  const float weight_scale =
      value_weighted && weight_sum > 0.0f ? visit_sum / weight_sum : 1.0f;
  if (weight_scale != 1.0f) {
    w = adj_init_util_est + weight_scale * (w - adj_init_util_est);
    w_outcome = node->init_outcome_est +
                weight_scale * (w_outcome - node->init_outcome_est);
    total_score = node->init_score_est +
                  weight_scale * (total_score - node->init_score_est);
    w_err = node->init_err_est + weight_scale * (w_err - node->init_err_est);
  }
  float v = w / node->n;
  float v_outcome = w_outcome / node->n;
  float score = total_score / node->n;
  float v_err = w_err / node->n;

  // Variance of a mixture distribution:
  // Var(Y) = (1/N) * sum(i : 0...k) ni(vi + (mi - m)^2)
  // Third moment of a mixture distribution:
  // Define d_i = u_i - u
  // M3(Y) = sum(i : 0...k) ni(m3_i + 3 * var_i * d_i + d_i ^ 3)
  // Derived by defining X - u = X - u_i + d_i, and solving for E[(X - u)^3]
  float m = adj_init_util_est - v;
  float m_outcome = node->init_outcome_est - v_outcome;
  float m2 = m * m;
  float m2_outcome = m_outcome * m_outcome;
  double m3 = m2 * m;
  double m3_outcome = m2_outcome * m_outcome;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (node->child_visits[a] == 0) {
      continue;
    }

    const float n_a =
        value_weighted ? weights[a] * weight_scale : node->child_visits[a];
    const TreeNode* child = node->child(a);
    const float dv = -child->v - v;
    const float dv_outcome = -child->v_outcome - v_outcome;
    const float dv2 = dv * dv;
    const float dv_outcome2 = dv_outcome * dv_outcome;
    const float dv3 = dv2 * dv;
    const float dv_outcome3 = dv_outcome2 * dv_outcome;
    m2 += n_a * (child->v_var + dv2);
    m2_outcome += n_a * (child->v_outcome_var + dv_outcome2);
    m3 += n_a * (-child->v_m3 + 3 * child->v_var * dv + dv3);
    m3_outcome += n_a * (-child->v_outcome_m3 +
                         3 * child->v_outcome_var * dv_outcome + dv_outcome3);
  }

  node->w = w;
  node->w_outcome = w_outcome;
  node->v = v;
  node->v_outcome = v_outcome;
  node->score = score;
  node->max_child_n = max_child_n;
  node->v_var = m2 / node->n;
  node->v_outcome_var = m2_outcome / node->n;
  node->v_m3 = m3 / node->n;
  node->v_outcome_m3 = m3_outcome / node->n;
  node->v_err = v_err;
}

// Snapshot of node->v_categorical as plain counts.
inline std::array<uint32_t, kNumVBuckets> VCategorical(const TreeNode* node) {
  std::array<uint32_t, kNumVBuckets> counts;
  for (int i = 0; i < kNumVBuckets; ++i) counts[i] = node->v_categorical[i];
  return counts;
}

inline void UpdateVCategorial(TreeNode* node, const float leaf_q_outcome) {
  // Add V to bucket.
  // This is inaccurate with idempotent updates. We will miss entries from
  // transposing paths in MCGS, and will not account for bias correction from
  // the bias cache. However, idempotent updates to this field are very
  // expensive, so we will keep it incremental.
  const int v_bucket =
      std::clamp(static_cast<int>((leaf_q_outcome + 1.0f) / kBucketRange), 0,
                 kNumVBuckets - 1);
  node->v_categorical[v_bucket] += 1;
}

// Must be called after node->v_outcome is recomputed.
inline void RecomputeVCategoricalStats(TreeNode* node,
                                       const int v_cat_prior_visits = 0) {
  // Recompute up/down variance.
  const float v_mean = [&]() {
    float w = 0.0f, n = 0.0f;
    for (int i = 0; i < kNumVBuckets; ++i) {
      const float bucket_mean =
          (i - constants::kVBucketMidpoint) * kBucketRange;
      const float bucket_n = v_cat_prior_visits * node->v_categorical_prior[i] +
                             node->v_categorical[i];
      w += bucket_mean * bucket_n;
      n += bucket_n;
    }
    return n > 0 ? (w / n) : 0.0f;
  }();
  float dist2_down = 0, dist2_up = 0;
  float n_down = 0, n_up = 0;
  for (int i = 0; i < kNumVBuckets; ++i) {
    const float bucket_mean = (i - constants::kVBucketMidpoint) * kBucketRange;
    const float bucket_n = v_cat_prior_visits * node->v_categorical_prior[i] +
                           node->v_categorical[i];
    const float l1_dist = bucket_mean - v_mean;
    if (bucket_mean < v_mean) {
      dist2_down += bucket_n * l1_dist * l1_dist;
      n_down += bucket_n;
    } else {
      dist2_up += bucket_n * l1_dist * l1_dist;
      n_up += bucket_n;
    }
  }

  node->v_cat_var_down = n_down > 0 ? dist2_down / n_down : 0.0f;
  node->v_cat_var_up = n_up > 0 ? dist2_up / n_up : 0.0f;
}

std::string VCategoricalHistogram(TreeNode* node, int granularity = 17);

}  // namespace mcts

#endif  // MCTS_TREE_H_
