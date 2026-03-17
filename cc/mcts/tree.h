#ifndef MCTS_TREE_H_
#define MCTS_TREE_H_

#include <cmath>
#include <memory>

#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
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
  // TODO: these are modified/read across threads. This is a latent data
  // corruption bug on ARM.
  int n = 0;
  float w = 0;
  float v = 0;
  float v_var = 0;
  double v_m3 = 0;  // non-standardized skewness (3rd moment)

#ifdef V_CATEGORICAL
  std::array<int16_t, kNumVBuckets> v_categorical{};
#endif

  float w_outcome = 0;
  float v_outcome = 0;
  float v_outcome_var = 0;
  double v_outcome_m3 = 0;  // non-standardized skewness (3rd moment)
  float score = 0;

  int max_child_n = 0;

  std::array<std::atomic<TreeNode*>, constants::kMaxMovesPerPosition>
      children{};
  std::array<int, constants::kMaxMovesPerPosition> child_visits{};

  // initial evaluation (conceptual write-once)
  std::array<float, constants::kMaxMovesPerPosition> move_logits{};
  std::array<float, constants::kMaxMovesPerPosition> move_probs{};
  float init_outcome_est = 0;
  float init_score_est = 0;
  float init_score_var = 0;  // Var[score] under the NN's predicted distribution.
  float init_util_est = 0;  // mix value estimate and score estimate.

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
  return node == nullptr ? 0 : node->score;
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
float Ucb(const TreeNode* node, int action, float alpha);

inline void AdvanceState(TreeNode* node) {
  if (node == nullptr) return;
  node->state.store(TreeNodeState::kNnEvaluated, std::memory_order_release);
}

inline void RecomputeNodeStats(TreeNode* node, const float obs_bias = 0.0f) {
  const float adj_init_util_est = node->init_util_est - obs_bias;
  float w = adj_init_util_est, w_outcome = node->init_outcome_est,
        total_score = node->init_score_est;
  int max_child_n = 0;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    // flip signs.
    const TreeNode* child = node->child(a);
    if (child == nullptr) {
      continue;
    }
    w -= (node->child_visits[a] * child->v);
    w_outcome -= (node->child_visits[a] * child->v_outcome);
    total_score -= (node->child_visits[a] * child->score);
    max_child_n = node->child_visits[a] > max_child_n ? node->child_visits[a]
                                                      : max_child_n;
  }
  float v = w / node->n;
  float v_outcome = w_outcome / node->n;
  float score = total_score / node->n;

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
    const int n_a = node->child_visits[a];
    if (n_a == 0) {
      continue;
    }

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
}

#ifdef V_CATEGORICAL
std::string VCategoricalHistogram(TreeNode* node);
#endif

}  // namespace mcts

#endif  // MCTS_TREE_H_
