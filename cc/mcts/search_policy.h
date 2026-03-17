#pragma once

#include <atomic>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/tree.h"

namespace mcts {
enum class PuctRootSelectionPolicy : uint8_t {
  kVisitCount = 0,
  kLcb = 1,
  kVisitCountSample = 2,
};

struct PuctParams {
  PuctRootSelectionPolicy kind = PuctRootSelectionPolicy::kVisitCount;
  float c_puct = 1.0f;
  float c_puct_visit_scaling = 0.45f;
  float c_puct_v_2 = 3.0f;
  bool use_puct_v = false;
  bool enable_var_scaling = false;
  // Prior visits for the variance-based child exploration scale factor.
  int var_scale_prior_visits = 0;
  float tau = 1.0f;
  bool enable_m3_bonus = false;
  // Prior visits for m3-bonus dampening (higher = slower ramp-in).
  int m3_prior_visits = 20;

  // Forward-declared; defined below once PuctParams is complete.
  class Builder;
};

class PuctParams::Builder {
 public:
  Builder() = default;
  Builder& set_kind(PuctRootSelectionPolicy v) {
    p_.kind = v;
    return *this;
  }
  Builder& set_c_puct(float v) {
    p_.c_puct = v;
    return *this;
  }
  Builder& set_c_puct_visit_scaling(float v) {
    p_.c_puct_visit_scaling = v;
    return *this;
  }
  Builder& set_c_puct_v_2(float v) {
    p_.c_puct_v_2 = v;
    return *this;
  }
  Builder& set_use_puct_v(bool v) {
    p_.use_puct_v = v;
    return *this;
  }
  Builder& set_enable_var_scaling(bool v) {
    p_.enable_var_scaling = v;
    return *this;
  }
  Builder& set_var_scale_prior_visits(int v) {
    p_.var_scale_prior_visits = v;
    return *this;
  }
  Builder& set_tau(float v) {
    p_.tau = v;
    return *this;
  }
  Builder& set_enable_m3_bonus(bool v) {
    p_.enable_m3_bonus = v;
    return *this;
  }
  Builder& set_m3_prior_visits(int v) {
    p_.m3_prior_visits = v;
    return *this;
  }
  PuctParams build() const { return p_; }

 private:
  PuctParams p_;
};

/*
 * Base class representing a search policy.
 */
class SearchPolicy {
 public:
  virtual game::Loc SelectNextAction(const TreeNode* node,
                                     const game::Game& game,
                                     game::Color color_to_move) const = 0;
  virtual ~SearchPolicy() = default;

 protected:
  SearchPolicy() = default;
};

/*
 * Gumbel non-root search policy.
 */
class GumbelNonRootSearchPolicy : public SearchPolicy {
 public:
  game::Loc SelectNextAction(const TreeNode* node, const game::Game& game,
                             game::Color color_to_move) const override;
};

/*
 * PUCT search policy.
 */
using PuctScores =
    std::array<std::pair<int, float>, constants::kMaxMovesPerPosition>;

template <typename QFn, typename NFn>
class PuctScorer final {
 public:
  PuctScorer(PuctParams params, const QFn& q_fn, const NFn& n_fn)
      : c_puct_(params.c_puct),
        c_puct_visit_scaling_(params.c_puct_visit_scaling),
        enable_var_scaling_(params.enable_var_scaling),
        var_scale_prior_visits_(params.var_scale_prior_visits),
        c_puct_v_2_(params.c_puct_v_2),
        use_puct_v_(params.use_puct_v),
        enable_m3_bonus_(params.enable_m3_bonus),
        m3_prior_visits_(params.m3_prior_visits),
        q_fn_(q_fn),
        n_fn_(n_fn){};

  static inline float ScaleCPuct(float c_puct, const float c_puct_visit_scaling,
                                 const int n) {
    const float c_puct_visit_scaling_term =
        c_puct_visit_scaling * std::log((n + 500.0f) / 500.0f);
    c_puct += c_puct_visit_scaling_term;
    return c_puct;
  }

  inline PuctScores ComputeScores(const TreeNode* node) const {
    constexpr float kFPU = 0.2f;
    DCHECK(node->state != TreeNodeState::kNew);

    // fetch everything once. this does not prevent concurrent accesses but
    // prevents key values from spuriously changing.
    const int n = node->n;
    const int n_in_flight = node->n_in_flight.load(std::memory_order_relaxed);
    const float v = node->v;
    const float v_var = node->v_var;
    const std::array<int, constants::kMaxMovesPerPosition> child_visits =
        node->child_visits;
    std::array<float, constants::kMaxMovesPerPosition> qs;
    std::array<int, constants::kMaxMovesPerPosition> child_visits_in_flight;
    std::array<float, constants::kMaxMovesPerPosition> qvars;
    std::array<double, constants::kMaxMovesPerPosition> q_m3s;
    float q_std_weighted_sum = 0.0f;
    double q_m3_std_weighted_sum = 0.0;
    for (int a = 0; a < qs.size(); ++a) {
      TreeNode* child = node->child(a);
      if (child == nullptr) {
        child_visits_in_flight[a] = 0;
        continue;
      }
      child_visits_in_flight[a] =
          child->n_in_flight.load(std::memory_order_relaxed);

      if (child_visits[a] > 0) {
        qs[a] = -child->v;
      }

      if (child_visits[a] >= 3) {
        qvars[a] = child->v_var;
        q_m3s[a] = -child->v_m3;
        q_std_weighted_sum += std::sqrt(qvars[a]) * child_visits[a];
        q_m3_std_weighted_sum += std::cbrt(q_m3s[a]) * child_visits[a];
      }
    }
    const float q_std_mean = q_std_weighted_sum / n;
    const double q_m3_std_mean = q_m3_std_weighted_sum / n;
    const float p_explored = [&]() {
      float mass = 0.0f;
      for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
        if (child_visits[a] + child_visits_in_flight[a] > 0) {
          mass += node->move_probs[a];
        }
      }
      return mass;
    }();
    const float v_fpu = v - kFPU * std::sqrt(p_explored);

    // Scale c_puct.
    const float c_puct = ScaleCPuct(c_puct_, c_puct_visit_scaling_, n);
    const float c_puct_v_2 = ScaleCPuct(c_puct_v_2_, c_puct_visit_scaling_, n);

    // Finally, compute PUCT values.
    // Compute PUCT values.
    // const float canonical_n = n_fn_(n, n_in_flight);
    const float canonical_n = [&]() {
      float total_n = 1;  // Initialize to 1 to account for visit to root. This
                          // prevents the case where the exploration term
                          // devolves to 0 when a node has no child visits.
      for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
        total_n += n_fn_(child_visits[a], child_visits_in_flight[a]);
      }
      return total_n;
    }();
    std::array<std::pair<int, float>, constants::kMaxMovesPerPosition> pucts{};
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      // Katago-style root variance scaling. I have no intuition as to why this
      // works -- it strongly compresses c_puct on most nodes.
      const auto c_puct_var_parent_scale_factor = [&]() {
        if (n < 3 || v_var == 0) {
          return 1.0;
        }
        const float stddev = std::sqrt(v_var);
        const float interpolated_stddev =
            (0.8 + v_var * child_visits[a]) / (child_visits[a] + 2);
        return 1.0 + 0.85 * (interpolated_stddev / 0.4);
      };
      // Gives children with high variance an extra exploration bonus. Less
      // principled version of PUCT-V.
      const auto c_puct_var_child_scale_factor = [&]() {
        const int child_n = child_visits[a];
        if (child_n < 3 || q_std_mean == 0) {
          return 1.0f;
        }
        const float prior_weight = static_cast<float>(var_scale_prior_visits_);
        // return (prior_weight + child_n * std::sqrt(qvars[a] / v_var)) /
        //        float(prior_weight + child_n);
        return (prior_weight + child_n * (std::sqrt(qvars[a]) / q_std_mean)) /
               float(prior_weight + child_n);
      };
      const float c_puct_var_scale_factor =
          enable_var_scaling_ ? c_puct_var_child_scale_factor() : 1.0f;
      const float canonical_child_n =
          n_fn_(child_visits[a], child_visits_in_flight[a]);
      const float canonical_q =
          q_fn_(child_visits[a] > 0 ? qs[a] : v_fpu, child_visits[a],
                child_visits_in_flight[a]);

      // Gives children with better skewness than other children a bonus.
      const auto compute_m3_bonus = [&]() -> double {
        const float prior_weight = static_cast<float>(m3_prior_visits_);
        const int child_n = child_visits[a];
        if (child_n < 3) {
          return 0.0f;
        }

        const double m3_child_std = std::cbrt(q_m3s[a]);
        // How much more of a positive tail does this child have than the
        // average child?
        const double abs_bonus = m3_child_std - q_m3_std_mean;
        return (prior_weight + abs_bonus) / double(prior_weight + child_n);
      };
      const double m3_bonus = enable_m3_bonus_ ? compute_m3_bonus() : 0.0;

      // PUCT-V formula.
      const auto compute_puct_v_explore_term = [&]() {
        float var = child_visits[a] < 3 ? (n < 3 ? 1.0f : v_var) : qvars[a];
        float stddev = std::sqrt(var);
        float var_scale_term =
            node->move_probs[a] * stddev *
            (std::sqrt(canonical_n) / (1 + canonical_child_n));
        float n_scale_term = node->move_probs[a] * std::log(canonical_n) /
                             (1 + canonical_child_n);
        return c_puct * var_scale_term + c_puct_v_2 * n_scale_term;
      };

      // PUCT formula.
      const auto compute_puct_explore_term = [&]() {
        return c_puct * c_puct_var_scale_factor * node->move_probs[a] *
               (std::sqrt(canonical_n) / (1 + canonical_child_n));
      };
      const auto compute_explore_term = [&]() {
        return use_puct_v_ ? compute_puct_v_explore_term()
                           : compute_puct_explore_term();
      };
      const float puct_explore_term = compute_explore_term();
      pucts[a] = {a, puct_explore_term + canonical_q + m3_bonus};
    }
    return pucts;
  }

  std::array<std::pair<int, float>, 4> TopScores(
      const TreeNode* node, const game::Game& game,
      const game::Color color_to_move) const {
    PuctScores pucts = ComputeScores(node);
    std::array<std::pair<int, float>, 4> top_scores = {
        {{game::kNoopLoc, kMinPuctScore},
         {game::kNoopLoc, kMinPuctScore},
         {game::kNoopLoc, kMinPuctScore},
         {game::kNoopLoc, kMinPuctScore}}};
    for (const auto& [a, puct_score] : pucts) {
      int a_ranking = 0;
      while (a_ranking < static_cast<int>(top_scores.size())) {
        if (puct_score > top_scores[a_ranking].second) {
          break;
        }
        ++a_ranking;
      }

      if (a_ranking >= top_scores.size() ||
          !game.IsValidMove(game::AsLoc(a), color_to_move)) {
        continue;
      }

      // Sift.
      for (int r = top_scores.size() - 1; r > a_ranking; --r) {
        top_scores[r] = top_scores[r - 1];
      }

      // Emplace
      top_scores[a_ranking] = {a, puct_score};
    }
    return top_scores;
  }

  game::Loc TopMove(const TreeNode* node, const game::Game& game,
                    const game::Color color_to_move) const {
    PuctScores pucts = ComputeScores(node);
    float max_puct_score = kMinPuctScore;
    game::Loc max_move = game::kNoopLoc;
    for (const auto& [a, puct_score] : pucts) {
      const game::Loc mv = game::AsLoc(a);
      if (puct_score > max_puct_score && game.IsValidMove(mv, color_to_move)) {
        max_puct_score = puct_score;
        max_move = mv;
      }
    }

    return max_move;
  }

 private:
  static constexpr float kMinPuctScore = -1e6;
  const float c_puct_;
  const float c_puct_visit_scaling_;
  const bool enable_var_scaling_;
  const int var_scale_prior_visits_;
  const float c_puct_v_2_;
  const bool use_puct_v_;
  const bool enable_m3_bonus_;
  const int m3_prior_visits_;
  QFn q_fn_;
  NFn n_fn_;
};

class PuctSearchPolicy : public SearchPolicy {
 public:
  PuctSearchPolicy(PuctParams params);
  game::Loc SelectNextAction(const TreeNode* node, const game::Game& game,
                             game::Color color_to_move) const override;

 private:
  const PuctParams params_;
};

/*
 * Q/N functions
 */
enum class QFnKind : uint8_t {
  kIdentity = 0,
  kVirtualLoss = 1,
  kVirtualLossSoft = 2,
};

enum class NFnKind : uint8_t {
  kIdentity = 0,
  kVirtualVisit = 1,
};

struct IdentityQ final {
  inline float operator()(const float q, const int n,
                          const int n_in_flight) const {
    return q;
  }
};

struct VirtualLossQ final {
  VirtualLossQ(const float vloss_delta) : vloss_delta(vloss_delta) {
    CHECK(vloss_delta <= 0);
  }
  inline float operator()(const float q, const int n,
                          const int n_in_flight) const {
    return q + n_in_flight * vloss_delta;
  }

 private:
  const float vloss_delta;
};

struct VirtualLossSoftQ final {
  VirtualLossSoftQ(const float vloss_delta) : vloss_delta(vloss_delta) {
    CHECK(vloss_delta <= 0);
  }
  inline float operator()(const float q, const int n,
                          const int n_in_flight) const {
    if (n_in_flight == 0) {
      return q;
    }
    const float q_adj = q * n + n_in_flight * vloss_delta;
    const float n_adj = n + n_in_flight;
    return q_adj / n_adj;
  }

 private:
  const float vloss_delta;
};

struct IdentityN final {
  inline float operator()(const int n, const int n_in_flight) const {
    return n;
  }
};

struct VirtualVisitN final {
  inline float operator()(const int n, const int n_in_flight) const {
    return n + n_in_flight;
  }
};
}  // namespace mcts
