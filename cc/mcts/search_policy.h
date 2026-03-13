#pragma once

#include <atomic>

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
  PuctRootSelectionPolicy kind;
  float c_puct = 1.0f;
  float c_puct_visit_scaling = 0.45f;
  float c_puct_v_2 = 3;
  bool use_puct_v = false;
  bool enable_var_scaling = false;
  float tau = 1.0f;
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
        c_puct_v_2_(params.c_puct_v_2),
        use_puct_v_(params.use_puct_v),
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
        qvars[a] = child->v_var;
      }
    }
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
      const float c_puct_var_parent_scale_factor = [&]() {
        if (n < 3) {
          return 1.0;
        }
        const float stddev = std::sqrt(v_var);
        const float interpolated_stddev =
            (0.8 + v_var * child_visits[a]) / (child_visits[a] + 2);
        return 1.0 + 0.85 * (interpolated_stddev / 0.4);
      }();
      // Gives children with high variance an extra exploration bonus. Less
      // principled version of PUCT-V.
      const float c_puct_var_child_scale_factor = [&]() {
        const int child_n = child_visits[a];
        if (child_visits[a] < 3 || n < 3) {
          return 1.0f;
        }
        constexpr int kPriorWeight = 3;
        return (kPriorWeight + child_n * std::sqrt(qvars[a] / v_var)) /
               float(kPriorWeight + child_n);
      }();
      const float c_puct_var_scale_factor = c_puct_var_child_scale_factor;
      const float canonical_child_n =
          n_fn_(child_visits[a], child_visits_in_flight[a]);
      const float canonical_q =
          q_fn_(child_visits[a] > 0 ? qs[a] : v_fpu, child_visits[a],
                child_visits_in_flight[a]);

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
      pucts[a] = {a, puct_explore_term + canonical_q};
    }
    return pucts;
  }
  std::array<std::pair<int, float>, 4> TopScores(
      const TreeNode* node, const game::Game& game,
      game::Color color_to_move) const {
    PuctScores pucts = ComputeScores(node);
    std::sort(pucts.begin(), pucts.end(), [](const auto& p0, const auto& p1) {
      return p0.second > p1.second;
    });

    std::array<std::pair<int, float>, 4> top_scores;
    int ranking = 0;
    for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
      if (ranking >= top_scores.size()) {
        break;
      }
      auto [a, puct] = pucts[i];
      if (game.IsValidMove(a, color_to_move)) {
        top_scores[ranking] = {a, puct};
        ++ranking;
      }
    }

    for (int r = ranking; r < top_scores.size(); ++r) {
      top_scores[r] = {game::kNoopLoc, -1000};
    }
    return top_scores;
  }

  game::Loc TopMove(const TreeNode* node, const game::Game& game,
                    game::Color color_to_move) const {
    PuctScores pucts = ComputeScores(node);
    std::sort(pucts.begin(), pucts.end(), [](const auto& p0, const auto& p1) {
      return p0.second > p1.second;
    });
    for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
      auto [a, _] = pucts[i];
      if (game.IsValidMove(a, color_to_move)) {
        return game::AsLoc(a);
      }
    }

    // No valid moves. Should never get here.
    CHECK(false) << "No valid moves in PUCT selection.";
    return game::kNoopLoc;
  }

 private:
  const float c_puct_;
  const float c_puct_visit_scaling_;
  const bool enable_var_scaling_;
  const float c_puct_v_2_;
  const bool use_puct_v_;
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
