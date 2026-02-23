#pragma once

#include "cc/game/color.h"
#include "cc/game/game.h"
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
#if 0
template <typename QFn, typename NFn>
class PuctScorer final {
  PuctScorer(PuctParams params, const QFn& q_fn, const NFn& n_fn);
  PuctScores ComputeScores(const TreeNode* node, const game::Game& game,
                           game::Color color_to_move);
  std::array<std::pair<int, float>, 4> TopScores(const TreeNode* node,
                                                 const game::Game& game,
                                                 game::Color color_to_move);

 private:
  const float c_puct_;
  const float c_puct_visit_scaling_;
  const bool enable_var_scaling_;
  const float c_puct_v_2_;
  const bool use_puct_v_;
  const QFn& q_fn_;
  const NFn& n_fn_;
};
#endif

class PuctSearchPolicy : public SearchPolicy {
 public:
  PuctSearchPolicy(PuctParams params);
  game::Loc SelectNextAction(const TreeNode* node, const game::Game& game,
                             game::Color color_to_move) const override;

 private:
  const float c_puct_;
  const float c_puct_visit_scaling_;
  const bool enable_var_scaling_;
  const float c_puct_v_2_;
  const bool use_puct_v_;
};

/*
 * Various Q/N functions
 */
struct IdentityQ final {
  inline float operator()(const float q, const int n, const int n_in_flight) {
    return q;
  }
};

struct VirtualLossQ final {
  VirtualLossQ(const float v_loss_penalty) : v_loss_penalty(v_loss_penalty) {}
  inline float operator()(const float q, const int n, const int n_in_flight) {
    return q - n_in_flight * v_loss_penalty;
  }

 private:
  const float v_loss_penalty;
};
}  // namespace mcts
