#include "cc/mcts/search_policy.h"

namespace mcts {
namespace {
using namespace ::game;
}

PuctSearchPolicy::PuctSearchPolicy(PuctParams params)
    : c_puct_(params.c_puct),
      c_puct_visit_scaling_(params.c_puct_visit_scaling),
      enable_var_scaling_(params.enable_var_scaling),
      c_puct_v_2_(params.c_puct_v_2),
      use_puct_v_(params.use_puct_v) {}

Loc PuctSearchPolicy::SelectNextAction(const TreeNode* node,
                                       const game::Game& game,
                                       game::Color color_to_move) const {
  static constexpr float kFPU = 0.2f;
  DCHECK(node->state != TreeNodeState::kNew);

  // Scale c_puct according to visit count.
  const float c_puct_visit_scaling_term =
      c_puct_visit_scaling_ * std::log((N(node) + 500.0f) / 500.0f);
  float c_puct = c_puct_ + c_puct_visit_scaling_term;
  float c_puct_v_2 = c_puct_v_2_ + c_puct_visit_scaling_term;
  if (enable_var_scaling_ && N(node) > 3) {
    static constexpr float kBaseStdDev = 0.15;
    float var = node->v_var == 0 ? kBaseStdDev * kBaseStdDev : node->v_var;
    float stddev = std::sqrt(var);
    float scale = stddev / kBaseStdDev;
    c_puct *= scale;
  }

  // Find total explored policy.
  float p_explored = 0.0f;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (NAction(node, a) > 0) {
      p_explored += node->move_probs[a];
    }
  }

  // Fallback V for unexplored children.
  float v_fpu = V(node) - kFPU * std::sqrt(p_explored);

  // Compute PUCT values.
  std::array<std::pair<int, float>, constants::kMaxMovesPerPosition> pucts{};
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    const auto compute_u = [&]() {
      if (use_puct_v_) {
        float var = NAction(node, a) < 3 ? (N(node) < 3 ? 1.0f : node->v_var)
                                         : node->child(a)->v_var;
        float stddev = std::sqrt(var);
        float var_scale_term = node->move_probs[a] * stddev *
                               (std::sqrt(N(node)) / (1 + NAction(node, a)));
        float n_scale_term =
            node->move_probs[a] * std::log(N(node)) / (1 + NAction(node, a));
        return c_puct * var_scale_term + c_puct_v_2 * n_scale_term;
      } else {
        return c_puct * node->move_probs[a] *
               (std::sqrt(N(node)) / (1 + NAction(node, a)));
      }
    };
    TreeNode* child = node->children[a];
    float u = compute_u();
    float v = child == nullptr ? v_fpu : Q(node, a);
    pucts[a] = {a, u + v};
  }

  // Reverse sort so the highest puct value is in front.
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
}  // namespace mcts
