#include "cc/mcts/search_policy.h"

#include "cc/game/board.h"

namespace mcts {
namespace {
using namespace ::game;
}

PuctSearchPolicy::PuctSearchPolicy(PuctParams params) : params_(params) {}

Loc PuctSearchPolicy::SelectNextAction(const TreeNode* node,
                                       const game::Game& game,
                                       game::Color color_to_move) const {
  PuctScorer puct_scorer(params_, IdentityQ{}, IdentityN{});
  return puct_scorer.TopMove(node, game, color_to_move);
}
}  // namespace mcts
