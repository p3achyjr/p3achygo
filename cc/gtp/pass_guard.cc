#include "cc/gtp/pass_guard.h"

#include "cc/constants/constants.h"

namespace gtp {

game::Loc AvoidLosingPass(game::Game& game, const mcts::TreeNode* root,
                          game::Color color, float lcb_min_visit_frac,
                          game::Loc move) {
  if (move != game::kPassLoc) return move;
  const game::Scores scores = game.GetScores();
  const float margin = color == BLACK ? scores.black_score - scores.white_score
                                      : scores.white_score - scores.black_score;
  if (margin > 0) return move;
  const int a = mcts::BestLcbAction(root, lcb_min_visit_frac, [&](int a) {
    return a != constants::kPassMoveEncoding && game.IsValidMove(a, color);
  });
  return a < 0 ? move : game::AsLoc(a);
}

}  // namespace gtp
