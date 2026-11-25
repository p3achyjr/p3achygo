#ifndef GTP_PASS_GUARD_H_
#define GTP_PASS_GUARD_H_

#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/tree.h"

namespace gtp {

// If the opponent answers our pass with a pass, the game ends on the current
// board, counted with `game`'s scoring (Tromp-Taylor in the GTP service, as on
// CGOS: no dead-stone removal). If `move` is a pass and that count does not
// win for `color`, returns the best legal non-pass root action by
// BestLcbAction, or pass if there is none. Otherwise returns `move`.
game::Loc AvoidLosingPass(game::Game& game, const mcts::TreeNode* root,
                          game::Color color, float lcb_min_visit_frac,
                          game::Loc move);

}  // namespace gtp

#endif
