#ifndef __GAME_MOVE_H_
#define __GAME_MOVE_H_

#include "absl/hash/hash.h"
#include "cc/constants/constants.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"

namespace game {

/*
 * Represents a move.
 */
struct Move {
  color color;
  Loc loc;
};

inline bool operator==(const Move& x, const Move& y) {
  return x.color == y.color && x.loc == y.loc;
}

inline std::ostream& operator<<(std::ostream& os, const Move& move) {
  char col_encoding =
      move.color == BLACK ? 'B' : (move.color == WHITE ? 'W' : 'E');
  return os << col_encoding << ": " << move.loc;
}

template <typename H>
H AbslHashValue(H h, const Move& move) {
  return H::combine(std::move(h), move.color, move.loc);
}

static constexpr Move kNoopMove = Move{EMPTY, kNoopLoc};

}  // namespace game

#endif
