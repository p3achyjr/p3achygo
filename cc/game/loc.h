#ifndef __GAME_LOC_H_
#define __GAME_LOC_H_

#include <iostream>
#include <utility>

#include "absl/hash/hash.h"

namespace game {

/*
 * Represents cartesian index into grid.
 */
struct Loc {
  int i;
  int j;

  // Index into a 1D representation of a 2D grid of length `len`.
  int16_t as_index(int len) const { return i * len + j; }
};

inline Loc AsLoc(int encoding, int grid_len) {
  return Loc{encoding / grid_len, encoding % grid_len};
}

template <typename H>
H AbslHashValue(H h, const Loc& loc) {
  return H::combine(std::move(h), loc.i, loc.j);
}

inline bool operator==(const Loc& x, const Loc& y) {
  return x.i == y.i && x.j == y.j;
}

inline std::ostream& operator<<(std::ostream& os, const Loc& loc) {
  return os << "Loc(" << loc.i << ", " << loc.j << ")";
}

static constexpr Loc kNoopLoc = Loc{-1, -1};
static constexpr Loc kPassLoc = Loc{19, 0};

}  // namespace game

#endif
