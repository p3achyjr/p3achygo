#ifndef __CC_GAME_SYMMETRY_H_
#define __CC_GAME_SYMMETRY_H_

#include <array>
#include <cstdint>

#include "cc/core/rand.h"
#include "cc/game/loc.h"

namespace game {
static constexpr int kSymUpperBound = 8;
enum class Symmetry : uint8_t {
  kIdentity = 0,
  kRot90 = 1,
  kRot180 = 2,
  kRot270 = 3,
  kFlip = 4,  // Flip across vertical line.
  kFlipRot90 = 5,
  kFlipRot180 = 6,
  kFlipRot270 = 7,
};

enum class Rotation : uint8_t {
  k90 = 0,
  k180 = 1,
  k270 = 2,
};

int Rotate(int index, size_t grid_len, Rotation rot);
int TransformIndex(Symmetry sym, int index, size_t grid_len);
int TransformInv(Symmetry sym, int index, size_t grid_len);

Symmetry GetRandomSymmetry(core::PRng& prng);

Loc ApplySymmetry(Symmetry sym, Loc loc, size_t grid_len);

template <typename T, size_t N>
std::array<T, N> ApplySymmetry(Symmetry sym, const std::array<T, N>& grid,
                               size_t grid_len) {
  std::array<T, N> sym_grid;
  for (size_t i = 0; i < N; ++i) {
    sym_grid[TransformIndex(sym, i, grid_len)] = grid[i];
  }

  return sym_grid;
}

template <typename T, size_t N>
std::array<T, N> ApplyInverse(Symmetry sym, const std::array<T, N>& grid,
                              size_t grid_len) {
  std::array<T, N> inv_grid;
  for (size_t i = 0; i < N; ++i) {
    inv_grid[TransformInv(sym, i, grid_len)] = grid[i];
  }

  return inv_grid;
}

}  // namespace game

#endif
