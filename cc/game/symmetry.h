#ifndef __CC_GAME_SYMMETRY_H_
#define __CC_GAME_SYMMETRY_H_

#include <array>
#include <cstdint>

#include "cc/core/rand.h"

namespace game {
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

Symmetry GetRandomSymmetry(core::PRng& prng);

template <typename T, int N>
std::array<T, N> ApplySymmetry(const std::array<T, N>) {}

}  // namespace game

#endif
