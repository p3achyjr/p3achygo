#include "cc/game/symmetry.h"

#include <array>
#include <cstdint>
#include <iostream>

#include "cc/core/rand.h"

namespace game {

inline int Inv(int i, size_t grid_len) { return grid_len - i - 1; }

inline int Flip(int index, size_t grid_len) {
  int i = index / grid_len;
  int j = index % grid_len;

  return i * grid_len + Inv(j, grid_len);
}

int Rotate(int index, size_t grid_len, Rotation rot) {
  int i = index / grid_len;
  int j = index % grid_len;

  switch (rot) {
    case Rotation::k90:
      return j * grid_len + Inv(i, grid_len);
    case Rotation::k180:
      return Inv(i, grid_len) * grid_len + Inv(j, grid_len);
    case Rotation::k270:
      return Inv(j, grid_len) * grid_len + i;
  }

  return index;
}

int TransformIndex(Symmetry sym, int index, size_t grid_len) {
  switch (sym) {
    case Symmetry::kIdentity:
      return index;
    case Symmetry::kRot90:
      return Rotate(index, grid_len, Rotation::k90);
    case Symmetry::kRot180:
      return Rotate(index, grid_len, Rotation::k180);
    case Symmetry::kRot270:
      return Rotate(index, grid_len, Rotation::k270);
    case Symmetry::kFlip:
      return Flip(index, grid_len);
    case Symmetry::kFlipRot90:
      return Rotate(Flip(index, grid_len), grid_len, Rotation::k90);
    case Symmetry::kFlipRot180:
      return Rotate(Flip(index, grid_len), grid_len, Rotation::k180);
    case Symmetry::kFlipRot270:
      return Rotate(Flip(index, grid_len), grid_len, Rotation::k270);
  }

  return index;
}

int TransformInv(Symmetry sym, int index, size_t grid_len) {
  switch (sym) {
    case Symmetry::kIdentity:
      return index;
    case Symmetry::kRot90:
      return Rotate(index, grid_len, Rotation::k270);
    case Symmetry::kRot180:
      return Rotate(index, grid_len, Rotation::k180);
    case Symmetry::kRot270:
      return Rotate(index, grid_len, Rotation::k90);
    case Symmetry::kFlip:
      return Flip(index, grid_len);
    case Symmetry::kFlipRot90:
      return Flip(Rotate(index, grid_len, Rotation::k270), grid_len);
    case Symmetry::kFlipRot180:
      return Flip(Rotate(index, grid_len, Rotation::k180), grid_len);
    case Symmetry::kFlipRot270:
      return Flip(Rotate(index, grid_len, Rotation::k90), grid_len);
  }

  return index;
}

}  // namespace game
