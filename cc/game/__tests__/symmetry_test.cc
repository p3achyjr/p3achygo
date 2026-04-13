#include "cc/game/symmetry.h"

#include "cc/core/doctest_include.h"

namespace game {

static constexpr int kGridSize = 25;
static constexpr int kGridLen = 5;

TEST_CASE("BoardTest") {
  // clang-format off
  std::array<int, kGridSize> grid = {0,  1,  2,  3,  4,
                                     5,  6,  7,  8,  9,
                                     10, 11, 12, 13, 14,
                                     15, 16, 17, 18, 19,
                                     20, 21, 22, 23, 24};
  std::array<int, kGridSize> grid_rot90 = {20, 15, 10, 5, 0,
                                           21, 16, 11, 6, 1,
                                           22, 17, 12, 7, 2,
                                           23, 18, 13, 8, 3,
                                           24, 19, 14, 9, 4};
  std::array<int, kGridSize> grid_rot180 = {24, 23, 22, 21, 20,
                                            19, 18, 17, 16, 15,
                                            14, 13, 12, 11, 10,
                                            9,  8,  7,  6,  5,
                                            4,  3,  2,  1,  0};
  std::array<int, kGridSize> grid_rot270 = {4, 9, 14, 19, 24,
                                            3, 8, 13, 18, 23,
                                            2, 7, 12, 17, 22,
                                            1, 6, 11, 16, 21,
                                            0, 5, 10, 15, 20};
  std::array<int, kGridSize> grid_flip = {4,  3,  2,  1,  0,
                                          9,  8,  7,  6,  5,
                                          14, 13, 12, 11, 10,
                                          19, 18, 17, 16, 15,
                                          24, 23, 22, 21, 20};
  std::array<int, kGridSize> grid_flip_rot90 = {24, 19, 14, 9, 4,
                                                23, 18, 13, 8, 3,
                                                22, 17, 12, 7, 2,
                                                21, 16, 11, 6, 1,
                                                20, 15, 10, 5, 0};
  std::array<int, kGridSize> grid_flip_rot180 = {20, 21, 22, 23, 24,
                                                 15, 16, 17, 18, 19,
                                                 10, 11, 12, 13, 14,
                                                 5,  6,  7,  8,  9,
                                                 0,  1,  2,  3,  4};
  std::array<int, kGridSize> grid_flip_rot270 = {0, 5, 10, 15, 20,
                                                 1, 6, 11, 16, 21,
                                                 2, 7, 12, 17, 22,
                                                 3, 8, 13, 18, 23,
                                                 4, 9, 14, 19, 24};
  // clang-format on

  CHECK(grid == ApplySymmetry(Symmetry::kIdentity, grid, kGridLen));
  CHECK(grid_rot90 == ApplySymmetry(Symmetry::kRot90, grid, kGridLen));
  CHECK(grid_rot180 == ApplySymmetry(Symmetry::kRot180, grid, kGridLen));
  CHECK(grid_rot270 == ApplySymmetry(Symmetry::kRot270, grid, kGridLen));
  CHECK(grid_flip == ApplySymmetry(Symmetry::kFlip, grid, kGridLen));
  CHECK(grid_flip_rot90 == ApplySymmetry(Symmetry::kFlipRot90, grid, kGridLen));
  CHECK(grid_flip_rot180 ==
        ApplySymmetry(Symmetry::kFlipRot180, grid, kGridLen));
  CHECK(grid_flip_rot270 ==
        ApplySymmetry(Symmetry::kFlipRot270, grid, kGridLen));

  CHECK(grid == ApplyInverse(Symmetry::kIdentity,
                             ApplySymmetry(Symmetry::kIdentity, grid, kGridLen),
                             kGridLen));
  CHECK(grid == ApplyInverse(Symmetry::kRot90,
                             ApplySymmetry(Symmetry::kRot90, grid, kGridLen),
                             kGridLen));
  CHECK(grid == ApplyInverse(Symmetry::kRot180,
                             ApplySymmetry(Symmetry::kRot180, grid, kGridLen),
                             kGridLen));
  CHECK(grid == ApplyInverse(Symmetry::kRot270,
                             ApplySymmetry(Symmetry::kRot270, grid, kGridLen),
                             kGridLen));
  CHECK(grid == ApplyInverse(Symmetry::kFlip,
                             ApplySymmetry(Symmetry::kFlip, grid, kGridLen),
                             kGridLen));
  CHECK(grid ==
        ApplyInverse(Symmetry::kFlipRot90,
                     ApplySymmetry(Symmetry::kFlipRot90, grid, kGridLen),
                     kGridLen));
  CHECK(grid ==
        ApplyInverse(Symmetry::kFlipRot180,
                     ApplySymmetry(Symmetry::kFlipRot180, grid, kGridLen),
                     kGridLen));
  CHECK(grid ==
        ApplyInverse(Symmetry::kFlipRot270,
                     ApplySymmetry(Symmetry::kFlipRot270, grid, kGridLen),
                     kGridLen));
}

TEST_CASE("LocTest") {
  Loc loc = Loc{2, 3};
  size_t grid_len = 9;
  CHECK_EQ(ApplySymmetry(Symmetry::kIdentity, loc, grid_len), Loc{2, 3});
  CHECK_EQ(ApplySymmetry(Symmetry::kRot90, loc, grid_len), Loc{3, 6});
  CHECK_EQ(ApplySymmetry(Symmetry::kRot180, loc, grid_len), Loc{6, 5});
  CHECK_EQ(ApplySymmetry(Symmetry::kRot270, loc, grid_len), Loc{5, 2});
  CHECK_EQ(ApplySymmetry(Symmetry::kFlip, loc, grid_len), Loc{2, 5});
  CHECK_EQ(ApplySymmetry(Symmetry::kFlipRot90, loc, grid_len), Loc{5, 6});
  CHECK_EQ(ApplySymmetry(Symmetry::kFlipRot180, loc, grid_len), Loc{6, 3});
  CHECK_EQ(ApplySymmetry(Symmetry::kFlipRot270, loc, grid_len), Loc{3, 2});
}

}  // namespace game
