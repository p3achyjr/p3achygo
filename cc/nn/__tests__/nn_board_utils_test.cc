#include "cc/nn/nn_board_utils.h"

#include "cc/core/doctest_include.h"
#include "cc/game/game.h"
#include "cc/game/symmetry.h"
#include "cc/nn/create_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

#define CHECK_GRID_EQUAL(grid, raw, index)                            \
  do {                                                                \
    for (int i = 0; i < BOARD_LEN; ++i) {                             \
      for (int j = 0; j < BOARD_LEN; ++j) {                           \
        CHECK_EQ((raw)(0, i, j, (index)), (grid)[i * BOARD_LEN + j]); \
      }                                                               \
    }                                                                 \
  } while (0);

#define CHECK_MOVE_GRID(loc, raw, index)          \
  do {                                            \
    for (int i = 0; i < BOARD_LEN; ++i) {         \
      for (int j = 0; j < BOARD_LEN; ++j) {       \
        if (i == (loc).i && j == (loc).j) {       \
          CHECK_EQ((raw)(0, i, j, (index)), 1.0); \
        } else {                                  \
          CHECK_EQ((raw)(0, i, j, (index)), 0.0); \
        }                                         \
      }                                           \
    }                                             \
  } while (0);

namespace nn {
namespace board_utils {

using namespace ::game;
using namespace ::tensorflow;

Board::BoardData GetColor(const Board::BoardData& grid, Color color) {
  Board::BoardData grid_of_color = {};
  for (int i = 0; i < BOARD_LEN; ++i) {
    for (int j = 0; j < BOARD_LEN; ++j) {
      if (grid[i * BOARD_LEN + j] == color) {
        grid_of_color[i * BOARD_LEN + j] = 1;
      }
    }
  }

  return grid_of_color;
}

// . . . . . . . .
// . . . . x o x .
// . . x x o x . .
// . . o o o . . .
// . . . . . . . .
TEST_CASE("TestFillNNInputBlack") {
  Game game;
  game.PlayMove(Loc{3, 3}, BLACK);
  game.PlayMove(Loc{2, 2}, WHITE);
  game.PlayMove(Loc{3, 2}, BLACK);
  game.PlayMove(Loc{2, 3}, WHITE);
  game.PlayMove(Loc{2, 4}, BLACK);
  game.PlayMove(Loc{1, 4}, WHITE);
  game.PlayMove(Loc{1, 5}, BLACK);
  game.PlayMove(Loc{2, 5}, WHITE);
  game.PlayMove(Loc{3, 4}, BLACK);
  game.PlayMove(Loc{1, 6}, WHITE);

  Tensor input_planes =
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape(
                 {1, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes}));
  Tensor input_global_state =
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({1, constants::kNumInputFeatureScalars}));
  input_planes.flat<float>().setZero();
  input_global_state.flat<float>().setZero();

  auto raw = input_planes.shaped<float, 4>(
      {1, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes});

  FillNNInput(0, 1, input_planes, input_global_state, game, BLACK,
              Symmetry::kIdentity);

  // Verify Grids (Board Position, Liberties)
  CHECK_GRID_EQUAL(GetColor(game.board().position(), BLACK), raw, 0);
  CHECK_GRID_EQUAL(GetColor(game.board().position(), WHITE), raw, 1);
  CHECK_GRID_EQUAL(GetColor(game.board().GetStonesInAtari(), BLACK), raw, 7);
  CHECK_GRID_EQUAL(GetColor(game.board().GetStonesInAtari(), WHITE), raw, 8);
  CHECK_GRID_EQUAL(GetColor(game.board().GetStonesWithLiberties(2), BLACK), raw,
                   9);
  CHECK_GRID_EQUAL(GetColor(game.board().GetStonesWithLiberties(2), WHITE), raw,
                   10);
  CHECK_GRID_EQUAL(GetColor(game.board().GetStonesWithLiberties(3), BLACK), raw,
                   11);
  CHECK_GRID_EQUAL(GetColor(game.board().GetStonesWithLiberties(3), WHITE), raw,
                   12);

  // Verify Moves
  CHECK_MOVE_GRID((Loc{1, 4}), raw, 2);
  CHECK_MOVE_GRID((Loc{1, 5}), raw, 3);
  CHECK_MOVE_GRID((Loc{2, 5}), raw, 4);
  CHECK_MOVE_GRID((Loc{3, 4}), raw, 5);
  CHECK_MOVE_GRID((Loc{1, 6}), raw, 6);

  // Global Features (Colors, Pass)
  CHECK_EQ(input_global_state.matrix<float>()(0, 0), 1.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 1), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 2), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 3), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 4), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 5), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 6), 0.0);
}

// . . . . . . . .
// . . . . x o x .
// . . x x o x . .
// . . o o o . . .
// . . . . . . . .
TEST_CASE("TestFillNNInputSymWhitePass") {
  Game game;
  game.PlayMove(Loc{3, 3}, BLACK);
  game.PlayMove(Loc{2, 2}, WHITE);
  game.PlayMove(Loc{3, 2}, BLACK);
  game.PlayMove(Loc{2, 3}, WHITE);
  game.PlayMove(Loc{2, 4}, BLACK);
  game.PlayMove(Loc{1, 4}, WHITE);
  game.PlayMove(Loc{1, 5}, BLACK);
  game.PlayMove(Loc{2, 5}, WHITE);
  game.PlayMove(Loc{3, 4}, BLACK);
  game.PlayMove(Loc{1, 6}, WHITE);
  game.PlayMove(kPassLoc, BLACK);

  Symmetry sym = Symmetry::kFlipRot90;
  Tensor input_planes =
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape(
                 {1, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes}));
  Tensor input_global_state =
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({1, constants::kNumInputFeatureScalars}));
  input_planes.flat<float>().setZero();
  input_global_state.flat<float>().setZero();

  auto raw = input_planes.shaped<float, 4>(
      {1, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes});

  FillNNInput(0, 1, input_planes, input_global_state, game, WHITE, sym);

  // Check Board State and Liberties.
  CHECK_GRID_EQUAL(
      GetColor(ApplySymmetry(sym, game.board().position(), BOARD_LEN), WHITE),
      raw, 0);
  CHECK_GRID_EQUAL(
      GetColor(ApplySymmetry(sym, game.board().position(), BOARD_LEN), BLACK),
      raw, 1);
  CHECK_GRID_EQUAL(
      GetColor(ApplySymmetry(sym, game.board().GetStonesInAtari(), BOARD_LEN),
               WHITE),
      raw, 7);
  CHECK_GRID_EQUAL(
      GetColor(ApplySymmetry(sym, game.board().GetStonesInAtari(), BOARD_LEN),
               BLACK),
      raw, 8);
  CHECK_GRID_EQUAL(
      GetColor(
          ApplySymmetry(sym, game.board().GetStonesWithLiberties(2), BOARD_LEN),
          WHITE),
      raw, 9);
  CHECK_GRID_EQUAL(
      GetColor(
          ApplySymmetry(sym, game.board().GetStonesWithLiberties(2), BOARD_LEN),
          BLACK),
      raw, 10);
  CHECK_GRID_EQUAL(
      GetColor(
          ApplySymmetry(sym, game.board().GetStonesWithLiberties(3), BOARD_LEN),
          WHITE),
      raw, 11);
  CHECK_GRID_EQUAL(
      GetColor(
          ApplySymmetry(sym, game.board().GetStonesWithLiberties(3), BOARD_LEN),
          BLACK),
      raw, 12);

  // Verify Moves.
  CHECK_MOVE_GRID(ApplySymmetry(sym, Loc{1, 5}, BOARD_LEN), raw, 2);
  CHECK_MOVE_GRID(ApplySymmetry(sym, Loc{2, 5}, BOARD_LEN), raw, 3);
  CHECK_MOVE_GRID(ApplySymmetry(sym, Loc{3, 4}, BOARD_LEN), raw, 4);
  CHECK_MOVE_GRID(ApplySymmetry(sym, Loc{1, 6}, BOARD_LEN), raw, 5);
  CHECK_GRID_EQUAL(Board::BoardData{}, raw, 6);

  CHECK_EQ(input_global_state.matrix<float>()(0, 0), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 1), 1.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 2), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 3), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 4), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 5), 0.0);
  CHECK_EQ(input_global_state.matrix<float>()(0, 6), 1.0);
}

}  // namespace board_utils
}  // namespace nn
