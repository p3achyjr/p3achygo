#include "cc/nn/nn_board_utils.h"

#include "absl/log/check.h"
#include "cc/constants/constants.h"
#include "cc/nn/create_tensor_shape.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"

namespace nn {
namespace board_utils {
namespace {

using namespace ::nn;
using namespace ::tensorflow;
using namespace ::game;

}  // namespace

void FillNNInput(int batch_id, int batch_size, Tensor& input_planes,
                 Tensor& input_global_state, const Game& game, Color color,
                 Symmetry sym) {
  DCHECK(game.moves().size() >= 5);

  auto raw = input_planes.shaped<float, 4>(
      {batch_size, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes});
  auto fill_plane_pair = [batch_id, color, &raw](Board::BoardData grid,
                                                 int our_index, int opp_index) {
    for (auto i = 0; i < BOARD_LEN; ++i) {
      for (auto j = 0; j < BOARD_LEN; ++j) {
        if (grid[i * BOARD_LEN + j] == color) {
          raw(batch_id, i, j, our_index) = 1;
        } else if (grid[i * BOARD_LEN + j] == OppositeColor(color)) {
          raw(batch_id, i, j, opp_index) = 1;
        }
      }
    }
  };

  const auto& board = game.board();
  const auto& moves = game.moves();
  auto sym_grid = ApplySymmetry(sym, board.position(), BOARD_LEN);

  // fill board state
  fill_plane_pair(sym_grid, 0, 1);

  // fill moves
  auto mv_offset = 2;
  for (auto i = 0; i < constants::kNumLastMoves; ++i) {
    Loc loc = moves[moves.size() - constants::kNumLastMoves + i].loc;
    if (loc == kNoopLoc) continue;
    if (loc == kPassLoc) continue;

    Loc sym_loc = ApplySymmetry(sym, loc, BOARD_LEN);
    raw(batch_id, sym_loc.i, sym_loc.j, i + mv_offset) = 1;
  }

  // atari
  auto stones_in_atari =
      ApplySymmetry(sym, board.GetStonesInAtari(), BOARD_LEN);
  fill_plane_pair(stones_in_atari, 7, 8);

  // two liberties
  auto stones_two_liberties =
      ApplySymmetry(sym, board.GetStonesWithLiberties(2), BOARD_LEN);
  fill_plane_pair(stones_two_liberties, 9, 10);

  // three liberties
  auto stones_three_liberties =
      ApplySymmetry(sym, board.GetStonesWithLiberties(3), BOARD_LEN);
  fill_plane_pair(stones_three_liberties, 11, 12);

  // fill global state.
  input_global_state.matrix<float>()(batch_id, 0) = color == BLACK ? 1 : 0;
  input_global_state.matrix<float>()(batch_id, 1) = color == WHITE ? 1 : 0;
  for (auto i = 0; i < constants::kNumLastMoves; ++i) {
    Loc loc = moves[moves.size() - constants::kNumLastMoves + i].loc;
    if (loc == kPassLoc) {
      input_global_state.matrix<float>()(batch_id, i + mv_offset) = 1;
    }
  }
}

}  // namespace board_utils
}  // namespace nn
