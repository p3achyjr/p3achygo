#include "cc/nn/nn_board_utils.h"

#include "absl/log/check.h"
#include "cc/constants/constants.h"
#include "cc/nn/create_tensor_shape.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"

namespace nn {
namespace board_utils {

using namespace ::nn;
using namespace ::tensorflow;
using namespace ::game;

Tensor GetBlack(const Board& board) {
  Tensor t(DataType::DT_FLOAT, CreateTensorShape({BOARD_LEN, BOARD_LEN}));
  auto t_data = t.matrix<float>();
  for (auto i = 0; i < BOARD_LEN; ++i) {
    for (auto j = 0; j < BOARD_LEN; ++j) {
      t_data(i, j) = board.at(i, j) == BLACK ? 1.0 : 0.0;
    }
  }

  return t;
}

Tensor GetWhite(const Board& board) {
  Tensor t(DataType::DT_FLOAT, CreateTensorShape({BOARD_LEN, BOARD_LEN}));
  auto t_data = t.matrix<float>();
  for (auto i = 0; i < BOARD_LEN; ++i) {
    for (auto j = 0; j < BOARD_LEN; ++j) {
      t_data(i, j) = board.at(i, j) == WHITE ? 1.0 : 0.0;
    }
  }

  return t;
}

Tensor AsOneHot(Loc loc) {
  Tensor t(DataType::DT_FLOAT, CreateTensorShape({BOARD_LEN, BOARD_LEN}));
  if (loc.i == -1 && loc.j == -1) {
    return t;
  }

  t.matrix<float>()(loc.i, loc.j) = 1.0;
  return t;
}

/* static */ void FillNNInput(int batch_id, int batch_size,
                              Tensor& input_features, Tensor& input_state,
                              const Game& game, Color color, Symmetry sym) {
  DCHECK(game.moves().size() >= 5);

  const auto& board = game.board();
  const auto& moves = game.moves();
  auto sym_grid = ApplySymmetry(sym, board.position(), game.board_len());

  auto raw = input_features.shaped<float, 4>(
      {batch_size, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes});

  // fill board state
  for (auto i = 0; i < BOARD_LEN; ++i) {
    for (auto j = 0; j < BOARD_LEN; ++j) {
      if (sym_grid[i * BOARD_LEN + j] == color) {
        raw(batch_id, i, j, 0) = 1;
      } else if (sym_grid[i * BOARD_LEN + j] == OppositeColor(color)) {
        raw(batch_id, i, j, 1) = 1;
      }
    }
  }

  // fill moves
  auto offset = 2;
  for (auto i = 0; i < constants::kNumLastMoves; ++i) {
    Loc loc = moves[moves.size() - constants::kNumLastMoves + i].loc;
    if (loc == kNoopLoc) continue;
    if (loc == kPassLoc) continue;

    Loc sym_loc = ApplySymmetry(sym, loc, game.board_len());
    raw(batch_id, sym_loc.i, sym_loc.j, i + offset) = 1;
  }

  // fill game state (just komi for now)
  input_state.matrix<float>()(batch_id, 0) =
      color == BLACK ? 0.0 : board.komi();
}

}  // namespace board_utils
}  // namespace nn
