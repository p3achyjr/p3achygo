#include "cc/nn/nn_board_utils.h"

#include "absl/log/check.h"
#include "cc/constants/constants.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"

namespace nn {

using namespace ::tensorflow;

Tensor NNBoardUtils::GetBlack(const game::Board& board) {
  Tensor t(DataType::DT_FLOAT, {BOARD_LEN, BOARD_LEN});
  auto t_data = t.matrix<float>();
  for (auto i = 0; i < BOARD_LEN; ++i) {
    for (auto j = 0; j < BOARD_LEN; ++j) {
      t_data(i, j) = board.board_[i][j] == BLACK ? 1.0 : 0.0;
    }
  }

  return t;
}

Tensor NNBoardUtils::GetWhite(const game::Board& board) {
  Tensor t(DataType::DT_FLOAT, {BOARD_LEN, BOARD_LEN});
  auto t_data = t.matrix<float>();
  for (auto i = 0; i < BOARD_LEN; ++i) {
    for (auto j = 0; j < BOARD_LEN; ++j) {
      t_data(i, j) = board.board_[i][j] == WHITE ? 1.0 : 0.0;
    }
  }

  return t;
}

Tensor NNBoardUtils::AsOneHot(game::Loc loc) {
  Tensor t(DataType::DT_FLOAT, {BOARD_LEN, BOARD_LEN});
  if (loc.i == -1 && loc.j == -1) {
    return t;
  }

  t.matrix<float>()(loc.i, loc.j) = 1.0;
  return t;
}

std::vector<std::pair<std::string, Tensor>> NNBoardUtils::ConstructNNInput(
    ClientSession& session, const Scope& scope, const game::Board& board,
    int color, const std::vector<game::Loc> moves,
    const std::vector<std::string>& input_names) {
  DCHECK(input_names.size() == 2);
  DCHECK(moves.size() >= 5);

  Tensor input_features(
      DataType::DT_FLOAT,
      {1, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes});
  auto raw = input_features.shaped<float, 4>(
      {1, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes});

  // zero initialize
  int total_size = BOARD_LEN * BOARD_LEN * constants::kNumInputFeaturePlanes;
  auto flat_data = input_features.flat<float>().data();
  std::fill(flat_data, flat_data + total_size, 0);
  for (auto i = 0; i < BOARD_LEN; ++i) {
    for (auto j = 0; j < BOARD_LEN; ++j) {
      if (board.board_[i][j] == color) {
        raw(0, i, j, 0) = 1;
      } else if (board.board_[i][j] == game::OppositeColor(color)) {
        raw(0, i, j, 1) = 1;
      }
    }
  }

  auto offset = 2;
  for (auto i = 0; i < constants::kNumLastMoves; ++i) {
    game::Loc loc = moves[moves.size() - constants::kNumLastMoves + i];
    if (loc == game::kNoopLoc) continue;
    if (loc == game::kPassLoc) continue;

    raw(0, loc.i, loc.j, i + offset) = 1;
  }

  Tensor input_state(DataType::DT_FLOAT, {1, 1});
  input_state.matrix<float>()(0, 0) = board.komi_ / 15.0;

  // @TODO add parameter for whether to cast.
  std::vector<Tensor> cast_output;
  Status status =
      session.Run({ops::Cast(scope, input_features, DataType::DT_HALF),
                   ops::Cast(scope, input_state, DataType::DT_HALF)},
                  &cast_output);

  if (!status.ok()) {
    LOG(ERROR) << "Failed to cast: " << status.error_message();
  }

  return std::vector<std::pair<std::string, Tensor>>{
      std::make_pair(input_names[0], cast_output[0]),
      std::make_pair(input_names[1], cast_output[1]),
  };
}

}  // namespace nn
