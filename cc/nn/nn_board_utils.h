#ifndef __NN_BOARD_UTILS_H_
#define __NN_BOARD_UTILS_H_

#include <string>
#include <vector>

#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"

namespace nn {

/*
 * Contains data translations from board to nn input.
 */
class NNBoardUtils final {
 public:
  static ::tensorflow::Tensor GetBlack(const game::Board& board);
  static ::tensorflow::Tensor GetWhite(const game::Board& board);
  static ::tensorflow::Tensor AsOneHot(game::Loc loc);

  // Fills `input_features` and `input_state` at batch `batch_id`.
  // Assumes tensors are float tensors.
  static void FillNNInput(int batch_id, int batch_size,
                          ::tensorflow::Tensor& input_features,
                          ::tensorflow::Tensor& input_state,
                          const game::Board& board, int color,
                          const std::vector<game::Loc> moves);
};

}  // namespace nn

#endif  // __NN_BOARD_UTILS_H_
