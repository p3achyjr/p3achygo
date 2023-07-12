#ifndef __NN_BOARD_UTILS_H_
#define __NN_BOARD_UTILS_H_

#include <string>
#include <vector>

#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/symmetry.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"

namespace nn {
namespace board_utils {

// Fills `input_features` and `input_state` at batch `batch_id`.
// Assumes tensors are float tensors.
void FillNNInput(int batch_id, int batch_size,
                 ::tensorflow::Tensor& input_features,
                 ::tensorflow::Tensor& input_state, const game::Game& board,
                 game::Color color, game::Symmetry sym);

}  // namespace board_utils
}  // namespace nn

#endif  // __NN_BOARD_UTILS_H_
