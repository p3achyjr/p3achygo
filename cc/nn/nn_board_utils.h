#ifndef __NN_BOARD_UTILS_H_
#define __NN_BOARD_UTILS_H_

#include <string>
#include <vector>

#include "cc/game/board.h"
#include "cc/game/constants.h"
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

  static std::vector<std::pair<std::string, ::tensorflow::Tensor>>
  ConstructNNInput(::tensorflow::ClientSession& session,
                   const ::tensorflow::Scope& scope, const game::Board& board,
                   int color, const std::vector<game::Loc> moves,
                   const std::vector<std::string>& input_names);
};

}  // namespace nn

#endif  // __NN_BOARD_UTILS_H_
