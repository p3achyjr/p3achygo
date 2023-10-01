#ifndef NN_ENGINE_BUF_UTILS_H_
#define NN_ENGINE_BUF_UTILS_H_

#include <array>
#include <iostream>

#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/color.h"

namespace nn {

/*
 * Computes sizes of each subslice.
 */
template <size_t N>
std::array<int, N> Sizes(std::array<int, N> shape) {
  std::array<int, N> sizes;
  int cm_size = 1;
  for (int i = N - 1; i >= 0; --i) {
    sizes[i] = cm_size;
    cm_size *= shape[i];
  }

  return sizes;
}

/*
 * Computes offset into a flat buffer viewed as a tensor of shape `shape`.
 */
template <size_t N>
int Offset(std::array<int, N> shape, std::array<int, N> index) {
  std::array<int, N> sizes = Sizes(shape);
  int off = 0;
  for (int i = 0; i < N; ++i) {
    off += index[i] * sizes[i];
  }

  return off;
}

/*
 * Computes size of slice of tensor with shape `shape` from index `i`.
 */
template <size_t N>
int SliceSize(std::array<int, N> shape, int i) {
  if (i > N) return 0;

  int size = Sizes(shape)[i];
  return size * shape[i]; // `size` is exclusive.
}

/*
 * Loads a host-allocated buffer with contents of `board_data`.
 *
 * `board_data` contains {-1, 0, 1} for {W, E, B}, and `buf` will contain
 * a pair of feature maps for each of {B, W}.
 *
 * Assumes `NHWC` memory layout.
 */
inline void FillPlanePair(
    float* buf, std::array<int, 4> shape, int batch_id, int our_channel,
    int opp_channel,
    std::array<game::Color, constants::kNumBoardLocs> board_data, int board_len,
    game::Color color) {
  for (int i = 0; i < board_len; ++i) {
    for (int j = 0; j < board_len; ++j) {
      auto c = board_data[i * board_len + j];
      if (c == color) {
        std::array<int, 4> index = {batch_id, i, j, our_channel};
        buf[Offset(shape, index)] = 1.0f;
      } else if (c == game::OppositeColor(color)) {
        std::array<int, 4> index = {batch_id, i, j, opp_channel};
        buf[Offset(shape, index)] = 1.0f;
      }
    }
  }

  // if (batch_id == 0 && our_channel == 0) {
  //   std::cerr << "Val Board:\n" << game::ToString(board_data) << "\n";

  //   std::array<game::Color, constants::kNumBoardLocs> loaded_board;
  //   for (int i = 0; i < board_len; ++i) {
  //     for (int j = 0; j < board_len; ++j) {
  //       std::array<int, 4> our_index = {batch_id, i, j, our_channel};
  //       auto ours = buf[Offset(shape, our_index)];
  //       std::array<int, 4> opp_index = {batch_id, i, j, opp_channel};
  //       auto opps = buf[Offset(shape, opp_index)];
  //       if (ours == 1.0f) {
  //         std::cerr << "(" << i << "," << j << "): " << Offset(shape,
  //         our_index)
  //                   << " Flat Index: " << (i * BOARD_LEN + j) << ". ";
  //         loaded_board[i * BOARD_LEN + j] = BLACK;
  //       } else if (opps == 1.0f) {
  //         loaded_board[i * BOARD_LEN + j] = WHITE;
  //       } else {
  //         loaded_board[i * BOARD_LEN + j] = EMPTY;
  //       }
  //     }
  //   }

  //   std::cerr << "Loaded Board:\n" << game::ToString(loaded_board) << "\n";
  // }
}

/*
 * Sets buf[index] = val. `buf` is host-allocated, and interpreted as a tensor
 * of shape `shape`.
 */
template <size_t N>
void SetIndex(float* buf, std::array<int, N> shape, std::array<int, N> index,
              float val) {
  buf[Offset(shape, index)] = val;
}

/*
 * Gets buf[index] = val.
 */
template <size_t N>
float GetIndex(float* buf, std::array<int, N> shape, std::array<int, N> index) {
  return buf[Offset(shape, index)];
}

}  // namespace nn

#endif
