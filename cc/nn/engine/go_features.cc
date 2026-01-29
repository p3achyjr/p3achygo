#include "cc/nn/engine/go_features.h"

#include <iostream>

#include "cc/game/board.h"
#include "cc/nn/engine/buf_utils.h"

namespace nn {

void LoadPlanes(float* planes_buf, std::array<int, 4> planes_shape,
                const GoFeatures& go_features, int batch_id, int version) {
  FillPlanePair(planes_buf, planes_shape, batch_id, 0, 1, go_features.board,
                go_features.bsize, go_features.color);
  FillPlanePair(planes_buf, planes_shape, batch_id, 7, 8,
                go_features.stones_atari, go_features.bsize, go_features.color);
  FillPlanePair(planes_buf, planes_shape, batch_id, 9, 10,
                go_features.stones_two_liberties, go_features.bsize,
                go_features.color);
  FillPlanePair(planes_buf, planes_shape, batch_id, 11, 12,
                go_features.stones_three_liberties, go_features.bsize,
                go_features.color);
  if (version >= 1) {
    FillPlanePair(planes_buf, planes_shape, batch_id, 13, 14,
                  go_features.stones_laddered, go_features.bsize,
                  go_features.color);
  }
  for (int i = 0; i < constants::kNumLastMoves; ++i) {
    game::Loc last_move = go_features.last_moves[i];
    if (last_move == game::kNoopLoc || last_move == game::kPassLoc) {
      continue;
    }

    int channel = i + 2;
    std::array<int, 4> index = {batch_id, last_move.i, last_move.j, channel};
    SetIndex(planes_buf, planes_shape, index, 1.0f);
  }
}

void LoadFeatures(float* feats_buf, std::array<int, 2> feats_shape,
                  const GoFeatures& go_features, int batch_id, int version) {
  std::array<int, 2> color_index = {batch_id,
                                    go_features.color == BLACK ? 0 : 1};
  SetIndex(feats_buf, feats_shape, color_index, 1.0f);
  feats_buf[go_features.color == BLACK ? 0 : 1] = 1.0f;
  for (int i = 0; i < constants::kNumLastMoves; ++i) {
    game::Loc last_move = go_features.last_moves[i];
    if (last_move != game::kPassLoc) {
      continue;
    }

    std::array<int, 2> pass_index = {batch_id, i + 2};
    SetIndex(feats_buf, feats_shape, pass_index, 1.0f);
  }

  if (version >= 1) {
    std::array<int, 2> komi_index = {batch_id, 7};
    SetIndex(feats_buf, feats_shape, komi_index, go_features.komi / 15.0f);
  }
}

void LoadGoFeatures(float* planes_buf, float* feats_buf,
                    std::array<int, 4> planes_shape,
                    std::array<int, 2> feats_shape,
                    const GoFeatures& go_features, int batch_id, int version) {
  LoadPlanes(planes_buf, planes_shape, go_features, batch_id, version);
  LoadFeatures(feats_buf, feats_shape, go_features, batch_id, version);
}
}  // namespace nn
