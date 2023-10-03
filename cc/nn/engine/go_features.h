#ifndef NN_ENGINE_GO_FEATURES_H_
#define NN_ENGINE_GO_FEATURES_H_

#include <array>

#include "cc/constants/constants.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"

namespace nn {

struct GoFeatures {
  int bsize;
  game::Color color;
  std::array<game::Color, constants::kNumBoardLocs> board;
  std::array<game::Loc, constants::kNumLastMoves> last_moves;
  std::array<game::Color, constants::kNumBoardLocs> stones_atari;
  std::array<game::Color, constants::kNumBoardLocs> stones_two_liberties;
  std::array<game::Color, constants::kNumBoardLocs> stones_three_liberties;
};

struct GoLabels {
  std::array<float, constants::kMaxMovesPerPosition> policy;
  float score_margin;
  bool did_win;
};

void LoadPlanes(float* planes_buf, std::array<int, 4> planes_shape,
                const GoFeatures& go_features, int batch_id);
void LoadFeatures(float* feats_buf, std::array<int, 2> feats_shape,
                  const GoFeatures& go_features, int batch_id);
void LoadGoFeatures(float* planes_buf, float* feats_buf,
                    std::array<int, 4> planes_shape,
                    std::array<int, 2> feats_shape,
                    const GoFeatures& go_features, int batch_id);

}  // namespace nn

#endif
