#ifndef RECORDER_MAKE_TF_EXAMPLE_H_
#define RECORDER_MAKE_TF_EXAMPLE_H_

#include "cc/constants/constants.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "example.pb.h"

namespace recorder {

template <typename T, size_t N>
tensorflow::Feature MakeBytesFeature(const std::array<T, N>& data) {
  tensorflow::Feature feature;
  feature.mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(data.data()), sizeof(T) * N);
  return feature;
}

inline tensorflow::Example MakeTfExample(
    const std::array<game::Color, BOARD_LEN * BOARD_LEN>& board,
    const std::array<int16_t, constants::kNumLastMoves>& last_moves,
    const std::array<game::Color, BOARD_LEN * BOARD_LEN>& stones_atari,
    const std::array<game::Color, BOARD_LEN * BOARD_LEN>& stones_two_liberties,
    const std::array<game::Color, BOARD_LEN * BOARD_LEN>&
        stones_three_liberties,
    const std::array<game::Color, BOARD_LEN * BOARD_LEN>& stones_in_ladder,
    const std::array<float, constants::kMaxMovesPerPosition>& pi_improved,
    int16_t pi_aux, const game::Game::Result result, const float q6,
    const float q16, const float q50, const float q6_score,
    const float q16_score, const float q50_score, game::Color color, float komi,
    uint8_t bsize) {
  tensorflow::Example example;
  auto& features = *example.mutable_features()->mutable_feature();

  features["bsize"].mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(&bsize), sizeof(uint8_t));
  features["board"] = MakeBytesFeature(board);
  features["last_moves"] = MakeBytesFeature(last_moves);
  features["stones_atari"] = MakeBytesFeature(stones_atari);
  features["stones_two_liberties"] = MakeBytesFeature(stones_two_liberties);
  features["stones_three_liberties"] = MakeBytesFeature(stones_three_liberties);
  features["stones_in_ladder"] = MakeBytesFeature(stones_in_ladder);
  features["color"].mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(&color), sizeof(game::Color));
  features["komi"].mutable_float_list()->add_value(komi);
  features["own"] = MakeBytesFeature(result.ownership);

  // This is completed Q-values.
  features["pi"] = MakeBytesFeature(pi_improved);

  // Moves past the end of the game are just encoded as pass.
  features["pi_aux"].mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(&pi_aux), sizeof(int16_t));

  float margin = color == BLACK ? result.bscore - result.wscore
                                : result.wscore - result.bscore;
  features["score_margin"].mutable_float_list()->add_value(margin);

  // These are exp-weighted Q-values at a horizon of q{n} moves.
  features["q6"].mutable_float_list()->add_value(q6);
  features["q16"].mutable_float_list()->add_value(q16);
  features["q50"].mutable_float_list()->add_value(q50);
  features["q6_score"].mutable_float_list()->add_value(q6_score);
  features["q16_score"].mutable_float_list()->add_value(q16_score);
  features["q50_score"].mutable_float_list()->add_value(q50_score);

  return example;
}
}  // namespace recorder

#endif
