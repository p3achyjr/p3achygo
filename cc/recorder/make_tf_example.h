#ifndef RECORDER_MAKE_TF_EXAMPLE_H_
#define RECORDER_MAKE_TF_EXAMPLE_H_

#include "cc/constants/constants.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "tensorflow/core/example/example.pb.h"

namespace recorder {

template <typename T, size_t N>
tensorflow::Feature MakeBytesFeature(const std::array<T, N>& data) {
  tensorflow::Feature feature;
  feature.mutable_bytes_list()->add_value(
      reinterpret_cast<const void*>(data.data()), sizeof(T) * N);
  return feature;
}

tensorflow::Example MakeTfExample(
    const std::array<game::Color, BOARD_LEN * BOARD_LEN>& board,
    const std::array<int16_t, constants::kNumLastMoves>& last_moves,
    const std::array<game::Color, BOARD_LEN * BOARD_LEN>& stones_atari,
    const std::array<game::Color, BOARD_LEN * BOARD_LEN>& stones_two_liberties,
    const std::array<game::Color, BOARD_LEN * BOARD_LEN>&
        stones_three_liberties,
    const std::array<float, constants::kMaxNumMoves>& pi_improved,
    int16_t pi_aux, const game::Game::Result result, const float q30,
    const float q100, const float q200, game::Color color, float komi,
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

  // These are Q-values at a horizon of q{n} moves.
  features["q30"].mutable_float_list()->add_value(q30);
  features["q100"].mutable_float_list()->add_value(q100);
  features["q200"].mutable_float_list()->add_value(q200);

  return example;
}
}  // namespace recorder

#endif
