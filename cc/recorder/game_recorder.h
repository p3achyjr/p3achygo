#ifndef __RECORDER_GAME_RECORDER_H_
#define __RECORDER_GAME_RECORDER_H_

#include <string>

#include "cc/game/game.h"
#include "cc/recorder/sgf_recorder.h"
#include "cc/recorder/tf_recorder.h"

namespace recorder {

/*
 * Class responsible for recording games to training examples and SGFs.
 *
 * Does not do any actual recording--proxies to `TfRecorder` and `SgfRecorder`.
 */
class GameRecorder {
 public:
  using ImprovedPolicies =
      std::vector<std::array<float, constants::kMaxNumMoves>>;
  virtual ~GameRecorder() = default;

  // Disable Copy and Move.
  GameRecorder(GameRecorder const&) = delete;
  GameRecorder& operator=(GameRecorder const&) = delete;
  GameRecorder(GameRecorder&&) = delete;
  GameRecorder& operator=(GameRecorder&&) = delete;

  virtual void RecordGame(int thread_id, const game::Game& game,
                          const ImprovedPolicies& mcts_pis,
                          const std::vector<uint8_t>& is_move_trainable) = 0;

  static std::unique_ptr<GameRecorder> Create(std::string path, int num_threads,
                                              int flush_interval, int gen,
                                              std::string worker_id);

 protected:
  GameRecorder() = default;
};
}  // namespace recorder

#endif
