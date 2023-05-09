#ifndef __RECORDER_GAME_RECORDER_H_
#define __RECORDER_GAME_RECORDER_H_

#include <string>

#include "cc/game/game.h"
#include "cc/recorder/recorder.h"
#include "cc/recorder/sgf_recorder.h"
#include "cc/recorder/tf_recorder.h"

namespace recorder {

/*
 * Class responsible for recording games to training examples and SGFs.
 *
 * Does not do any actual recording--proxies to `TfRecorder` and `SgfRecorder`.
 */
class GameRecorder : public Recorder {
 public:
  ~GameRecorder() = default;

  // Disable Copy and Move.
  GameRecorder(GameRecorder const&) = delete;
  GameRecorder& operator=(GameRecorder const&) = delete;
  GameRecorder(GameRecorder&&) = delete;
  GameRecorder& operator=(GameRecorder&&) = delete;

  void RecordGame(int thread_id, const game::Game& game) override = 0;

  static std::unique_ptr<GameRecorder> Create(std::string path, int num_threads,
                                              int flush_interval);

 protected:
  GameRecorder() = default;
};
}  // namespace recorder

#endif
