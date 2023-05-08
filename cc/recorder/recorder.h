#ifndef __RECORDER_RECORDER_H_
#define __RECORDER_RECORDER_H_

#include <string>

#include "cc/game/game.h"

namespace recorder {

/*
 * Abstract base class for recorders.
 */
class Recorder {
 public:
  virtual ~Recorder() = default;

  // Disable Copy and Move.
  Recorder(Recorder const&) = delete;
  Recorder& operator=(Recorder const&) = delete;
  Recorder(Recorder&&) = delete;
  Recorder& operator=(Recorder&&) = delete;

  virtual void RecordGame(int thread_id, const game::Game& game) = 0;

 protected:
  Recorder() = default;
};
}  // namespace recorder

#endif
