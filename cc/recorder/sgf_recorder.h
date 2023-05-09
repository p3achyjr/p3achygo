#ifndef __RECORDER_SGF_RECORDER_H_
#define __RECORDER_SGF_RECORDER_H_

#include <memory>
#include <string>

#include "absl/container/inlined_vector.h"
#include "cc/game/game.h"
#include "cc/recorder/recorder.h"

namespace recorder {

/*
 * Class responsible for recording games to SGFs.
 */
class SgfRecorder : public Recorder {
 public:
  ~SgfRecorder() = default;

  // Disable Copy and Move.
  SgfRecorder(SgfRecorder const&) = delete;
  SgfRecorder& operator=(SgfRecorder const&) = delete;
  SgfRecorder(SgfRecorder&&) = delete;
  SgfRecorder& operator=(SgfRecorder&&) = delete;

  // Recorder Impl.
  void RecordGame(int thread_id, const game::Game& game) override = 0;

  std::unique_ptr<SgfRecorder> static Create(std::string path, int num_threads,
                                             int flush_interval);

 protected:
  SgfRecorder() = default;
};
}  // namespace recorder

#endif
