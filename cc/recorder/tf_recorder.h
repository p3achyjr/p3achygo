#ifndef __RECORDER_TF_RECORDER_H_
#define __RECORDER_TF_RECORDER_H_

#include <memory>
#include <string>

#include "cc/game/game.h"
#include "cc/recorder/recorder.h"

namespace recorder {

/*
 * Class responsible for recording games to TF examples.
 *
 * This class is not thread safe.
 */
class TfRecorder : public Recorder {
 public:
  ~TfRecorder() = default;

  // Disable Copy and Move.
  TfRecorder(TfRecorder const&) = delete;
  TfRecorder& operator=(TfRecorder const&) = delete;
  TfRecorder(TfRecorder&&) = delete;
  TfRecorder& operator=(TfRecorder&&) = delete;

  // Recorder Impl.
  void RecordGame(int thread_id, const game::Game& game) override = 0;

  // Flushes all pending writes. Not thread safe.
  virtual void Flush() = 0;

  static std::unique_ptr<TfRecorder> Create(std::string path, int num_threads);

 protected:
  TfRecorder() = default;
};
}  // namespace recorder

#endif
