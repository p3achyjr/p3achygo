#ifndef RECORDER_TF_RECORDER_H_
#define RECORDER_TF_RECORDER_H_

#include <memory>
#include <string>

#include "cc/game/game.h"

namespace recorder {

/*
 * Class responsible for recording games to TF examples.
 *
 * This class is not thread safe.
 */
class TfRecorder {
 public:
  using ImprovedPolicies =
      std::vector<std::array<float, constants::kMaxMovesPerPosition>>;
  virtual ~TfRecorder() = default;

  // Disable Copy and Move.
  TfRecorder(TfRecorder const&) = delete;
  TfRecorder& operator=(TfRecorder const&) = delete;
  TfRecorder(TfRecorder&&) = delete;
  TfRecorder& operator=(TfRecorder&&) = delete;

  // Recorder Impl.
  virtual void RecordGame(int thread_id, const game::Board& init_board,
                          const game::Game& game,
                          const ImprovedPolicies& mcts_pis,
                          const std::vector<uint8_t>& is_move_trainable,
                          const std::vector<float>& root_qs) = 0;

  // Flushes all pending writes. Not thread safe.
  virtual void Flush() = 0;

  static std::unique_ptr<TfRecorder> Create(std::string path, int num_threads,
                                            int gen, std::string worker_id);

 protected:
  TfRecorder() = default;
};
}  // namespace recorder

#endif
