#ifndef SELF_PLAY_THREAD_H_
#define SELF_PLAY_THREAD_H_

#include "cc/nn/nn_interface.h"
#include "cc/recorder/game_recorder.h"
#include "cc/selfplay/reuse_buffer.h"

namespace selfplay {

struct GumbelParams {
  int n;
  int k;
};

struct SPConfig {
  const int max_moves;
  const GumbelParams selected_params;
  const GumbelParams default_params;
  // Base multiplier for selection probability. If 0, sel_mult is disabled
  // (falls back to 1.0). Otherwise, the full signal-based multiplier is
  // computed and scaled by this value.
  const float sel_mult_base = 0.0f;
};

void Run(size_t seed, int thread_id, nn::NNInterface* nn_interface,
         recorder::GameRecorder* game_recorder, ReuseBuffer* reuse_buffer,
         std::string logfile, SPConfig config);

void SignalStop();
bool IsRunning();

}  // namespace selfplay

#endif  // __SELF_PLAY_THREAD_H_
