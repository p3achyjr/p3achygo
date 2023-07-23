#ifndef SELF_PLAY_THREAD_H_
#define SELF_PLAY_THREAD_H_

#include "cc/nn/nn_interface.h"
#include "cc/recorder/game_recorder.h"
#include "cc/selfplay/go_exploit_buffer.h"

namespace selfplay {

struct GumbelParams {
  int n;
  int k;
};

struct SPConfig {
  const int max_moves;
  const GumbelParams selected_params;
  const GumbelParams default_params;
};

void Run(size_t seed, int thread_id, nn::NNInterface* nn_interface,
         recorder::GameRecorder* game_recorder,
         GoExploitBuffer* go_exploit_buffer, std::string logfile,
         SPConfig config);

void SignalStop();
bool IsRunning();

}  // namespace selfplay

#endif  // __SELF_PLAY_THREAD_H_
