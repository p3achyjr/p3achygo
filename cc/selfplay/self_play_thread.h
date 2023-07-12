#ifndef __SELF_PLAY_THREAD_H_
#define __SELF_PLAY_THREAD_H_

#include "cc/nn/nn_interface.h"
#include "cc/recorder/game_recorder.h"
#include "cc/selfplay/go_exploit_buffer.h"

namespace selfplay {

void Run(size_t seed, int thread_id, nn::NNInterface* nn_interface,
         recorder::GameRecorder* game_recorder,
         GoExploitBuffer* go_exploit_buffer, std::string logfile,
         int max_moves);

void SignalStop();
bool IsRunning();

}  // namespace selfplay

#endif  // __SELF_PLAY_THREAD_H_
