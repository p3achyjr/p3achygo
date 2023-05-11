#ifndef __SELF_PLAY_THREAD_H_
#define __SELF_PLAY_THREAD_H_

#include "cc/nn/nn_interface.h"
#include "cc/recorder/game_recorder.h"

void ExecuteSelfPlay(int thread_id, nn::NNInterface* nn_interface,
                     recorder::GameRecorder* game_recorder, std::string logfile,
                     int gumbel_n, int gumbel_k, int max_moves);

#endif  // __SELF_PLAY_THREAD_H_
