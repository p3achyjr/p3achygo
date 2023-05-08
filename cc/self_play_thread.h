#ifndef __SELF_PLAY_THREAD_H_
#define __SELF_PLAY_THREAD_H_

#include "cc/nn/nn_interface.h"
#include "cc/recorder/sgf_recorder.h"

void ExecuteSelfPlay(int thread_id, nn::NNInterface* nn_interface,
                     recorder::SgfRecorder* sgf_recorder, std::string logfile,
                     int gumbel_n, int gumbel_k, int max_num_moves);

#endif  // __SELF_PLAY_THREAD_H_
