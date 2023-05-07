#ifndef __SELF_PLAY_THREAD_H_
#define __SELF_PLAY_THREAD_H_

#include "cc/nn/nn_interface.h"
#include "cc/recorder/sgf_recorder.h"

void ExecuteSelfPlay(int thread_id, nn::NNInterface* nn_interface,
                     recorder::SgfRecorder* sgf_recorder, std::string logfile);

#endif  // __SELF_PLAY_THREAD_H_
