#ifndef __SELF_PLAY_THREAD_H_
#define __SELF_PLAY_THREAD_H_

#include "cc/nn/nn_interface.h"

void ExecuteSelfPlay(int thread_id, nn::NNInterface* nn_interface,
                     std::string logfile);

#endif  // __CC_SELF_PLAY_THREAD_H_
