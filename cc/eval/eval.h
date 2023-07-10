#ifndef __EVAL_EVAL_H_
#define __EVAL_EVAL_H_

#include <future>

#include "cc/game/color.h"
#include "cc/nn/nn_interface.h"

enum class Winner : uint8_t {
  kCur = 0,
  kCand = 1,
};

void PlayEvalGame(size_t seed, int thread_id, nn::NNInterface* cur_nn,
                  nn::NNInterface* cand_nn, std::string logfile,
                  std::promise<Winner> winner);

#endif  // __EVAL_EVAL_H_
