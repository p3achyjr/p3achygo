#ifndef EVAL_EVAL_H_
#define EVAL_EVAL_H_

#include <future>
#include <optional>

#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/zobrist.h"
#include "cc/nn/nn_interface.h"
#include "cc/recorder/game_recorder.h"

enum class Winner : uint8_t {
  kCur = 0,
  kCand = 1,
};

struct EvalConfig {
  const std::string cur_name;
  const std::string cand_name;
  const int cur_n;
  const int cur_k;
  const int cand_n;
  const int cand_k;
  const float cur_noise_scaling;
  const float cand_noise_scaling;
  const bool cur_use_puct;
  const bool cur_use_lcb;
  const float cur_c_puct;
  const bool cand_use_puct;
  const bool cand_use_lcb;
  const float cand_c_puct;
};

struct EvalResult {
  Winner winner;
  int num_moves;
  int roots_shared;
};

void PlayEvalGame(size_t seed, int thread_id, nn::NNInterface* cur_nn,
                  nn::NNInterface* cand_nn, std::string logfile,
                  std::promise<EvalResult> result,
                  recorder::GameRecorder* recorder, EvalConfig config);

#endif  // EVAL_EVAL_H_
