#ifndef EVAL_EVAL_H_
#define EVAL_EVAL_H_

#include <future>
#include <optional>

#include "cc/eval/player_config.h"
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
  eval::PlayerSearchConfig cur;
  eval::PlayerSearchConfig cand;
};

struct EvalResult {
  Winner winner;
  int num_moves;
  int roots_shared;
};

void PlayEvalGame(size_t seed, int game_id, int total_num_workers,
                  nn::NNInterface* cur_nn, nn::NNInterface* cand_nn,
                  std::string logfile, std::promise<EvalResult> result,
                  recorder::GameRecorder* recorder, EvalConfig config);

#endif  // EVAL_EVAL_H_
