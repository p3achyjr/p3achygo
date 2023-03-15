#include <stdlib.h>

#include <iostream>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "cc/core/rand.h"
#include "cc/game/board.h"
#include "cc/nn/nn_board_utils.h"
#include "cc/nn/nn_evaluator.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/public/session.h"

using namespace ::tensorflow;
using ::game::Loc;

ABSL_FLAG(std::string, model_path, "", "Path to model.");

// @TODO add switch for whether to use float or half
const static std::vector<std::string> kInputNames = {
    "infer_mixed_board_state:0",
    "infer_mixed_game_state:0",
};

const static std::vector<std::string> kOutputNames = {
    "StatefulPartitionedCall:0", "StatefulPartitionedCall:1",
    "StatefulPartitionedCall:2", "StatefulPartitionedCall:3",
    "StatefulPartitionedCall:4",
};

static constexpr auto kNumInputFeatures = 7;
static constexpr auto kNumLastMoves = 5;
static constexpr auto kNumScoreTargets = 800;
static constexpr auto kPassMove = 362;
static constexpr auto kMovesToTest = 10;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_model_path) == "") {
    LOG(WARNING) << "No Path Specified";
    return 1;
  }

  Scope root_scope = Scope::NewRootScope();
  ClientSession session(root_scope);

  nn::NNEvaluator nn_evaluator;
  Status status = nn_evaluator.InitFromPath(absl::GetFlag(FLAGS_model_path));
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
    return 1;
  }

  std::vector<Loc> moves = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};
  game::ZobristTable zobrist_table;
  game::Board board(&zobrist_table);

  std::vector<Tensor> output_buf = {
      Tensor(DataType::DT_HALF, {1, BOARD_LEN * BOARD_LEN + 1}),  // logits
      Tensor(DataType::DT_HALF, {1, 2}),                          // win percent
      Tensor(DataType::DT_HALF, {1, BOARD_LEN, BOARD_LEN, 1}),    // ownership
      Tensor(DataType::DT_HALF, {1, kNumScoreTargets}),  // score prediction
      Tensor(DataType::DT_HALF, {1, 1})                  // gamma, just ignore
  };

  std::vector<std::pair<std::string, Tensor>> nn_input;
  auto convert_to_move = [](const std::string& move) {
    return Loc{std::stoi(move.substr(1)), move[0] - 'a'};
  };

  std::string line;
  int color_to_move = BLACK;
  while (!board.IsGameOver()) {
    // model move

    // Get input
    nn_input = nn::NNBoardUtils::ConstructNNInput(
        session, root_scope, board, color_to_move, moves, kInputNames);

    // Feed to session
    status = nn_evaluator.Infer(nn_input, kOutputNames, &output_buf);
    if (!status.ok()) {
      LOG(ERROR) << "Eval Failed: " << status.code()
                 << ", msg: " << status.error_message();
      return 1;
    }

    std::vector<Tensor> output;
    status = session.Run(
        {ops::TopK(root_scope,
                   ops::Cast(root_scope, output_buf[0], DataType::DT_FLOAT),
                   Input(kMovesToTest))
             .indices,
         ops::Softmax(root_scope, ops::Cast(root_scope, output_buf[1],
                                            DataType::DT_FLOAT))},
        &output);

    if (!status.ok()) {
      LOG(ERROR) << "Cast Failed: " << status.code()
                 << ", msg: " << status.error_message();
      return 1;
    }

    int k = 0;
    int move;
    Loc move_loc;
    while (k < kMovesToTest) {
      move = output[0].flat<int32>()(k);
      move_loc = board.MoveAsLoc(move);
      if (board.Move(move_loc, color_to_move)) {
        break;
      }

      ++k;
    }

    LOG(INFO) << "------- Model Stats -------";
    LOG(INFO) << "Top Move: " << move_loc;
    LOG(INFO) << "Win: " << output[1].matrix<float>()(0, 1)
              << " Loss: " << output[1].matrix<float>()(0, 0);
    LOG(INFO) << "-----Board-----\n" << board;

    moves.emplace_back(move_loc);
    color_to_move = game::OppositeColor(color_to_move);
    sleep(1);
  }

  LOG(INFO) << "Game Over: ";
  LOG(INFO) << "  Black Score: " << board.Score(BLACK);
  LOG(INFO) << "  White Score: " << board.Score(WHITE);

  return 0;
}
