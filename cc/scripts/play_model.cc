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
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"

using namespace ::tensorflow;
using ::game::Loc;

ABSL_FLAG(std::string, model_path, "", "Path to model");

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
  while (!board.IsGameOver()) {
    // model move
    nn_input = nn::NNBoardUtils::ConstructNNInput(session, root_scope, board,
                                                  BLACK, moves, kInputNames);
    status = nn_evaluator.Infer(nn_input, kOutputNames, &output_buf);
    if (!status.ok()) {
      LOG(ERROR) << "Eval Failed: " << status.code()
                 << ", msg: " << status.error_message();
      return 1;
    }

    std::vector<Tensor> output;
    status = session.Run(
        {ops::ArgMax(root_scope,
                     ops::Cast(root_scope, output_buf[0], DataType::DT_FLOAT),
                     Input(1)),
         ops::Softmax(root_scope, ops::Cast(root_scope, output_buf[1],
                                            DataType::DT_FLOAT))},
        &output);

    if (!status.ok()) {
      LOG(ERROR) << "Cast Failed: " << status.code()
                 << ", msg: " << status.error_message();
      return 1;
    }

    int64_t move = output[0].flat<int64>()(0);
    if (output[0].flat<int64>()(0) == kPassMove) {
      board.MovePass(BLACK);
    } else {
      board.Move(board.MoveAsLoc(move), BLACK);
    }

    LOG(INFO) << "------- Model Stats -------";
    LOG(INFO) << "Top Move: " << board.MoveAsLoc(move);
    LOG(INFO) << "Win: " << output[1].matrix<float>()(0, 1)
              << " Loss: " << output[1].matrix<float>()(0, 0);
    LOG(INFO) << "-----Board-----\n" << board;

    // human move
    // while (true) {
    //   LOG(INFO) << "Enter Move: ";
    //   std::getline(std::cin, line);
    //   game::Loc human_move = convert_to_move(line);

    //   moves.emplace_back(human_move);
    //   if (board.Move(human_move, WHITE)) {
    //     LOG(INFO) << "-----Board-----\n" << board;
    //     break;
    //   }
    // }
    break;
  }

  LOG(INFO) << output_buf[0].DebugString(362);

  return 0;
}
