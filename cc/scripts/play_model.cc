#include <stdlib.h>

#include <iostream>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/core/rand.h"
#include "cc/game/board.h"
#include "cc/nn/nn_interface.h"
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

  std::vector<Loc> move_history = {
      {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};
  game::Game game;
  std::unique_ptr<nn::NNInterface> nn_interface =
      std::make_unique<nn::NNInterface>(1);
  CHECK_OK(nn_interface->Initialize(absl::GetFlag(FLAGS_model_path)));

  auto convert_to_move = [](const std::string& move) {
    return Loc{std::stoi(move.substr(1)), move[0] - 'a'};
  };

  std::string line;
  int color_to_move = BLACK;
  while (!game.IsGameOver()) {
    CHECK_OK(nn_interface->LoadBatch(0, game, color_to_move));
    nn::NNInferResult nn_result = nn_interface->GetInferenceResult(0);

    std::vector<std::pair<int, float>> moves;
    for (int i = 0; i < constants::kMaxNumMoves; ++i) {
      moves.emplace_back(std::make_pair(i, nn_result.move_logits[i]));
    }

    std::sort(moves.begin(), moves.end(),
              [](auto& x, auto& y) { return x.first > y.first; });

    int k = 0;
    int move;
    Loc move_loc;
    while (k < kMovesToTest) {
      move = moves[k].first;
      move_loc = game::AsLoc(move, game.board_len());
      if (game.PlayMove(move_loc, color_to_move)) {
        break;
      }

      ++k;
    }

    LOG(INFO) << "------- Model Stats -------";
    LOG(INFO) << "Top Move: " << move_loc;
    LOG(INFO) << "Win: " << nn_result.value_probability[0]
              << " Loss: " << nn_result.value_probability[1];
    LOG(INFO) << "-----Board-----\n" << game.board();

    move_history.emplace_back(move_loc);
    color_to_move = game::OppositeColor(color_to_move);
    sleep(1);
  }

  game::Scores scores = game.GetScores();

  LOG(INFO) << "Game Over: ";
  LOG(INFO) << "  Black Score: " << scores.black_score;
  LOG(INFO) << "  White Score: " << scores.white_score;

  return 0;
}
