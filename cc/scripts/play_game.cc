/*
 * Interactive script to enter moves on the game.
 *
 * Mainly for debugging.
 */
#include <iostream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "cc/constants/constants.h"
#include "cc/game/game.h"
#include "cc/nn/nn_interface.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");

static constexpr game::Loc kInvalidMove = game::Loc{-1, -1};
static constexpr game::Loc kPassMove = game::Loc{19, 0};

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  if (absl::GetFlag(FLAGS_model_path) == "") {
    LOG(WARNING) << "No Model Path Specified.";
    return 1;
  }

  game::Game game;
  std::unique_ptr<nn::NNInterface> nn_interface =
      std::make_unique<nn::NNInterface>(1);
  CHECK_OK(nn_interface->Initialize(absl::GetFlag(FLAGS_model_path)));

  auto convert_to_move = [](const std::string& move) {
    if (move == "pass") {
      return kPassMove;
    }
    if (move.length() < 2 || move.length() > 3) {
      return kInvalidMove;
    }
    auto loc = game::Loc{std::stoi(move.substr(1)), move[0] - 'a'};
    if (loc.i < 0 || loc.i > 19 || loc.j < 0 || loc.j > 18) {
      return kInvalidMove;
    }

    return loc;
  };

  int color_to_move = BLACK;
  while (!game.IsGameOver()) {
    while (true) {
      std::string move_str;
      std::cout << "Input Move:\n";
      std::cin >> move_str;
      game::Loc move = convert_to_move(move_str);

      if (move == kInvalidMove) {
        std::cout << "Invalid Move Encoding.\n";
        continue;
      }

      if (!game.PlayMove(move, color_to_move)) {
        std::cout << "Invalid Move.\n";
        continue;
      }

      break;
    }

    std::cout << game.board() << "\n";
    CHECK_OK(nn_interface->LoadBatch(0, game, color_to_move));
    nn::NNInferResult nn_result = nn_interface->GetInferenceResult(0);

    LOG(INFO) << "Loss: " << nn_result.value_probs[0]
              << "Win: " << nn_result.value_probs[1];
    color_to_move = game::OppositeColor(color_to_move);
  }
}
