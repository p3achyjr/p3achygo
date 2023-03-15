/*
 * Interactive script to enter moves on the board.
 *
 * Mainly for debugging.
 */
#include <iostream>

#include "cc/game/board.h"
#include "cc/game/constants.h"
#include "cc/game/zobrist_hash.h"

static constexpr game::Loc kInvalidMove = game::Loc{-1, -1};
static constexpr game::Loc kPassMove = game::Loc{19, 0};

int main(int argc, char** argv) {
  game::ZobristTable zobrist_table;
  game::Board board(&zobrist_table);

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
  while (!board.IsGameOver()) {
    while (true) {
      std::string move_str;
      std::cout << "Input Move:\n";
      std::cin >> move_str;
      game::Loc move = convert_to_move(move_str);

      if (move == kInvalidMove) {
        std::cout << "Invalid Move Encoding.\n";
        continue;
      }

      if (!board.Move(move, color_to_move)) {
        std::cout << "Invalid Move.\n";
        continue;
      }

      break;
    }

    std::cout << board << "\n";
    color_to_move = game::OppositeColor(color_to_move);
  }
}