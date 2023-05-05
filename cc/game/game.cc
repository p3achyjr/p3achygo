#include "cc/game/game.h"

#include "cc/game/board.h"
#include "cc/game/loc.h"

namespace game {

Game::Game()
    : moves_({kNoopMove, kNoopMove, kNoopMove, kNoopMove, kNoopMove}) {}

bool Game::IsGameOver() const { return board_.IsGameOver(); }

bool Game::IsValidMove(int index, color color) const {
  return IsValidMove(AsLoc(index, board_len()), color);
}

bool Game::IsValidMove(Loc loc, color color) const {
  return board_.IsValidMove(loc, color);
}

bool Game::PlayMove(Loc loc, color color) {
  bool succeeded = board_.PlayMove(loc, color);
  if (succeeded) {
    moves_.emplace_back(game::Move{color, loc});
  }

  return succeeded;
}

bool Game::Pass(color color) {
  bool succeeded = board_.Pass(color);
  if (succeeded) {
    moves_.emplace_back(game::Move{color, kPassLoc});
  }

  return succeeded;
}

Scores Game::GetScores() { return board_.GetScores(); }

}  // namespace game
