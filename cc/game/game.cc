#include "cc/game/game.h"

#include "cc/game/board.h"
#include "cc/game/loc.h"

namespace game {

Game::Game()
    : moves_({kNoopMove, kNoopMove, kNoopMove, kNoopMove, kNoopMove}),
      result_(Result{ResultTag::kUnknown, 0}) {}

const Board& Game::board() const { return board_; }

int Game::board_len() const { return board_.length(); }

const absl::InlinedVector<Move, constants::kMaxGameLen>& Game::moves() const {
  return moves_;
}

int Game::move_num() const { return moves_.size() - kMoveOffset; }

Move Game::move(int move_num) const { return moves_[move_num + kMoveOffset]; }

Game::Result Game::result() const { return result_; }

bool Game::has_result() const { return result_.tag != ResultTag::kUnknown; }

float Game::komi() const { return board_.komi(); }

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

void Game::WriteResult() {
  Scores scores = GetScores();
  if (scores.black_score > scores.white_score) {
    result_ = Result{ResultTag::kBlackWinByScore,
                     scores.black_score - scores.white_score};
  } else if (scores.white_score > scores.black_score) {
    result_ = Result{ResultTag::kWhiteWinByScore,
                     scores.white_score - scores.black_score};
  } else {
    result_ = Result{ResultTag::kUnknown, 0};
  }
}

}  // namespace game
