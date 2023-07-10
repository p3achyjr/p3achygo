#include "cc/game/game.h"

#include "cc/game/board.h"
#include "cc/game/loc.h"

namespace game {

Game::Game(const Board& board,
           const absl::InlinedVector<Move, constants::kMaxGameLen> last_moves)
    : board_(board), moves_(last_moves), result_(Result{EMPTY, 0, 0, {}}) {}

Game::Game()
    : moves_({kNoopMove, kNoopMove, kNoopMove, kNoopMove, kNoopMove}),
      result_(Result{EMPTY, 0, 0, {}}) {}

const Board& Game::board() const { return board_; }

const absl::InlinedVector<Move, constants::kMaxGameLen>& Game::moves() const {
  return moves_;
}

int Game::num_moves() const { return moves_.size() - kMoveOffset; }

Move Game::move(int move_num) const { return moves_[move_num + kMoveOffset]; }

Game::Result Game::result() const { return result_; }

bool Game::has_result() const { return result_.winner != EMPTY; }

float Game::komi() const { return board_.komi(); }

bool Game::IsGameOver() const { return board_.IsGameOver(); }

bool Game::IsValidMove(int index, Color color) const {
  return IsValidMove(AsLoc(index), color);
}

bool Game::IsValidMove(Loc loc, Color color) const {
  return board_.IsValidMove(loc, color);
}

bool Game::PlayMove(Loc loc, Color color) {
  bool ok = MoveOk(board_.PlayMove(loc, color));
  if (ok) {
    moves_.emplace_back(game::Move{color, loc});
  }

  return ok;
}

bool Game::Pass(Color color) {
  bool ok = MoveOk(board_.Pass(color));
  if (ok) {
    moves_.emplace_back(game::Move{color, kPassLoc});
  }

  return ok;
}

void Game::CalculatePassAliveRegions() { board_.CalculatePassAliveRegions(); }

Scores Game::GetScores() { return board_.GetScores(); }

void Game::WriteResult() {
  Scores scores = GetScores();
  if (scores.black_score > scores.white_score) {
    result_ = Result{BLACK, scores.black_score, scores.white_score,
                     false /* by_resign */, scores.ownership};
  } else if (scores.white_score > scores.black_score) {
    result_ = Result{WHITE, scores.black_score, scores.white_score,
                     false /* by_resign */, scores.ownership};
  }
}

void Game::SetWinner(Color winner) { result_.winner = winner; }

}  // namespace game
