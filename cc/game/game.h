#ifndef __GAME_GAME_H_
#define __GAME_GAME_H_

#include "absl/container/inlined_vector.h"
#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/move.h"

namespace game {

/*
 * Wrapper class containing all information pertaining to a single game.
 */
class Game final {
 public:
  Game();
  ~Game() = default;

  const Board& board() const { return board_; }
  int board_len() const { return board_.length(); }
  const absl::InlinedVector<Move, constants::kMaxGameLen>& moves() const {
    return moves_;
  }

  int move_num() const { return moves_.size() - kMoveOffset; }
  Move move(int move_num) const { return moves_[move_num + kMoveOffset]; }

  bool IsGameOver() const;

  bool IsValidMove(int index, color color) const;
  bool IsValidMove(Loc loc, color color) const;

  bool PlayMove(Loc loc, color color);
  bool Pass(color color);

  Scores GetScores();

 private:
  static constexpr int kMoveOffset = 5;
  Board board_;
  absl::InlinedVector<struct Move, constants::kMaxGameLen> moves_;
};
}  // namespace game

#endif
