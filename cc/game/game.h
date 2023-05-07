#ifndef __GAME_GAME_H_
#define __GAME_GAME_H_

#include "absl/container/inlined_vector.h"
#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/move.h"

namespace game {

/*
 * Wrapper class containing all information pertaining to a single game.
 */
class Game final {
 public:
  enum class ResultTag {
    kUnknown = 0,
    kBlackWinByScore,
    kWhiteWinByScore,
  };

  struct Result {
    ResultTag tag;
    float margin;
  };

  Game();
  Game(float komi);
  ~Game() = default;

  const Board& board() const;
  int board_len() const;
  const absl::InlinedVector<Move, constants::kMaxGameLen>& moves() const;
  int move_num() const;
  Move move(int move_num) const;
  Result result() const;
  bool has_result() const;
  float komi() const;

  bool IsGameOver() const;
  bool IsValidMove(int index, color color) const;
  bool IsValidMove(Loc loc, color color) const;
  bool PlayMove(Loc loc, color color);
  bool Pass(color color);

  Scores GetScores();
  void WriteResult();

 private:
  static constexpr int kMoveOffset = 5;
  Board board_;
  absl::InlinedVector<struct Move, constants::kMaxGameLen> moves_;
  Result result_;
};
}  // namespace game

#endif
