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
  static constexpr int kMoveOffset = 5;
  struct Result {
    Color winner;
    float bscore;
    float wscore;
    bool by_resign;

    std::array<Color, BOARD_LEN * BOARD_LEN> ownership;
  };

  explicit Game(
      const Board& board,
      const absl::InlinedVector<Move, constants::kMaxGameLen> last_moves);
  Game();
  ~Game() = default;

  const Board& board() const;

  // returns moves _with_ noop padding at the beginning.
  const absl::InlinedVector<Move, constants::kMaxGameLen>& moves() const;

  // returns number of _player made_ moves.
  int num_moves() const;

  // returns the `move_num`th _player made_ move.
  Move move(int move_num) const;

  Result result() const;
  bool has_result() const;
  float komi() const;

  bool IsGameOver() const;
  bool IsValidMove(int index, Color color) const;
  bool IsValidMove(Loc loc, Color color) const;
  bool PlayMove(Loc loc, Color color);
  bool Pass(Color color);

  void CalculatePassAliveRegions();

  Scores GetScores();
  void WriteResult();
  void SetWinner(Color winner) { result_.winner = winner; }
  void SetDidResign(bool did_resign) { result_.by_resign = true; }

 private:
  Board board_;
  absl::InlinedVector<struct Move, constants::kMaxGameLen> moves_;
  Result result_;
};
}  // namespace game

#endif
