#include "cc/game/board.h"

namespace game {

/*
 * Metadata for a single game.
 * 
 * Use in conjunction with `Board`
 */
class GameRecord final {
 public:
  bool IsGameEnded() const;

 private:
  std::vector<Transition> move_transitions;
};

}  // namespace game