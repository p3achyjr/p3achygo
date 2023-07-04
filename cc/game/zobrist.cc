#include "cc/game/zobrist.h"

#include "cc/core/rand.h"

namespace game {

using ::core::PRng;

Zobrist::Zobrist() {
  PRng prng;

  for (auto i = 0; i < BOARD_LEN; i++) {
    for (auto j = 0; j < BOARD_LEN; j++) {
      for (auto k = 0; k < NUM_STATES; k++) {
        table_[i][j][k] = prng.next128();
      }
    }
  }
}

/* static */ const Zobrist& Zobrist::get() {
  static Zobrist zobrist;
  return zobrist;
}

}  // namespace game
