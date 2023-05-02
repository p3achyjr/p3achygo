#include "cc/game/zobrist.h"

#include "absl/log/log.h"
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

Zobrist::Hash Zobrist::hash_at(unsigned i, unsigned j, unsigned state) const {
  return table_[i][j][state];
}

/* static */ const Zobrist& Zobrist::get() {
  static Zobrist zobrist;
  return zobrist;
}

}  // namespace game
