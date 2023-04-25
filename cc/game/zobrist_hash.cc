#include "cc/game/zobrist_hash.h"

#include "absl/log/log.h"
#include "cc/core/rand.h"

namespace game {

using ::core::PRng;

Zobrist::Zobrist() {
  static bool created_table{false};

  if (created_table) {
    LOG(WARNING)
        << "Zobrist Table Already Created. Use the one that already exists.";
  }

  PRng prng;

  for (auto i = 0; i < BOARD_LEN; i++) {
    for (auto j = 0; j < BOARD_LEN; j++) {
      for (auto k = 0; k < NUM_STATES; k++) {
        table_[i][j][k] = prng.next128();
      }
    }
  }

  created_table = true;
}

Zobrist::Hash Zobrist::hash_at(unsigned i, unsigned j, unsigned state) {
  return table_[i][j][state];
}

}  // namespace game
