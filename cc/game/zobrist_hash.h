#ifndef __GAME_ZOBRIST_HASH_H_
#define __GAME_ZOBRIST_HASH_H_

#include <cstdint>

#include "absl/numeric/int128.h"
#include "cc/constants/constants.h"

namespace game {

/*
 * Class Encapsulating Zobrist Initialization State.
 *
 * Should be globally initialized once, and immutable once initialized.
 */
class Zobrist final {
 public:
  using Hash = absl::uint128;

  Zobrist();
  ~Zobrist() = default;

  // Disable Copy
  Zobrist(Zobrist const &) = delete;
  Zobrist &operator=(Zobrist const &) = delete;

  Hash hash_at(unsigned i, unsigned j, unsigned state);

 private:
  // globally once-initialized
  Hash table_[BOARD_LEN][BOARD_LEN][NUM_STATES];
};

}  // namespace game

#endif  // __GAME_ZOBRIST_HASH_H_
