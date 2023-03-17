#ifndef __GAME_ZOBRIST_HASH_H_
#define __GAME_ZOBRIST_HASH_H_

#include <cstdint>

#include "absl/numeric/int128.h"
#include "cc/constants/constants.h"

namespace game {

using HashKey = absl::uint128;

/*
 * Class Encapsulating Zobrist Initialization State.
 *
 * Should be globally initialized once, and immutable once initialized.
 */
class ZobristTable final {
 public:
  ZobristTable();
  ~ZobristTable() = default;

  // Disable Copy
  ZobristTable(ZobristTable const &) = delete;
  ZobristTable &operator=(ZobristTable const &) = delete;

  HashKey hash_at(unsigned i, unsigned j, unsigned state);

 private:
  // globally once-initialized
  HashKey table_[BOARD_LEN][BOARD_LEN][NUM_STATES];
};

}  // namespace game

#endif  // __GAME_ZOBRIST_HASH_H_