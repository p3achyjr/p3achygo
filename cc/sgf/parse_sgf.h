#ifndef SGF_PARSE_SGF_H_
#define SGF_PARSE_SGF_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "cc/constants/constants.h"
#include "cc/game/game.h"
#include "cc/sgf/sgf_tree.h"

namespace sgf {

/*
 * These are not verified to be grammar-compliant.
 *
 * Grammar:
 *
 * Collection = GameTree { GameTree }
 * GameTree   = "(" Sequence { GameTree } ")"
 * Sequence   = Node { Node }
 * Node       = ";" { Property }
 * Property   = PropIdent PropValue { PropValue }
 * PropIdent  = UcLetter { UcLetter }
 * PropValue  = "[" CValueType "]"
 * CValueType = (ValueType | Compose)
 * ValueType  = (None | Number | Real | Double | Color | SimpleText |
 *               Text | Point  | Move | Stone)
 */
absl::StatusOr<std::unique_ptr<sgf::SgfNode>> ParseSgf(std::string sgf_string);
absl::StatusOr<std::unique_ptr<sgf::SgfNode>> ParseSgfFile(
    std::string sgf_filename);

/*
 * Extracts relevant game information from an SGF tree.
 */
struct GameInfo {
  int board_size = BOARD_LEN;
  float komi;
  std::string b_player_name;
  std::string w_player_name;
  game::Game::Result result;
  int handicap = 0;
  absl::InlinedVector<game::Move, constants::kMaxNumMoves> main_variation;
};

GameInfo ExtractGameInfo(sgf::SgfNode* root);

}  // namespace sgf

#endif
