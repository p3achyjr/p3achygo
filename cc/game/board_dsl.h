#ifndef GAME_BOARD_DSL_H_
#define GAME_BOARD_DSL_H_

#include <string>
#include <string_view>
#include <vector>

#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"

namespace game {

/**
 * Board DSL - A Domain-Specific Language for creating board positions from ASCII art.
 *
 * This DSL allows you to create board positions using a visual grid representation,
 * making tests much more readable and maintainable.
 *
 * Syntax:
 * - 'x' or 'X': Black stone
 * - 'o' or 'O': White stone
 * - '.' or '+': Empty intersection
 * - Whitespace (spaces, newlines): Ignored
 *
 * The grid can be written naturally with spaces between stones for readability:
 *
 * Example:
 *   auto board = ParseBoardDSL(R"(
 *     . . x . x o .
 *     . . x x o . o
 *     . . . . . . .
 *   )");
 *
 * Or more compactly:
 *   auto board = ParseBoardDSL("..x.xo...xxo.o.......");
 *
 * The DSL automatically infers the board dimensions and validates that the
 * grid matches the compile-time BOARD_LEN setting.
 */

/**
 * Parse a board position from a DSL string.
 *
 * @param dsl The DSL string representing the board position
 * @return A Board object with the specified position
 * @throws std::runtime_error if the DSL is invalid or doesn't match BOARD_LEN
 */
Board ParseBoardDSL(std::string_view dsl);

/**
 * Parse a board position from a DSL string with automatic move ordering.
 *
 * This function determines the order in which stones should be placed by
 * analyzing connectivity and groups. This is useful when the exact move
 * order doesn't matter for the test.
 *
 * @param dsl The DSL string representing the board position
 * @param first_color The color that moves first (default: BLACK)
 * @return A Board object with the specified position
 */
Board ParseBoardDSLAuto(std::string_view dsl, Color first_color = BLACK);

/**
 * Parse board data into a simple grid representation without playing moves.
 *
 * This returns a raw representation of the board position without going through
 * the game rules. Useful for testing board analysis functions directly.
 *
 * @param dsl The DSL string representing the board position
 * @return An array of colors representing each board location
 */
std::array<Color, BOARD_LEN * BOARD_LEN> ParseBoardGrid(std::string_view dsl);

/**
 * Convert a board position to a DSL string for debugging or display.
 *
 * @param board The board to convert
 * @param compact If true, produce compact output; if false, add spaces
 * @return A DSL string representing the board position
 */
std::string BoardToDSL(const Board& board, bool compact = false);

namespace internal {

// Internal structure to hold parsed DSL data
struct ParsedDSL {
  std::vector<Loc> black_stones;
  std::vector<Loc> white_stones;
  int inferred_size;
};

// Parse the DSL string into structured data
ParsedDSL ParseDSLString(std::string_view dsl);

// Validate that the parsed DSL matches the expected board size
void ValidateBoardSize(const ParsedDSL& parsed);

}  // namespace internal

}  // namespace game

#endif  // GAME_BOARD_DSL_H_
