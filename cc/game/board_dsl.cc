#include "cc/game/board_dsl.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"

namespace game {
namespace internal {

ParsedDSL ParseDSLString(std::string_view dsl) {
  ParsedDSL result;
  std::vector<char> cleaned;

  // Remove whitespace and validate characters
  for (char c : dsl) {
    if (std::isspace(c)) {
      continue;
    }

    switch (c) {
      case 'x':
      case 'X':
      case 'o':
      case 'O':
      case '.':
      case '+':
        cleaned.push_back(c);
        break;
      default:
        throw std::runtime_error(std::string("Invalid character in DSL: '") +
                                 c + "'");
    }
  }

  // Infer board size
  int total_positions = cleaned.size();
  double side_length = std::sqrt(total_positions);

  if (side_length != std::floor(side_length)) {
    throw std::runtime_error(
        "Board DSL must represent a square grid. Got " +
        std::to_string(total_positions) + " positions.");
  }

  result.inferred_size = static_cast<int>(side_length);

  // Parse stones
  for (int idx = 0; idx < total_positions; ++idx) {
    int i = idx / result.inferred_size;
    int j = idx % result.inferred_size;
    Loc loc{i, j};

    switch (cleaned[idx]) {
      case 'x':
      case 'X':
        result.black_stones.push_back(loc);
        break;
      case 'o':
      case 'O':
        result.white_stones.push_back(loc);
        break;
      case '.':
      case '+':
        // Empty intersection
        break;
    }
  }

  return result;
}

void ValidateBoardSize(const ParsedDSL& parsed) {
  if (parsed.inferred_size != BOARD_LEN) {
    throw std::runtime_error(
        "Board DSL size mismatch: expected " + std::to_string(BOARD_LEN) +
        "x" + std::to_string(BOARD_LEN) + " board, but got " +
        std::to_string(parsed.inferred_size) + "x" +
        std::to_string(parsed.inferred_size));
  }
}

}  // namespace internal

Board ParseBoardDSL(std::string_view dsl) {
  auto parsed = internal::ParseDSLString(dsl);
  internal::ValidateBoardSize(parsed);

  Board board;

  // Place all black stones first, then all white stones
  // The order doesn't affect the final position for legal positions
  for (const auto& loc : parsed.black_stones) {
    auto result = board.PlayMove(loc, BLACK);
    if (!MoveOk(result)) {
      throw std::runtime_error(
          "Failed to place black stone at " + std::to_string(loc.i) + "," +
          std::to_string(loc.j) + " (illegal position or move order issue)");
    }
  }

  for (const auto& loc : parsed.white_stones) {
    auto result = board.PlayMove(loc, WHITE);
    if (!MoveOk(result)) {
      throw std::runtime_error(
          "Failed to place white stone at " + std::to_string(loc.i) + "," +
          std::to_string(loc.j) + " (illegal position or move order issue)");
    }
  }

  return board;
}

Board ParseBoardDSLAuto(std::string_view dsl, Color first_color) {
  auto parsed = internal::ParseDSLString(dsl);
  internal::ValidateBoardSize(parsed);

  Board board;

  // Interleave moves between black and white to handle captures and complex positions
  auto& first_moves = (first_color == BLACK) ? parsed.black_stones : parsed.white_stones;
  auto& second_moves = (first_color == BLACK) ? parsed.white_stones : parsed.black_stones;

  size_t first_idx = 0;
  size_t second_idx = 0;

  while (first_idx < first_moves.size() || second_idx < second_moves.size()) {
    // Try placing a stone from the first color
    if (first_idx < first_moves.size()) {
      const auto& loc = first_moves[first_idx];
      auto result = board.PlayMoveDry(loc, first_color);
      if (MoveOk(result)) {
        board.PlayMove(loc, first_color);
        first_idx++;
      } else {
        // If we can't place this stone yet, try the other color
        if (second_idx < second_moves.size()) {
          const auto& loc2 = second_moves[second_idx];
          result = board.PlayMoveDry(loc2, OppositeColor(first_color));
          if (MoveOk(result)) {
            board.PlayMove(loc2, OppositeColor(first_color));
            second_idx++;
          } else {
            // Neither color can move - this might be a ko or complex situation
            // Try skipping to the next stone of the first color
            first_idx++;
          }
        } else {
          first_idx++;
        }
      }
    } else if (second_idx < second_moves.size()) {
      // Only second color moves left
      const auto& loc = second_moves[second_idx];
      auto result = board.PlayMove(loc, OppositeColor(first_color));
      if (!MoveOk(result)) {
        throw std::runtime_error(
            "Failed to place stone at " + std::to_string(loc.i) + "," +
            std::to_string(loc.j) + " - position may be illegal");
      }
      second_idx++;
    }
  }

  return board;
}

std::array<Color, BOARD_LEN * BOARD_LEN> ParseBoardGrid(std::string_view dsl) {
  auto parsed = internal::ParseDSLString(dsl);
  internal::ValidateBoardSize(parsed);

  std::array<Color, BOARD_LEN * BOARD_LEN> grid;
  grid.fill(EMPTY);

  for (const auto& loc : parsed.black_stones) {
    grid[loc.i * BOARD_LEN + loc.j] = BLACK;
  }

  for (const auto& loc : parsed.white_stones) {
    grid[loc.i * BOARD_LEN + loc.j] = WHITE;
  }

  return grid;
}

std::string BoardToDSL(const Board& board, bool compact) {
  std::ostringstream oss;

  for (int i = 0; i < BOARD_LEN; ++i) {
    for (int j = 0; j < BOARD_LEN; ++j) {
      Color c = board.at(i, j);

      if (c == BLACK) {
        oss << 'x';
      } else if (c == WHITE) {
        oss << 'o';
      } else {
        oss << '.';
      }

      if (!compact && j < BOARD_LEN - 1) {
        oss << ' ';
      }
    }

    if (i < BOARD_LEN - 1) {
      oss << '\n';
    }
  }

  return oss.str();
}

}  // namespace game
