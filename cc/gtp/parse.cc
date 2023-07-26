#include "cc/gtp/parse.h"

namespace gtp {

std::string ToLower(std::string s) {
  std::string ls;
  for (const auto& c : s) {
    ls += std::tolower(c);
  }

  return ls;
}

bool ParseVertex(std::string s, game::Loc* loc) {
  static const std::string kColIndices = "abcdefghjklmnopqrst";

  s = ToLower(s);
  if (s == "pass") {
    *loc = game::kPassLoc;
    return true;
  }

  std::string row_encoding = s.substr(1);
  char col_encoding = s[0];

  int row;
  int col = kColIndices.find(col_encoding);
  if (col == std::string::npos) {
    return false;
  }

  if (!absl::SimpleAtoi(row_encoding, &row)) {
    return false;
  }

  if (row < 1 || row > BOARD_LEN) {
    return false;
  }

  // normalize row to fit in [0, 18], and to index top to bottom.
  row = BOARD_LEN - row;

  *loc = game::Loc{row, col};
  return true;
}

bool ParseColor(std::string s, game::Color* color) {
  std::string ls = ToLower(s);
  if (ls == "black" || ls == "b") {
    *color = BLACK;
    return true;
  } else if (ls == "white" || ls == "w") {
    *color = WHITE;
    return true;
  }

  return false;
}

bool ParseMove(std::string cs, std::string vs, game::Move* move) {
  game::Color color;
  game::Loc loc;
  if (!ParseColor(cs, &color)) {
    return false;
  }

  if (!ParseVertex(vs, &loc)) {
    return false;
  }

  *move = game::Move{color, loc};
  return true;
}

std::vector<std::string> SplitLines(const std::string& s) {
  std::vector<std::string> lines;

  size_t start = 0, end;
  while ((end = s.find('\n', start)) != std::string::npos) {
    lines.push_back(s.substr(start, end - start));
    start = end + 1;
  }
  // Add the last line (or the only line if there are no newline characters)
  lines.push_back(s.substr(start));

  return lines;
}

std::string PreProcess(std::string cmd_string) {
  static constexpr char kElidedControlChars[] = {'\r', '\v', '\a', '\0',
                                                 '\b', '\f', '\e'};

  // Remove/convert elided chars.
  std::string cmd_string_cleaned;
  for (const char& c : cmd_string) {
    if (std::find(std::begin(kElidedControlChars),
                  std::end(kElidedControlChars),
                  c) != std::end(kElidedControlChars)) {
      continue;
    }

    cmd_string_cleaned += c == '\t' ? ' ' : c;
  }

  // Remove comments.
  int comment_start = cmd_string_cleaned.find('#');
  if (comment_start != std::string::npos) {
    cmd_string_cleaned.erase(comment_start);
  }

  // Remove whitespace lines.
  std::vector<std::string> lines = SplitLines(cmd_string_cleaned);
  std::string cmd_string_final;
  for (const std::string& line : lines) {
    auto is_whitespace_line = [&line]() {
      for (const auto& c : line) {
        if (!std::isspace(c)) {
          return false;
        }
      }

      return true;
    };

    if (!is_whitespace_line()) {
      cmd_string_final += line;
    }
  }

  return cmd_string_final;
}

}