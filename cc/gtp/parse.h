#ifndef GTP_PARSE_H_
#define GTP_PARSE_H_

#include <string>

#include "absl/strings/numbers.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"

namespace gtp {

std::string ToLower(std::string s);
bool ParseVertex(std::string s, game::Loc* loc);
bool ParseColor(std::string s, game::Color* color);
bool ParseMove(std::string cs, std::string vs, game::Move* move);
std::vector<std::string> SplitLines(const std::string& s);
std::string PreProcess(std::string cmd_string);

}  // namespace gtp

#endif
