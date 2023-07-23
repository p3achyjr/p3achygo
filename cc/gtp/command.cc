#include "cc/gtp/command.h"

#include <string>
#include <vector>

namespace gtp {

GTPCode StringToGTPCode(std::string token) {
  if (token == "protocol_version") {
    return GTPCode::kProtocolVersion;
  } else if (token == "name") {
    return GTPCode::kName;
  } else if (token == "version") {
    return GTPCode::kVersion;
  } else if (token == "known_command") {
    return GTPCode::kKnownCommand;
  } else if (token == "list_commands") {
    return GTPCode::kListCommands;
  } else if (token == "quit") {
    return GTPCode::kQuit;
  } else if (token == "boardsize") {
    return GTPCode::kBoardSize;
  } else if (token == "clear_board") {
    return GTPCode::kClearBoard;
  } else if (token == "komi") {
    return GTPCode::kKomi;
  } else if (token == "play") {
    return GTPCode::kPlay;
  } else if (token == "genmove") {
    return GTPCode::kGenMove;
  } else if (token == "print_board") {
    return GTPCode::kPrintBoard;
  } else if (token == "play_dbg") {
    return GTPCode::kPlayDbg;
  } else if (token == "genmove_dbg") {
    return GTPCode::kGenMoveDbg;
  } else {
    return GTPCode::kUnknownCommand;
  }
}

std::string GTPCodeToString(GTPCode code) {
  switch (code) {
    case GTPCode::kProtocolVersion:
      return "protocol_version";
    case GTPCode::kName:
      return "name";
    case GTPCode::kVersion:
      return "version";
    case GTPCode::kKnownCommand:
      return "known_command";
    case GTPCode::kListCommands:
      return "list_commands";
    case GTPCode::kQuit:
      return "quit";
    case GTPCode::kBoardSize:
      return "boardsize";
    case GTPCode::kClearBoard:
      return "clear_board";
    case GTPCode::kKomi:
      return "komi";
    case GTPCode::kPlay:
      return "play";
    case GTPCode::kGenMove:
      return "genmove";
    case GTPCode::kPrintBoard:
      return "print_board";
    case GTPCode::kPlayDbg:
      return "play_dbg";
    case GTPCode::kGenMoveDbg:
      return "genmove_dbg";
    case GTPCode::kUnknown:
    case GTPCode::kServerError:
    case GTPCode::kCommandParseError:
    case GTPCode::kUnknownCommand:
    default:
      return "";
  }
}

std::string GtpValueString(std::monostate _) { return ""; }

std::string GtpValueString(int x) { return std::to_string(x); }

std::string GtpValueString(float x) { return std::to_string(x); }

std::string GtpValueString(std::string s) { return s; }

std::string GtpValueString(bool b) { return b ? "true" : "false"; }

std::string GtpValueString(game::Loc loc) {
  static constexpr char kColIndices[] = "abcdefghjklmnopqrst";
  if (loc == game::kPassLoc) {
    return "pass";
  }

  std::string vertex_string;
  vertex_string += kColIndices[loc.j];
  vertex_string += std::to_string(BOARD_LEN - loc.i);

  return vertex_string;
}

}  // namespace gtp
