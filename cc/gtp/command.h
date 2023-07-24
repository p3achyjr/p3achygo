#ifndef GTP_COMMAND_H_
#define GTP_COMMAND_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "cc/game/game.h"

namespace gtp {

enum class GTPCode : uint8_t {
  kUnknown = 0,

  // Commands.
  kProtocolVersion = 1,
  kName = 2,
  kVersion = 3,
  kKnownCommand = 4,
  kListCommands = 5,
  kQuit = 6,
  kBoardSize = 7,
  kClearBoard = 8,
  kKomi = 9,
  kPlay = 10,
  kGenMove = 11,
  kPrintBoard = 12,
  kFinalScore = 13,

  // [13 - 100 reserved]
  // Private Commands.
  kPlayDbg = 101,
  kGenMoveDbg = 102,

  // [100 - 200 reserved]
  // Errors.
  kServerError = 201,
  kCommandParseError = 202,
  kUnknownCommand = 203,
};

static constexpr GTPCode kSupportedCommands[] = {
    // Common.
    GTPCode::kProtocolVersion,
    GTPCode::kName,
    GTPCode::kVersion,
    GTPCode::kKnownCommand,
    GTPCode::kListCommands,
    GTPCode::kQuit,
    GTPCode::kBoardSize,
    GTPCode::kClearBoard,
    GTPCode::kKomi,
    GTPCode::kPlay,
    GTPCode::kGenMove,
    GTPCode::kPrintBoard,
    GTPCode::kFinalScore,

    // Private.
    GTPCode::kPlayDbg,
    GTPCode::kGenMoveDbg,
};

struct Command {
  std::optional<int> id;
  GTPCode code = GTPCode::kUnknown;
  std::vector<std::string> arg_tokens;
};

template <typename T = std::monostate>
struct Response {
  std::optional<int> id;
  std::optional<std::string> error_message;
  T resp_value;
};

template <typename T>
Response<T> MakeResponse(std::optional<int> id, T val) {
  return Response<T>{id, {}, val};
}

template <typename T>
Response<T> MakeErrorResponse(std::optional<int> id, std::string msg) {
  return Response<T>{id, msg};
}

inline Response<> MakeResponse(std::optional<int> id) {
  return Response<>{id, {}};
}

inline Response<> MakeErrorResponse(std::optional<int> id, std::string msg) {
  return Response<>{id, msg, std::monostate{}};
}

GTPCode StringToGTPCode(std::string token);
std::string GTPCodeToString(GTPCode code);

std::string GtpValueString(std::monostate _);
std::string GtpValueString(int x);
std::string GtpValueString(float x);
std::string GtpValueString(std::string s);
std::string GtpValueString(bool b);
std::string GtpValueString(game::Loc loc);
std::string GtpValueString(game::Scores loc);

}  // namespace gtp

#endif
