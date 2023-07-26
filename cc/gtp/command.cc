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
  } else if (token == "final_score") {
    return GTPCode::kFinalScore;
  } else if (token == "analyze") {
    return GTPCode::kAnalyze;
  } else if (token == "genmove_analyze") {
    return GTPCode::kGenMoveAnalyze;
  } else if (token == "play_dbg") {
    return GTPCode::kPlayDbg;
  } else if (token == "genmove_dbg") {
    return GTPCode::kGenMoveDbg;
  } else if (token == "ownership") {
    return GTPCode::kOwnership;
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
    case GTPCode::kFinalScore:
      return "final_score";
    case GTPCode::kAnalyze:
      return "analyze";
    case GTPCode::kGenMoveAnalyze:
      return "genmove_analyze";
    case GTPCode::kPlayDbg:
      return "play_dbg";
    case GTPCode::kGenMoveDbg:
      return "genmove_dbg";
    case GTPCode::kOwnership:
      return "ownership";
    case GTPCode::kUnknown:
    case GTPCode::kServerError:
    case GTPCode::kCommandParseError:
    case GTPCode::kUnknownCommand:
      return "";
  }

  return "";
}

std::string GtpValueString(std::monostate _) { return ""; }

std::string GtpValueString(int x) { return std::to_string(x); }

std::string GtpValueString(float x) { return std::to_string(x); }

std::string GtpValueString(std::string s) { return s; }

std::string GtpValueString(bool b) { return b ? "true" : "false"; }

std::string GtpValueString(game::Loc loc) {
  static constexpr char kColIndices[] = "ABCDEFGHJKLMNOPQRST";
  if (loc == game::kNoopLoc) {
    return "resign";
  }

  if (loc == game::kPassLoc) {
    return "pass";
  }

  std::string vertex_string;
  vertex_string += kColIndices[loc.j];
  vertex_string += std::to_string(BOARD_LEN - loc.i);

  return vertex_string;
}

std::string GtpValueString(game::Scores scores) {
  std::stringstream ss;
  ss << "B: " << scores.black_score << ", W: " << scores.white_score;

  return ss.str();
}

std::string GtpValueString(std::array<float, BOARD_LEN * BOARD_LEN> ownership) {
  std::stringstream ss;
  ss << "\n";

  float bown_est = 0, wown_est = 0;
  for (auto i = 0; i < BOARD_LEN; i++) {
    int gtp_row = BOARD_LEN - i;
    if (gtp_row < 10) {
      ss << gtp_row << "  ";
    } else {
      ss << gtp_row << " ";
    }

    for (auto j = 0; j < BOARD_LEN; j++) {
      float own = ownership[i * BOARD_LEN + j];
      if (own < -0.75f) {
        ss << "● ";
      } else if (own < -0.25f) {
        ss << "◆ ";
      } else if (own < 0.25f) {
        ss << "⋅ ";
      } else if (own < 0.75f) {
        ss << "◇ ";
      } else {
        ss << "○ ";
      }

      if (own < 0) {
        wown_est -= own;
      } else {
        bown_est += own;
      }
    }

    ss << "\n";
  }

  ss << "   "
     << "A B C D E F G H J K L M N O P Q R S T\n";
  ss << "Black Own Pred: " << bown_est
     << ", White Own Pred (No Komi): " << wown_est;
  return ss.str();
}

std::string GtpValueString(analysis::AnalysisSnapshot snapshot) {
  auto row_string = [](const analysis::AnalysisSnapshot::Row row) {
    std::stringstream ss;
    ss << "info";
    ss << " move " << GtpValueString(row.loc);
    ss << " visits " << row.visits;
    ss << " winrate " << row.winrate;
    ss << " prior " << row.prior;
    ss << " pv";
    for (const auto& mv : row.principal_variation) {
      ss << " " << GtpValueString(mv);
    }

    return ss.str();
  };

  std::stringstream ss;
  for (const auto& row : snapshot.rows) {
    ss << row_string(row) << " ";
  }

  return ss.str();
}

}  // namespace gtp
