#include "cc/gtp/client.h"

#include <iostream>
#include <sstream>
#include <thread>

#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/synchronization/mutex.h"

#define ARITY_CHECK(cmd, arity)                                     \
  if ((cmd).arg_tokens.size() != (arity)) {                         \
    AddResponse(MakeErrorResponse(                                  \
        (cmd).id, GTPCodeToString((cmd).code) + ": wrong arity.")); \
    return;                                                         \
  }

namespace gtp {
namespace {

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

}  // namespace

Client::Client() : running_(false) {}

Client::~Client() {
  client_thread_.join();
  CHECK(cmd_queue_.empty());
  CHECK(response_queue_.empty());
}

absl::Status Client::Start(std::string model_path) {
  absl::StatusOr<std::unique_ptr<Service>> service =
      Service::CreateService(model_path);
  if (!service.ok()) {
    return service.status();
  }

  service_ = std::move(service.value());
  running_.store(true, std::memory_order_release);
  client_thread_ = std::thread(&Client::ClientLoop, this);
  return absl::OkStatus();
}

InputLoopStatus Client::ParseAndAddCommand(std::string cmd_string) {
  std::string cmd_string_preprocessed = PreProcess(cmd_string);
  std::vector<std::string> tokens;
  std::istringstream ss(cmd_string_preprocessed);

  std::string token;
  while (ss >> token) {
    tokens.push_back(token);
  }

  Command cmd;
  for (int i = 0; i < tokens.size(); ++i) {
    const std::string& token = tokens[i];
    if (i == 0) {
      int id;
      if (absl::SimpleAtoi(token, &id)) {
        cmd.id = id;
      } else {
        cmd.code = StringToGTPCode(token);
      }
    } else if (i == 1 && cmd.code == GTPCode::kUnknown) {
      cmd.code = StringToGTPCode(token);
    } else {
      cmd.arg_tokens.emplace_back(token);
    }
  }

  absl::MutexLock l(&mu_);
  if (cmd.code == GTPCode::kUnknownCommand) {
    // unknown or ill-formatted command.
    cmd_queue_.emplace_back(
        Command{std::nullopt, GTPCode::kUnknownCommand, {"unknown command"}});
    return InputLoopStatus::kContinue;
  }

  cmd_queue_.emplace_back(cmd);
  if (cmd.code == GTPCode::kQuit) {
    return InputLoopStatus::kStop;
  }

  return InputLoopStatus::kContinue;
}

void Client::ClientLoop() {
  while (running_.load(std::memory_order_acquire)) {
    absl::MutexLock l(&mu_,
                      absl::Condition(this, &Client::ShouldWakeClientThread));
    // Consume entire command queue.
    std::vector<Command> cmds;
    while (!cmd_queue_.empty()) {
      cmds.emplace_back(cmd_queue_.front());
      cmd_queue_.pop_front();
    }

    // Handle all commands in order.
    for (const Command& cmd : cmds) {
      HandleCommand(cmd);
    }

    // Consume entire response queue.
    while (!response_queue_.empty()) {
      std::cout << response_queue_.front() << std::flush;
      response_queue_.pop_front();
    }
  }
}

bool Client::ShouldWakeClientThread() {
  return !running_.load(std::memory_order_acquire) || !cmd_queue_.empty();
}

void Client::HandleCommand(Command cmd) {
  switch (cmd.code) {
    case GTPCode::kProtocolVersion:
      AddResponse(service_->GtpProtocolVersion(cmd.id));
      return;
    case GTPCode::kName:
      AddResponse(service_->GtpName(cmd.id));
      return;
    case GTPCode::kVersion:
      AddResponse(service_->GtpVersion(cmd.id));
      return;
    case GTPCode::kKnownCommand:
      ARITY_CHECK(cmd, 1);
      AddResponse(service_->GtpKnownCommand(cmd.id, cmd.arg_tokens[0]));
      return;
    case GTPCode::kListCommands:
      AddResponse(service_->GtpListCommands(cmd.id));
      return;
    case GTPCode::kBoardSize:
      ARITY_CHECK(cmd, 1);
      {
        int bsize;
        if (!absl::SimpleAtoi(cmd.arg_tokens[0], &bsize)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "boardsize: could not parse argument into int."));
          return;
        }
        AddResponse(service_->GtpBoardSize(cmd.id, bsize));
        return;
      }
    case GTPCode::kClearBoard:
      AddResponse(service_->GtpClearBoard(cmd.id));
      return;
    case GTPCode::kKomi:
      ARITY_CHECK(cmd, 1);
      {
        float komi;
        if (!absl::SimpleAtof(cmd.arg_tokens[0], &komi)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "komi: could not parse argument into float."));
          return;
        }

        AddResponse(service_->GtpKomi(cmd.id, komi));
        return;
      }
    case GTPCode::kPlay:
      ARITY_CHECK(cmd, 2);
      {
        game::Move move;
        if (!ParseMove(cmd.arg_tokens[0], cmd.arg_tokens[1], &move)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "play: could not parse arguments into move."));
          return;
        }

        AddResponse(service_->GtpPlay(cmd.id, move));
        return;
      }
    case GTPCode::kGenMove:
      ARITY_CHECK(cmd, 1);
      {
        game::Color color;
        if (!ParseColor(cmd.arg_tokens[0], &color)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "genmove: could not parse argument into color."));
          return;
        }

        AddResponse(service_->GtpGenMove(cmd.id, color));
        return;
      }
    case GTPCode::kUnknownCommand:
      AddResponse(MakeErrorResponse(cmd.id, "unknown command"));
      return;
    case GTPCode::kQuit:
      AddResponse(MakeResponse(cmd.id));
      running_.store(false, std::memory_order_release);
      return;
    case GTPCode::kPrintBoard:
      AddResponse(service_->GtpPrintBoard(cmd.id));
      return;
    case GTPCode::kPlayDbg:
      ARITY_CHECK(cmd, 2);
      {
        game::Move move;
        if (!ParseMove(cmd.arg_tokens[0], cmd.arg_tokens[1], &move)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "play: could not parse arguments into move."));
          return;
        }

        AddResponse(service_->GtpPlayDbg(cmd.id, move));
        return;
      }
    case GTPCode::kGenMoveDbg:
      ARITY_CHECK(cmd, 1);
      {
        game::Color color;
        if (!ParseColor(cmd.arg_tokens[0], &color)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "genmove: could not parse argument into color."));
          return;
        }

        AddResponse(service_->GtpGenMoveDbg(cmd.id, color));
        return;
      }
    case GTPCode::kCommandParseError:
    case GTPCode::kUnknown:
    case GTPCode::kServerError:
      return;
  }
}

}  // namespace gtp
