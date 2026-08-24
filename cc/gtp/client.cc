#include "cc/gtp/client.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <thread>

#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/synchronization/mutex.h"
#include "cc/gtp/parse.h"

#define ARITY_CHECK(cmd, arity)                                     \
  if ((cmd).arg_tokens.size() != (arity)) {                         \
    AddResponse(MakeErrorResponse(                                  \
        (cmd).id, GTPCodeToString((cmd).code) + ": wrong arity.")); \
    return;                                                         \
  }

namespace gtp {

Client::Client()
    : running_(false),
      analyze_running_(false),
      genmove_analyze_running_(false) {}

Client::~Client() {
  if (genmove_analyze_thread_.joinable()) {
    genmove_analyze_thread_.join();
  }

  if (analyze_thread_.joinable()) {
    analyze_thread_.join();
  }

  if (client_thread_.joinable()) {
    client_thread_.join();
  }

  if (response_thread_.joinable()) {
    response_thread_.join();
  }

  CHECK(cmd_queue_.empty());
  CHECK(response_queue_.empty());
}

absl::Status Client::Start(std::string model_path, eval::PlayerSearchConfig cfg,
                           bool verbose) {
  absl::StatusOr<std::unique_ptr<Service>> service =
      Service::CreateService(model_path, cfg, verbose);
  if (!service.ok()) {
    return service.status();
  }

  service_ = std::move(service.value());
  running_.store(true, std::memory_order_release);
  client_thread_ = std::thread(&Client::ClientLoop, this);
  response_thread_ = std::thread(&Client::ResponseLoop, this);
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

  Command cmd{std::nullopt, GTPCode::kUnknown, {}};
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

void Client::StopAnalysis() {
  if (genmove_analyze_thread_.joinable()) {
    // This is basically just a synchronous thread. Wait for it to end.
    genmove_analyze_thread_.join();
    return;
  }

  mu_.Lock();
  analyze_running_ = false;
  mu_.Unlock();

  if (analyze_thread_.joinable()) {
    analyze_thread_.join();
  }
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
  }
}

bool Client::ShouldWakeClientThread() {
  return !running_.load(std::memory_order_acquire) || !cmd_queue_.empty();
}

void Client::ResponseLoop() {
  while (running_.load(std::memory_order_acquire)) {
    absl::MutexLock l(&mu_,
                      absl::Condition(this, &Client::ShouldWakeResponseThread));
    // Consume entire response queue.
    while (!response_queue_.empty()) {
      std::cout << response_queue_.front() << std::flush;
      response_queue_.pop_front();
    }
  }
}

bool Client::ShouldWakeResponseThread() {
  return !running_.load(std::memory_order_acquire) || !response_queue_.empty();
}

void Client::AnalysisSnapshotLoop(game::Color color, int centiseconds,
                                  bool* running) {
  if (!(*running)) {
    return;
  }

  while (true) {
    bool stopped = mu_.LockWhenWithTimeout(
        absl::Condition(
            +[](bool* running) { return !(*running); }, running),
        absl::Milliseconds(centiseconds * 10));
    if (stopped) {
      service_->GtpStopAnalysis();
      mu_.Unlock();
      return;
    }

    AddAnalysisSnapshot(service_->GtpAnalysisSnapshot(color));
    mu_.Unlock();
  }
}

void Client::GenmoveAnalyze(game::Color color, int centiseconds) {
  DCHECK(genmove_analyze_running_);
  analyze_thread_ = std::thread(&Client::AnalysisSnapshotLoop, this, color,
                                centiseconds, &genmove_analyze_running_);
  game::Loc move = service_->GtpGenMoveAnalyze(color);

  mu_.Lock();
  genmove_analyze_running_ = false;
  mu_.Unlock();

  if (analyze_thread_.joinable()) {
    analyze_thread_.join();
  }

  mu_.Lock();
  AddFinalAnalysisMove(move);
  mu_.Unlock();
}

namespace {

// Writes one whole line per call. Chained operator<< interleaves between
// threads and produces spliced, misleading traces.
void TraceCmd(const char* event, GTPCode code, bool analyze_running,
              bool genmove_analyze_running, bool resume) {
  static const bool on = [] {
    const char* v = std::getenv("P3_PONDER_TRACE");
    return v != nullptr && v[0] == '1';
  }();
  if (!on) return;
  std::ostringstream ss;
  ss << "PTRACE " << event << " code=" << static_cast<int>(code)
     << " resume=" << resume << " analyze_running=" << analyze_running
     << " genmove_analyze_running=" << genmove_analyze_running << "\n";
  std::cerr << ss.str() << std::flush;
}

// Whether pondering should resume once a command finishes.
//
// Resuming is the default: a query like `printboard` should not cost the engine
// its thinking time. Only list a command here when resuming would actually be
// wrong -- because something else owns the engine, or because we are shutting
// down. "Game is over" is handled service-side in EnsurePondering(), which has
// the game state.
bool ShouldResumePonderingAfter(GTPCode code) {
  switch (code) {
    // Analysis owns the engine until it is stopped; pondering alongside it
    // would run two searches over the same tree and NN slot.
    case GTPCode::kAnalyze:
    case GTPCode::kGenMoveAnalyze:
    // Shutting down -- do not start work we are about to join.
    case GTPCode::kQuit:
      return false;
    default:
      return true;
  }
}

// Pauses pondering for the duration of a command, and resumes it afterwards for
// the commands above. Pausing unconditionally is the point: a newly added
// command is safe by default rather than safe only if its author remembered.
//
// Analysis commands never resume here, and do not need to: InputLoop calls
// StopAnalysis() before queueing each command, so analysis is never live while
// HandleCommand runs.
class [[nodiscard]] ScopedPonderPause {
 public:
  ScopedPonderPause(Service* s, std::function<bool()> should_resume)
      : s_(s), should_resume_(std::move(should_resume)) {
    s_->PausePondering();
  }
  ~ScopedPonderPause() {
    if (should_resume_()) s_->EnsurePondering();
  }
  ScopedPonderPause(const ScopedPonderPause&) = delete;
  ScopedPonderPause& operator=(const ScopedPonderPause&) = delete;

 private:
  Service* s_;
  std::function<bool()> should_resume_;
};

}  // namespace

void Client::HandleCommand(Command cmd) {
  // Evaluated at scope exit, after the handler has had a chance to start
  // analysis. Do not assume analysis has already been torn down: InputLoop
  // calls StopAnalysis() before *queueing* a command, but the analysis threads
  // are spawned here, on this thread, when the command is later dequeued -- so
  // StopAnalysis() can run before the spawn and find nothing to join. Both
  // flags are guarded by mu_, which is held for all of HandleCommand.
  TraceCmd("CMD_BEGIN", cmd.code, analyze_running_, genmove_analyze_running_,
           false);
  ScopedPonderPause ponder_pause(service_.get(), [this, code = cmd.code] {
    const bool resume = ShouldResumePonderingAfter(code) && !analyze_running_ &&
                        !genmove_analyze_running_;
    TraceCmd("CMD_END", code, analyze_running_, genmove_analyze_running_,
             resume);
    return resume;
  });
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
    case GTPCode::kFinalScore:
      AddResponse(service_->GtpFinalScore(cmd.id));
      return;
    case GTPCode::kAnalyze:
      ARITY_CHECK(cmd, 2);
      {
        game::Color color;
        int centiseconds;
        if (!ParseColor(cmd.arg_tokens[0], &color)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "analyze: could not parse argument into color."));
          return;
        }

        if (!absl::SimpleAtoi(cmd.arg_tokens[1], &centiseconds)) {
          AddResponse(MakeErrorResponse(
              cmd.id,
              "analyze: could not parse argument into centisecond interval."));
          return;
        }
        AddResponse(service_->GtpStartAnalysis(cmd.id, color));
        analyze_running_ = true;
        analyze_thread_ = std::thread(&Client::AnalysisSnapshotLoop, this,
                                      color, centiseconds, &analyze_running_);
        return;
      }
    case GTPCode::kGenMoveAnalyze:
      ARITY_CHECK(cmd, 2);
      {
        game::Color color;
        int centiseconds;
        if (!ParseColor(cmd.arg_tokens[0], &color)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "genmove_analyze: could not parse argument into color."));
          return;
        }

        if (!absl::SimpleAtoi(cmd.arg_tokens[1], &centiseconds)) {
          AddResponse(MakeErrorResponse(cmd.id,
                                        "genmove_analyze: could not parse "
                                        "argument into centisecond interval."));
          return;
        }
        AddResponse(Response<>{});
        genmove_analyze_running_ = true;
        genmove_analyze_thread_ =
            std::thread(&Client::GenmoveAnalyze, this, color, centiseconds);
        return;
      }
    case GTPCode::kCgosGenmoveAnalyze:
      ARITY_CHECK(cmd, 1);
      {
        game::Color color;
        if (!ParseColor(cmd.arg_tokens[0], &color)) {
          AddResponse(MakeErrorResponse(
              cmd.id,
              "cgos-genmove_analyze: could not parse argument into color."));
          return;
        }
        std::string analysis_json;
        game::Loc move = service_->GtpCgosGenmoveAnalyze(color, &analysis_json);
        if (!analysis_json.empty()) {
          AddRawResponse(analysis_json + "\n");
        }
        std::stringstream ss;
        ss << "= play " << (color == BLACK ? "b" : "w") << " "
           << GtpValueString(move) << "\n\n";
        AddRawResponse(ss.str());
        return;
      }
    case GTPCode::kUndo:
      AddResponse(service_->GtpUndo(cmd.id));
      return;
    case GTPCode::kExplainLastMove:
      AddResponse(service_->GtpExplainLastMove(cmd.id));
      return;
    case GTPCode::kTimeSettings:
      ARITY_CHECK(cmd, 3);
      {
        int main_time, byoyomi_time, byoyomi_stones;
        if (!absl::SimpleAtoi(cmd.arg_tokens[0], &main_time) ||
            !absl::SimpleAtoi(cmd.arg_tokens[1], &byoyomi_time) ||
            !absl::SimpleAtoi(cmd.arg_tokens[2], &byoyomi_stones)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "time_settings: could not parse arguments into ints."));
          return;
        }
        AddResponse(service_->GtpTimeSettings(cmd.id, main_time, byoyomi_time,
                                              byoyomi_stones));
        return;
      }
    case GTPCode::kTimeLeft:
      ARITY_CHECK(cmd, 3);
      {
        game::Color color;
        int seconds, stones;
        if (!ParseColor(cmd.arg_tokens[0], &color)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "time_left: could not parse argument into color."));
          return;
        }
        if (!absl::SimpleAtoi(cmd.arg_tokens[1], &seconds) ||
            !absl::SimpleAtoi(cmd.arg_tokens[2], &stones)) {
          AddResponse(MakeErrorResponse(
              cmd.id, "time_left: could not parse arguments into ints."));
          return;
        }
        AddResponse(service_->GtpTimeLeft(cmd.id, color, seconds, stones));
        return;
      }
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
    case GTPCode::kOwnership:
      AddResponse(service_->GtpOwnership(cmd.id));
      return;
    case GTPCode::kSerializeSgfWithTrees:
      ARITY_CHECK(cmd, 1);
      AddResponse(
          service_->GtpSerializeSgfWithTrees(cmd.id, cmd.arg_tokens[0]));
      return;
    case GTPCode::kLoadSgf:
      ARITY_CHECK(cmd, 1);
      AddResponse(service_->GtpLoadSgf(cmd.id, cmd.arg_tokens[0]));
      return;
    case GTPCode::kCommandParseError:
    case GTPCode::kUnknown:
    case GTPCode::kServerError:
      return;
  }
}

}  // namespace gtp
