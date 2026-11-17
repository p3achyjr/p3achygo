#ifndef GTP_CLIENT_H_
#define GTP_CLIENT_H_

#include <atomic>
#include <deque>
#include <functional>
#include <sstream>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "cc/analysis/analysis.h"
#include "cc/eval/player_config.h"
#include "cc/gtp/command.h"
#include "cc/gtp/service.h"

namespace gtp {

enum class InputLoopStatus : uint8_t { kContinue = 0, kStop = 1 };

/*
 * GTP Client Thread. Parses commands and forwards them to the service.
 */
class Client final {
 public:
  Client();
  ~Client();

  absl::Status Start(std::string model_path, eval::PlayerSearchConfig cfg,
                     bool verbose = false);

  // Batch GTP: drive an already-built service, writing responses through
  // `sink` instead of stdout. Each call to `sink` is one complete response.
  void Start(std::unique_ptr<Service> service,
             std::function<void(const std::string&)> sink);

  // False once a quit command has been processed. The destructor then returns
  // promptly.
  bool Running() const { return running_.load(std::memory_order_acquire); }

  // Queues a silent quit: already queued commands still run and answer, then
  // the threads finish without writing a quit response. Used by batch mode at
  // shutdown for games that never received a quit.
  void Stop() ABSL_LOCKS_EXCLUDED(mu_);

  // Parses and adds `cmd_string` to the list of commands to process.
  // Returns whether to continue running the loop.
  InputLoopStatus ParseAndAddCommand(std::string cmd_string)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Stop analysis thread.
  void StopAnalysis() ABSL_LOCKS_EXCLUDED(mu_);

 private:
  // Client Loop.
  void ClientLoop();
  bool ShouldWakeClientThread();

  // Response Loop.
  void ResponseLoop();
  bool ShouldWakeResponseThread();

  // Analyze Loop (Spawned on `analyze`, `genmove_analyze` commands)
  void AnalysisSnapshotLoop(game::Color color, int centiseconds, bool* running);

  // genmove_analyze Command
  void GenmoveAnalyze(game::Color color, int centiseconds);

  // Main command handler.
  void HandleCommand(Command cmd) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  template <typename T>
  void AddResponse(Response<T> resp) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    std::stringstream ss;

    if (resp.error_message) {
      ss << "?" << (resp.id ? std::to_string(resp.id.value()) : "") << " "
         << resp.error_message.value() << "\n\n";
    } else {
      ss << "=" << (resp.id ? std::to_string(resp.id.value()) : "") << " "
         << GtpValueString(resp.resp_value) << "\n\n";
    }
    response_queue_.emplace_back(ss.str());
  }

  void AddAnalysisSnapshot(analysis::AnalysisSnapshot snapshot)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    response_queue_.emplace_back(GtpValueString(snapshot) + "\n");
  }

  void AddFinalAnalysisMove(game::Loc loc) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    response_queue_.emplace_back("play " + GtpValueString(loc) + "\n");
  }

  void AddRawResponse(std::string s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    response_queue_.emplace_back(std::move(s));
  }

  friend std::ostream& operator<<(std::ostream& os, const std::monostate& _) {
    return os;
  }

  void StartThreads();

  std::unique_ptr<Service> service_;
  std::function<void(const std::string&)> sink_;
  std::deque<Command> cmd_queue_ ABSL_GUARDED_BY(mu_);
  std::deque<std::string> response_queue_ ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
  std::atomic<bool> running_;
  // Set by ClientLoop on exit; ResponseLoop drains and exits after it.
  bool client_loop_done_ ABSL_GUARDED_BY(mu_) = false;
  // Set by Stop(): the pending quit gets no response.
  bool silent_quit_ ABSL_GUARDED_BY(mu_) = false;
  std::thread client_thread_;
  std::thread response_thread_;

  // analyze thread.
  bool analyze_running_;
  std::thread analyze_thread_;

  bool genmove_analyze_running_;
  std::thread genmove_analyze_thread_;
};

}  // namespace gtp

#endif
