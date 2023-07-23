#ifndef GTP_CLIENT_H_
#define GTP_CLIENT_H_

#include <atomic>
#include <deque>
#include <sstream>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
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

  absl::Status Start(std::string model_path);

  // Parses and adds `cmd_string` to the list of commands to process.
  // Returns whether to continue running the loop.
  InputLoopStatus ParseAndAddCommand(std::string cmd_string)
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  void ClientLoop();
  bool ShouldWakeClientThread();
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

  friend std::ostream& operator<<(std::ostream& os, const std::monostate& _) {
    return os;
  }

  std::unique_ptr<Service> service_;
  std::deque<Command> cmd_queue_ ABSL_GUARDED_BY(mu_);
  std::deque<std::string> response_queue_ ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
  std::atomic<bool> running_;
  std::thread client_thread_;
};

}  // namespace gtp

#endif
