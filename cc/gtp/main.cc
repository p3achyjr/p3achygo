#include <iostream>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "cc/gtp/client.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(int, n, 128, "Number of visits.");
ABSL_FLAG(int, k, 8, "Number of samples to draw.");

void InputLoop(gtp::Client* client) {
  while (true) {
    std::string cmd;
    std::getline(std::cin, cmd);

    client->StopAnalysis();
    gtp::InputLoopStatus status = client->ParseAndAddCommand(cmd);
    if (status == gtp::InputLoopStatus::kStop) {
      return;
    }
  }
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path == "") {
    std::cerr << "No Model Path (--model_path) specified.\n";
    return 1;
  }

  std::unique_ptr<gtp::Client> client = std::make_unique<gtp::Client>();

  std::cerr << "Initializing...\n";
  absl::Status status =
      client->Start(model_path, absl::GetFlag(FLAGS_n), absl::GetFlag(FLAGS_k));
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }
  std::cerr << "Gtp Protocol Ready!\n";

  std::thread input_thread = std::thread(&InputLoop, client.get());
  input_thread.join();

  return 0;
}
