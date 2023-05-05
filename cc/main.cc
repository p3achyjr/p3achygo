/*
 * Main function for standalone cc binary
 */

#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/nn/nn_interface.h"
#include "cc/self_play_thread.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  if (absl::GetFlag(FLAGS_model_path) == "") {
    LOG(WARNING) << "No Model Path Specified.";
    return 1;
  }

  int num_threads = absl::GetFlag(FLAGS_num_threads);
  LOG(INFO) << "Max Hardware Concurrency: "
            << std::thread::hardware_concurrency() << ". Using " << num_threads
            << " threads.";

  // initialize NN evaluator.
  std::unique_ptr<nn::NNInterface> nn_interface =
      std::make_unique<nn::NNInterface>(num_threads);
  CHECK_OK(nn_interface->Initialize(absl::GetFlag(FLAGS_model_path)));

  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    LOG(INFO) << "Spawning Thread " << thread_id << ".";
    std::thread thread(ExecuteSelfPlay, thread_id, nn_interface.get(),
                       absl::StrFormat("/tmp/thread%d_log.txt", thread_id));
    threads.emplace_back(std::move(thread));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  LOG(INFO) << "Self-Play Done!";
  return 0;
}
