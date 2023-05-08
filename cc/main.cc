/*
 * Main function for standalone cc binary
 */

#include <filesystem>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/nn/nn_interface.h"
#include "cc/recorder/sgf_recorder.h"
#include "cc/self_play_thread.h"

namespace {
namespace fs = std::filesystem;

static constexpr char kSgfDir[] = "sgf";
static constexpr char kTfDir[] = "tf";
}  // namespace

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");
ABSL_FLAG(int, gumbel_n, 64, "Number of visits from root of Gumbel MCTS");
ABSL_FLAG(int, gumbel_k, 8, "Number of top moves to consider in Gumbel MCTS.");
ABSL_FLAG(std::string, recorder_path, "",
          "Path to write SGF files and TF examples. 'sgf' and 'tf' are "
          "appended to the path.");
ABSL_FLAG(int, flush_interval, 100, "Number of games to buffer before flush.");
ABSL_FLAG(int, max_moves, 600, "Maximum number of moves per game.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path == "") {
    LOG(ERROR) << "No Model Path Specified.";
    return 1;
  }

  fs::path recorder_path = absl::GetFlag(FLAGS_recorder_path);
  if (recorder_path == "") {
    LOG(WARNING) << "No Recorder Path Specified. SGF and TF files will be "
                    "written to /tmp/";
    recorder_path = "/tmp/";
  }

  fs::create_directory(recorder_path / kSgfDir);
  fs::create_directory(recorder_path / kTfDir);

  int num_threads = absl::GetFlag(FLAGS_num_threads);
  LOG(INFO) << "Max Hardware Concurrency: "
            << std::thread::hardware_concurrency() << ". Using " << num_threads
            << " threads.";

  int gumbel_n = absl::GetFlag(FLAGS_gumbel_n);
  int gumbel_k = absl::GetFlag(FLAGS_gumbel_k);
  LOG(INFO) << "Using " << gumbel_n << " visits and " << gumbel_k
            << " top moves for Gumbel MCTS.";

  // initialize NN evaluator.
  std::unique_ptr<nn::NNInterface> nn_interface =
      std::make_unique<nn::NNInterface>(num_threads);
  CHECK_OK(nn_interface->Initialize(std::move(model_path)));

  // initialize serialization objects.
  std::unique_ptr<recorder::SgfRecorder> sgf_recorder =
      recorder::SgfRecorder::Create(recorder_path / kSgfDir, num_threads,
                                    absl::GetFlag(FLAGS_flush_interval));

  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    LOG(INFO) << "Spawning Thread " << thread_id << ".";
    std::thread thread(
        ExecuteSelfPlay, thread_id, nn_interface.get(), sgf_recorder.get(),
        absl::StrFormat("/tmp/thread%d_log.txt", thread_id),
        absl::GetFlag(FLAGS_gumbel_n), absl::GetFlag(FLAGS_gumbel_k),
        absl::GetFlag(FLAGS_max_moves));
    threads.emplace_back(std::move(thread));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  LOG(INFO) << "Self-Play Done!";
  return 0;
}
