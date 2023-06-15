/*
 * Main function for eval binary.
 */

#include <sys/stat.h>

#include <future>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/constants/constants.h"
#include "cc/core/filepath.h"
#include "cc/eval/eval.h"
#include "cc/nn/nn_interface.h"

ABSL_FLAG(std::string, cur_model_path, "", "Path to current best model.");
ABSL_FLAG(std::string, cand_model_path, "", "Path to candidate model.");

static constexpr int kNumEvalGames = 48;
static constexpr int64_t kTimeoutUs = 3000;

std::string ToString(const Winner& winner) {
  return winner == Winner::kCur ? "CUR" : "CAND";
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  std::string cur_model_path = absl::GetFlag(FLAGS_cur_model_path);
  if (cur_model_path == "") {
    LOG(ERROR) << "Current Model Path Not Specified.";
    return 1;
  }

  std::string cand_model_path = absl::GetFlag(FLAGS_cand_model_path);
  if (cand_model_path == "") {
    LOG(ERROR) << "Candidate Model Path Not Specified.";
    return 1;
  }

  // Initialize NN evaluators. Disable caching to enforce stepping in lockstep.
  std::unique_ptr<nn::NNInterface> cur_nn_interface =
      std::make_unique<nn::NNInterface>(kNumEvalGames, kTimeoutUs);
  std::unique_ptr<nn::NNInterface> cand_nn_interface =
      std::make_unique<nn::NNInterface>(kNumEvalGames, kTimeoutUs);
  CHECK_OK(cur_nn_interface->Initialize(std::move(cur_model_path)));
  CHECK_OK(cand_nn_interface->Initialize(std::move(cand_model_path)));

  std::vector<std::thread> threads;
  std::vector<std::future<Winner>> winners;
  for (int thread_id = 0; thread_id < kNumEvalGames; ++thread_id) {
    LOG(INFO) << "Spawning Thread " << thread_id << ".";
    std::promise<Winner> p;
    winners.emplace_back(p.get_future());
    std::thread thread(PlayEvalGame, thread_id, cur_nn_interface.get(),
                       cand_nn_interface.get(),
                       absl::StrFormat("/tmp/eval%d_log.txt", thread_id),
                       std::move(p));
    threads.emplace_back(std::move(thread));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  int num_cand_won = 0;
  for (auto& winner : winners) {
    Winner res = winner.get();
    num_cand_won += res == Winner::kCand ? 1 : 0;
    LOG(INFO) << "Winner: " << ToString(res);
  }

  LOG(INFO) << "Cand won " << num_cand_won << " games of " << kNumEvalGames
            << " for "
            << (static_cast<float>(num_cand_won) /
                static_cast<float>(kNumEvalGames))
            << " winrate.";
  return 0;
}
