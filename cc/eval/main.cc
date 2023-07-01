/*
 * Main function for eval binary.
 */

#include <sys/stat.h>

#include <chrono>
#include <future>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/constants/constants.h"
#include "cc/core/elo.h"
#include "cc/core/filepath.h"
#include "cc/eval/eval.h"
#include "cc/nn/nn_interface.h"

ABSL_FLAG(std::string, cur_model_path, "", "Path to current best model.");
ABSL_FLAG(std::string, cand_model_path, "", "Path to candidate model.");
ABSL_FLAG(std::string, res_write_path, "", "Path to write result to.");
ABSL_FLAG(int, cache_size, constants::kDefaultNNCacheSize / 2,
          "Default size of cache.");

static constexpr int kNumEvalGames = 48;
static constexpr int64_t kTimeoutUs = 4000;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  std::string cur_model_path = absl::GetFlag(FLAGS_cur_model_path);
  if (cur_model_path == "") {
    LOG(ERROR) << "--cur_model_path Not Specified.";
    return 1;
  }

  std::string cand_model_path = absl::GetFlag(FLAGS_cand_model_path);
  if (cand_model_path == "") {
    LOG(ERROR) << "--cand_model_path Not Specified.";
    return 1;
  }

  std::string res_write_path = absl::GetFlag(FLAGS_res_write_path);
  if (res_write_path == "") {
    LOG(ERROR) << "--res_write_path Not Specified.";
    return 1;
  }

  // Initialize NN evaluators.
  int cache_size = absl::GetFlag(FLAGS_cache_size);
  std::unique_ptr<nn::NNInterface> cur_nn_interface =
      std::make_unique<nn::NNInterface>(kNumEvalGames, kTimeoutUs, cache_size);
  std::unique_ptr<nn::NNInterface> cand_nn_interface =
      std::make_unique<nn::NNInterface>(kNumEvalGames, kTimeoutUs, cache_size);
  CHECK_OK(cur_nn_interface->Initialize(std::move(cur_model_path)));
  CHECK_OK(cand_nn_interface->Initialize(std::move(cand_model_path)));

  size_t time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();

  std::vector<std::thread> threads;
  std::vector<std::future<Winner>> winners;
  for (int thread_id = 0; thread_id < kNumEvalGames; ++thread_id) {
    std::promise<Winner> p;
    winners.emplace_back(p.get_future());

    size_t seed = absl::HashOf(time, thread_id);
    std::thread thread(PlayEvalGame, seed, thread_id, cur_nn_interface.get(),
                       cand_nn_interface.get(),
                       absl::StrFormat("/tmp/eval%d_log.txt", thread_id),
                       std::move(p));
    threads.emplace_back(std::move(thread));
  }

  LOG(INFO) << "Playing " << kNumEvalGames << " eval games.";

  for (auto& thread : threads) {
    thread.join();
  }

  int num_cand_won = 0;
  for (auto& winner : winners) {
    Winner res = winner.get();
    num_cand_won += res == Winner::kCand ? 1 : 0;
  }

  float winrate =
      (static_cast<float>(num_cand_won) / static_cast<float>(kNumEvalGames));
  float rel_elo = core::RelativeElo(winrate);

  LOG(INFO) << "Cand won " << num_cand_won << " games of " << kNumEvalGames
            << " for " << winrate << " winrate and " << rel_elo
            << " relative Elo.";

  FILE* const file = fopen(res_write_path.c_str(), "w");
  absl::FPrintF(file, "%f", rel_elo);
  fclose(file);
  return 0;
}
