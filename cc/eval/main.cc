/*
 * Main function for eval binary.
 */

#include <sys/stat.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <regex>
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
#include "cc/recorder/game_recorder.h"

namespace {
namespace fs = std::filesystem;
static constexpr int64_t kTimeoutUs = 4000;
static constexpr int kDefaultGumbelN = 128;
static constexpr int kDefaultGumbelK = 8;
static constexpr int kPositionCacheSize =
    67108864;  // 2 ^ 26. The cache holds a single int, so size should be ok.
static constexpr int kRootsCacheSize = 131072;  // 2 ^ 16.
}  // namespace

ABSL_FLAG(std::string, cur_model_path, "", "Path to current best model.");
ABSL_FLAG(std::string, cand_model_path, "", "Path to candidate model.");
ABSL_FLAG(std::string, res_write_path, "", "Path to write result to.");
ABSL_FLAG(std::string, recorder_path, "", "Path to write SGF files.");
ABSL_FLAG(int, num_games, 0, "Number of eval games");
ABSL_FLAG(int, cache_size, constants::kDefaultNNCacheSize / 2,
          "Default size of cache.");
ABSL_FLAG(int, cur_n, kDefaultGumbelN, "N for current player");
ABSL_FLAG(int, cur_k, kDefaultGumbelK, "K for current player");
ABSL_FLAG(int, cand_n, kDefaultGumbelN, "N for candidate player");
ABSL_FLAG(int, cand_k, kDefaultGumbelK, "K for candidate player");

float ConfidenceDelta(float z_score, float num_sims, float wr) {
  return z_score * std::sqrt(wr * (1 - wr) / num_sims);
}

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
    LOG(WARNING)
        << "--res_write_path Not Specified. Result will not be written.";
  }

  core::FilePath recorder_path(absl::GetFlag(FLAGS_recorder_path));
  if (recorder_path == "") {
    LOG(WARNING)
        << "--recorder_path Not Specified. No SGF files will be written";
  }

  int num_games = absl::GetFlag(FLAGS_num_games);
  if (num_games == 0) {
    LOG(ERROR) << "--num_games Not Specified.";
    return 1;
  }

  // Initialize NN evaluators.
  int cache_size = absl::GetFlag(FLAGS_cache_size);
  std::unique_ptr<nn::NNInterface> cur_nn_interface =
      std::make_unique<nn::NNInterface>(num_games, kTimeoutUs, cache_size);
  std::unique_ptr<nn::NNInterface> cand_nn_interface =
      std::make_unique<nn::NNInterface>(num_games, kTimeoutUs, cache_size);
  CHECK_OK(cur_nn_interface->Initialize(std::move(cur_model_path)));
  CHECK_OK(cand_nn_interface->Initialize(std::move(cand_model_path)));

  size_t time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();

  // N, K
  int cur_n = absl::GetFlag(FLAGS_cur_n);
  int cand_n = absl::GetFlag(FLAGS_cand_n);
  int cur_k = absl::GetFlag(FLAGS_cur_k);
  int cand_k = absl::GetFlag(FLAGS_cand_k);

  // Game Recorder.
  auto find_model_name = [](fs::path path) {
    static constexpr char kModelNameRegex[] = "model_(\\d+)";
    static const std::regex re(kModelNameRegex);

    std::smatch match;
    for (const auto& part : path) {
      std::string part_string = part.filename().string();
      if (std::regex_match(part_string, match, re)) {
        return part_string;
      }
    }

    return std::string("UNKNOWN");
  };
  std::string cur_name = find_model_name(fs::path(cur_model_path)) + "n" +
                         absl::StrFormat("%d", cur_n) + "k" +
                         absl::StrFormat("%d", cur_k);
  std::string cand_name = find_model_name(fs::path(cand_model_path)) + "n" +
                          absl::StrFormat("%d", cand_n) + "k" +
                          absl::StrFormat("%d", cand_k);
  std::unique_ptr<recorder::GameRecorder> game_recorder =
      recorder::GameRecorder::Create(
          recorder_path, num_games, 10000, 0,
          absl::StrFormat("EVAL_%s_%s", cur_name, cand_name));

  // Spawn games.
  EvalConfig config =
      EvalConfig{cur_name, cand_name, cur_n, cur_k, cand_n, cand_k};
  std::vector<std::thread> threads;
  std::vector<std::future<EvalResult>> eval_results;
  for (int thread_id = 0; thread_id < num_games; ++thread_id) {
    std::promise<EvalResult> eval_result;
    eval_results.emplace_back(eval_result.get_future());

    size_t seed = absl::HashOf(time, thread_id);
    std::thread thread(PlayEvalGame, seed, thread_id, cur_nn_interface.get(),
                       cand_nn_interface.get(),
                       absl::StrFormat("/tmp/eval%d_log.txt", thread_id),
                       std::move(eval_result), game_recorder.get(), config);
    threads.emplace_back(std::move(thread));
  }

  LOG(INFO) << "Playing " << num_games << " eval games.";

  for (auto& thread : threads) {
    thread.join();
  }

  int num_cand_won = 0;
  int total_num_moves = 0;
  for (auto& eval_result : eval_results) {
    EvalResult res = eval_result.get();
    Winner winner = res.winner;
    num_cand_won += winner == Winner::kCand ? 1 : 0;
    total_num_moves += res.num_moves;
  }

  float winrate =
      (static_cast<float>(num_cand_won) / static_cast<float>(num_games));
  float rel_elo = core::RelativeElo(winrate);
  float c95 = ConfidenceDelta(1.96f, num_games, winrate);
  float elo_c95 = core::RelativeElo(.5f + c95);

  LOG(INFO) << "\n--- N, K ---\nCur N: " << cur_n << ", K: " << cur_k
            << "\nCand N: " << cand_n << ", K: " << cand_k;
  LOG(INFO) << "\n--- Elo, Winrate ---\nCand won " << num_cand_won
            << " games of " << num_games << "\nWin Rate (p95): "
            << absl::StrFormat("%.3f +- %.3f", winrate, c95)
            << "\nRelative Elo (p95): "
            << absl::StrFormat("%.3f +- %.3f", rel_elo, elo_c95);

  if (res_write_path != "") {
    FILE* const file = fopen(res_write_path.c_str(), "w");
    absl::FPrintF(file, "%f", rel_elo);
    fclose(file);
  }
  return 0;
}
