/*
 * Main function for eval binary.
 */

#include <sys/stat.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <regex>
#include <thread>

#include "absl/flags/commandlineflag.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/reflection.h"
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
#include "cc/eval/player_config.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"
#include "cc/recorder/dir.h"
#include "cc/recorder/game_recorder.h"

namespace {
namespace fs = std::filesystem;
static constexpr int64_t kTimeoutUs = 4000;
static constexpr int kDefaultGumbelN = 128;
static constexpr int kDefaultGumbelK = 8;
}  // namespace

ABSL_FLAG(std::string, cur_model_path, "", "Path to current best model.");
ABSL_FLAG(std::string, cand_model_path, "", "Path to candidate model.");
ABSL_FLAG(std::string, res_write_path, "", "Path to write result to.");
ABSL_FLAG(std::string, recorder_path, "", "Path to write SGF files.");
ABSL_FLAG(int, num_games, 0, "Number of eval games");
ABSL_FLAG(int, cache_size, constants::kDefaultNNCacheSize / 2,
          "Default size of cache.");

// Per-player config files. When provided, each file sets the base config for
// that player. Individual flags below still take priority if explicitly passed
// on the command line.
ABSL_FLAG(std::string, cur_config, "",
          "Path to cur player config file (key: value, one per line). "
          "Explicit cur_* flags take priority over the file.");
ABSL_FLAG(std::string, cand_config, "",
          "Path to cand player config file (key: value, one per line). "
          "Explicit cand_* flags take priority over the file.");

// Individual per-player flags (legacy; always override the config file when
// explicitly specified on the command line).
ABSL_FLAG(int, cur_n, kDefaultGumbelN, "N for current player");
ABSL_FLAG(int, cur_k, kDefaultGumbelK, "K for current player");
ABSL_FLAG(float, cur_noise_scaling, 1.0f, "Cur gumbel noise scaling");
ABSL_FLAG(bool, cur_use_puct, true, "Whether to use PUCT for cur.");
ABSL_FLAG(bool, cur_use_lcb, true, "Whether to use LCB in PUCT for cur.");
ABSL_FLAG(float, cur_c_puct, 1.0f, "c_puct for cur.");
ABSL_FLAG(bool, cur_var_scale_cpuct, false,
          "Whether to scale c_puct based on variance for cur.");
ABSL_FLAG(
    bool, cur_use_mcgs, false,
    "Whether to use Monte-Carlo Graph Search (transposition table) for cur.");
ABSL_FLAG(bool, cur_use_puct_v, false, "Whether to use PUCT-V for cur.");
ABSL_FLAG(float, cur_c_puct_v_2, 3.0, "cur c_puct_v_2 scaling.");
ABSL_FLAG(int, cand_n, kDefaultGumbelN, "N for candidate player");
ABSL_FLAG(int, cand_k, kDefaultGumbelK, "K for candidate player");
ABSL_FLAG(float, cand_noise_scaling, 1.0f, "Cand gumbel noise scaling");
ABSL_FLAG(bool, cand_use_puct, true, "Whether to use PUCT for cand.");
ABSL_FLAG(bool, cand_use_lcb, true, "Whether to use LCB in PUCT for cand.");
ABSL_FLAG(float, cand_c_puct, 1.0f, "c_puct for cand.");
ABSL_FLAG(bool, cand_var_scale_cpuct, false,
          "Whether to scale c_puct based on variance for cand.");
ABSL_FLAG(
    bool, cand_use_mcgs, false,
    "Whether to use Monte-Carlo Graph Search (transposition table) for cand.");
ABSL_FLAG(bool, cand_use_puct_v, false, "Whether to use PUCT-V for cand.");
ABSL_FLAG(float, cand_c_puct_v_2, 3.0, "cand c_puct_v_2 scaling.");

float ConfidenceDelta(float z_score, float num_sims, float wr) {
  return z_score * std::sqrt(wr * (1 - wr) / num_sims);
}

// Returns true if the named flag was explicitly set to a value other than its
// default. (If the user passes the default value explicitly that's a no-op.)
bool IsOnCommandLine(const char* name) {
  const absl::CommandLineFlag* f = absl::FindCommandLineFlag(name);
  return f && f->CurrentValue() != f->DefaultValue();
}

// Applies any cur_* flags that were explicitly passed on the command line,
// overriding the corresponding fields in `cfg`.
void ApplyCurCommandLineFlags(eval::PlayerSearchConfig& cfg) {
  if (IsOnCommandLine("cur_n")) cfg.n = absl::GetFlag(FLAGS_cur_n);
  if (IsOnCommandLine("cur_k")) cfg.k = absl::GetFlag(FLAGS_cur_k);
  if (IsOnCommandLine("cur_noise_scaling"))
    cfg.noise_scaling = absl::GetFlag(FLAGS_cur_noise_scaling);
  if (IsOnCommandLine("cur_use_puct"))
    cfg.use_puct = absl::GetFlag(FLAGS_cur_use_puct);
  if (IsOnCommandLine("cur_use_lcb"))
    cfg.use_lcb = absl::GetFlag(FLAGS_cur_use_lcb);
  if (IsOnCommandLine("cur_c_puct"))
    cfg.c_puct = absl::GetFlag(FLAGS_cur_c_puct);
  if (IsOnCommandLine("cur_var_scale_cpuct"))
    cfg.var_scale_cpuct = absl::GetFlag(FLAGS_cur_var_scale_cpuct);
  if (IsOnCommandLine("cur_use_mcgs"))
    cfg.use_mcgs = absl::GetFlag(FLAGS_cur_use_mcgs);
  if (IsOnCommandLine("cur_use_puct_v"))
    cfg.use_puct_v = absl::GetFlag(FLAGS_cur_use_puct_v);
  if (IsOnCommandLine("cur_c_puct_v_2"))
    cfg.c_puct_v_2 = absl::GetFlag(FLAGS_cur_c_puct_v_2);
}

// Same for cand_* flags.
void ApplyCandCommandLineFlags(eval::PlayerSearchConfig& cfg) {
  if (IsOnCommandLine("cand_n")) cfg.n = absl::GetFlag(FLAGS_cand_n);
  if (IsOnCommandLine("cand_k")) cfg.k = absl::GetFlag(FLAGS_cand_k);
  if (IsOnCommandLine("cand_noise_scaling"))
    cfg.noise_scaling = absl::GetFlag(FLAGS_cand_noise_scaling);
  if (IsOnCommandLine("cand_use_puct"))
    cfg.use_puct = absl::GetFlag(FLAGS_cand_use_puct);
  if (IsOnCommandLine("cand_use_lcb"))
    cfg.use_lcb = absl::GetFlag(FLAGS_cand_use_lcb);
  if (IsOnCommandLine("cand_c_puct"))
    cfg.c_puct = absl::GetFlag(FLAGS_cand_c_puct);
  if (IsOnCommandLine("cand_var_scale_cpuct"))
    cfg.var_scale_cpuct = absl::GetFlag(FLAGS_cand_var_scale_cpuct);
  if (IsOnCommandLine("cand_use_mcgs"))
    cfg.use_mcgs = absl::GetFlag(FLAGS_cand_use_mcgs);
  if (IsOnCommandLine("cand_use_puct_v"))
    cfg.use_puct_v = absl::GetFlag(FLAGS_cand_use_puct_v);
  if (IsOnCommandLine("cand_c_puct_v_2"))
    cfg.c_puct_v_2 = absl::GetFlag(FLAGS_cand_c_puct_v_2);
}

// Builds a PlayerSearchConfig from the cur_* flags (all fields, no file).
eval::PlayerSearchConfig CurConfigFromFlags() {
  eval::PlayerSearchConfig cfg;
  cfg.n = absl::GetFlag(FLAGS_cur_n);
  cfg.k = absl::GetFlag(FLAGS_cur_k);
  cfg.noise_scaling = absl::GetFlag(FLAGS_cur_noise_scaling);
  cfg.use_puct = absl::GetFlag(FLAGS_cur_use_puct);
  cfg.use_lcb = absl::GetFlag(FLAGS_cur_use_lcb);
  cfg.c_puct = absl::GetFlag(FLAGS_cur_c_puct);
  cfg.var_scale_cpuct = absl::GetFlag(FLAGS_cur_var_scale_cpuct);
  cfg.use_mcgs = absl::GetFlag(FLAGS_cur_use_mcgs);
  cfg.use_puct_v = absl::GetFlag(FLAGS_cur_use_puct_v);
  cfg.c_puct_v_2 = absl::GetFlag(FLAGS_cur_c_puct_v_2);
  return cfg;
}

// Builds a PlayerSearchConfig from the cand_* flags (all fields, no file).
eval::PlayerSearchConfig CandConfigFromFlags() {
  eval::PlayerSearchConfig cfg;
  cfg.n = absl::GetFlag(FLAGS_cand_n);
  cfg.k = absl::GetFlag(FLAGS_cand_k);
  cfg.noise_scaling = absl::GetFlag(FLAGS_cand_noise_scaling);
  cfg.use_puct = absl::GetFlag(FLAGS_cand_use_puct);
  cfg.use_lcb = absl::GetFlag(FLAGS_cand_use_lcb);
  cfg.c_puct = absl::GetFlag(FLAGS_cand_c_puct);
  cfg.var_scale_cpuct = absl::GetFlag(FLAGS_cand_var_scale_cpuct);
  cfg.use_mcgs = absl::GetFlag(FLAGS_cand_use_mcgs);
  cfg.use_puct_v = absl::GetFlag(FLAGS_cand_use_puct_v);
  cfg.c_puct_v_2 = absl::GetFlag(FLAGS_cand_c_puct_v_2);
  return cfg;
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

  int perms =
      S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
  mkdir((recorder_path / recorder::kSgfDir).c_str(), perms);

  // Resolve per-player configs.
  // Priority: explicit command-line flags > config file > PlayerSearchConfig
  // defaults.
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

  std::string cur_config_path = absl::GetFlag(FLAGS_cur_config);
  eval::PlayerSearchConfig cur_cfg;
  if (cur_config_path.empty()) {
    cur_cfg = CurConfigFromFlags();
  } else {
    cur_cfg = eval::ParsePlayerConfigFile(cur_config_path);
    ApplyCurCommandLineFlags(cur_cfg);
  }
  cur_cfg.name = find_model_name(fs::path(cur_model_path)) + "n" +
                 absl::StrFormat("%d", cur_cfg.n) + "k" +
                 absl::StrFormat("%d", cur_cfg.k);

  std::string cand_config_path = absl::GetFlag(FLAGS_cand_config);
  eval::PlayerSearchConfig cand_cfg;
  if (cand_config_path.empty()) {
    cand_cfg = CandConfigFromFlags();
  } else {
    cand_cfg = eval::ParsePlayerConfigFile(cand_config_path);
    ApplyCandCommandLineFlags(cand_cfg);
  }
  cand_cfg.name = find_model_name(fs::path(cand_model_path)) + "n" +
                  absl::StrFormat("%d", cand_cfg.n) + "k" +
                  absl::StrFormat("%d", cand_cfg.k);

  // Initialize NN evaluators.
  // For parallel search (num_threads_per_game > 1 or time_ms > 0) each game
  // owns a contiguous slice of [game_id*N, (game_id+1)*N) thread IDs, so the
  // engine batch size must be num_games * num_threads_per_game and the
  // NNInterface uses kExplicit signalling with
  // num_shared_search_tasks=num_games. For serial Gumbel the batch is num_games
  // with kAuto signalling (unchanged).
  const bool cur_uses_search =
      cur_cfg.num_threads_per_game > 1 || cur_cfg.time_ms > 0;
  const bool cand_uses_search =
      cand_cfg.num_threads_per_game > 1 || cand_cfg.time_ms > 0;
  const int cur_batch_size = num_games * cur_cfg.num_threads_per_game;
  const int cand_batch_size = num_games * cand_cfg.num_threads_per_game;

  int cache_size = absl::GetFlag(FLAGS_cache_size);
  std::unique_ptr<nn::Engine> cur_engine = nn::CreateEngine(
      nn::KindFromEnginePath(cur_model_path), cur_model_path, cur_batch_size,
      nn::GetVersionFromModelPath(cur_model_path));
  std::unique_ptr<nn::Engine> cand_engine = nn::CreateEngine(
      nn::KindFromEnginePath(cand_model_path), cand_model_path, cand_batch_size,
      nn::GetVersionFromModelPath(cand_model_path));
  std::unique_ptr<nn::NNInterface> cur_nn_interface =
      cur_uses_search
          ? std::make_unique<nn::NNInterface>(
                cur_batch_size, kTimeoutUs, cache_size, std::move(cur_engine),
                nn::NNInterface::SignalKind::kExplicit, num_games)
          : std::make_unique<nn::NNInterface>(num_games, kTimeoutUs, cache_size,
                                              std::move(cur_engine));
  std::unique_ptr<nn::NNInterface> cand_nn_interface =
      cand_uses_search
          ? std::make_unique<nn::NNInterface>(
                cand_batch_size, kTimeoutUs, cache_size, std::move(cand_engine),
                nn::NNInterface::SignalKind::kExplicit, num_games)
          : std::make_unique<nn::NNInterface>(num_games, kTimeoutUs, cache_size,
                                              std::move(cand_engine));

  size_t time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();

  // Game Recorder.
  std::unique_ptr<recorder::GameRecorder> game_recorder =
      recorder::GameRecorder::Create(
          recorder_path, num_games, 10000, 0,
          absl::StrFormat("EVAL_%s_%s", cur_cfg.name, cand_cfg.name));

  // Spawn games.
  EvalConfig config = EvalConfig{cur_cfg, cand_cfg};
  std::vector<std::thread> threads;
  std::vector<std::future<EvalResult>> eval_results;
  for (int game_id = 0; game_id < num_games; ++game_id) {
    std::promise<EvalResult> eval_result;
    eval_results.emplace_back(eval_result.get_future());

    size_t seed = absl::HashOf(time, game_id);
    std::thread thread(PlayEvalGame, seed, game_id, cur_nn_interface.get(),
                       cand_nn_interface.get(),
                       absl::StrFormat("/tmp/eval%d_log.txt", game_id),
                       std::move(eval_result), game_recorder.get(), config);
    threads.emplace_back(std::move(thread));
  }

  LOG(INFO) << "Playing " << num_games << " eval games.";

  for (auto& thread : threads) {
    thread.join();
  }

  int num_cand_won = 0;
  for (auto& eval_result : eval_results) {
    EvalResult res = eval_result.get();
    Winner winner = res.winner;
    num_cand_won += winner == Winner::kCand ? 1 : 0;
  }

  float winrate =
      (static_cast<float>(num_cand_won) / static_cast<float>(num_games));
  float rel_elo = core::RelativeElo(winrate);
  float c95 = ConfidenceDelta(1.96f, num_games, winrate);
  float elo_c95 = core::RelativeElo(.5f + c95);

  LOG(INFO) << "\n--- N, K ---\nCur N: " << cur_cfg.n << ", K: " << cur_cfg.k
            << "\nCand N: " << cand_cfg.n << ", K: " << cand_cfg.k;
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
