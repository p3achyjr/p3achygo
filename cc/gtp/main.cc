#include <iostream>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "cc/eval/player_config.h"
#include "cc/gtp/client.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(std::string, config, "", "Path to player config file.");
ABSL_FLAG(int, n, 100, "Number of visits (overrides config).");
ABSL_FLAG(bool, verbose, false, "Log config on startup.");

// GTP-specific defaults that differ from PlayerSearchConfig defaults.
eval::PlayerSearchConfig GtpDefaultConfig() {
  eval::PlayerSearchConfig cfg;
  cfg.var_scale_cpuct = true;
  cfg.var_scale_prior_visits = 10;
  cfg.p_opt_weight = 1.0f;
  cfg.use_bias_cache = true;
  cfg.bias_cache_alpha = 0.85f;
  cfg.bias_cache_lambda = 0.45f;
  return cfg;
}

void LogConfig(const eval::PlayerSearchConfig& cfg) {
  auto b = [](bool v) { return v ? "true" : "false"; };
  std::cerr << "p3achygo config:\n";
  if (cfg.use_puct) {
    std::cerr << "  n:                    " << cfg.n << "\n";
  } else {
    std::cerr << "  n:                    " << cfg.n << "  k: " << cfg.k
              << "\n";
  }
  std::cerr << "  use_puct:             " << b(cfg.use_puct) << "\n";
  std::cerr << "  c_puct:               " << cfg.c_puct << "\n";
  std::cerr << "  c_puct_visit_scaling: " << cfg.c_puct_visit_scaling << "\n";
  std::cerr << "  var_scale_cpuct:      " << b(cfg.var_scale_cpuct) << "\n";
  if (cfg.var_scale_cpuct) {
    std::cerr << "  var_scale_prior_visits: " << cfg.var_scale_prior_visits
              << "\n";
  }
  std::cerr << "  use_lcb:              " << b(cfg.use_lcb) << "\n";
  std::cerr << "  p_opt_weight:         " << cfg.p_opt_weight << "\n";
  std::cerr << "  use_bias_cache:       " << b(cfg.use_bias_cache) << "\n";
  if (cfg.use_bias_cache) {
    std::cerr << "  bias_cache_alpha:     " << cfg.bias_cache_alpha << "\n";
    std::cerr << "  bias_cache_lambda:    " << cfg.bias_cache_lambda << "\n";
  }
  if (cfg.time_ms != 0) {
    std::cerr << "  time_ms:              " << cfg.time_ms << "\n";
  }
  if (cfg.num_threads_per_game != 1) {
    std::cerr << "  num_threads_per_game: " << cfg.num_threads_per_game << "\n";
    std::cerr << "  q_fn:                 " << cfg.q_fn << "\n";
    std::cerr << "  n_fn:                 " << cfg.n_fn << "\n";
    std::cerr << "  collision_policy:     " << cfg.collision_policy << "\n";
    std::cerr << "  collision_detector:   " << cfg.collision_detector << "\n";
    std::cerr << "  vl_delta:             " << cfg.vl_delta << "\n";
    std::cerr << "  max_collision_retries:" << cfg.max_collision_retries
              << "\n";
    std::cerr << "  search_mode:          " << cfg.search_mode << "\n";
    std::cerr << "  descent_policy:       " << cfg.descent_policy << "\n";
    std::cerr << "  max_o_ratio:          " << cfg.max_o_ratio << "\n";
  }
}

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
  if (model_path.empty()) {
    std::cerr << "No Model Path (--model_path) specified.\n";
    return 1;
  }

  eval::PlayerSearchConfig cfg = GtpDefaultConfig();
  std::string config_path = absl::GetFlag(FLAGS_config);
  if (!config_path.empty()) {
    try {
      cfg = eval::ParsePlayerConfigFile(config_path);
    } catch (const std::exception& e) {
      std::cerr << "Failed to load config: " << e.what() << "\n";
      return 1;
    }
  }
  cfg.n = absl::GetFlag(FLAGS_n);

  if (absl::GetFlag(FLAGS_verbose)) {
    LogConfig(cfg);
  }

  std::unique_ptr<gtp::Client> client = std::make_unique<gtp::Client>();
  absl::Status status = client->Start(model_path, cfg, absl::GetFlag(FLAGS_verbose));
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }

  std::thread input_thread = std::thread(&InputLoop, client.get());
  input_thread.join();
  return 0;
}
