#include <cstring>
#include <iostream>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "cc/eval/player_config.h"
#include "cc/gtp/client.h"

ABSL_FLAG(std::string, model_path, "", "Path to the TensorRT engine (.trt).");
ABSL_FLAG(std::string, config, "", "Path to a player config file.");
ABSL_FLAG(int, n, 0, "Visit budget per move; overrides `n` in the config.");
ABSL_FLAG(int, analyze_max_visits, 1000000,
          "lz-/kata-analyze stop at this many root visits (<= 0: no limit).");
ABSL_FLAG(int, analyze_max_time_s, 60,
          "lz-/kata-analyze stop after this many seconds (<= 0: no limit).");
ABSL_FLAG(bool, verbose, false,
          "Log the config at startup and per-move search details (stderr).");

namespace {
constexpr char kUsage[] =
    R"(p3achygo: a Go engine that speaks GTP on stdin/stdout.

Usage:
  p3achygo --model_path=<engine.trt> [--config=<player.cfg>] [--n=<visits>]
           [--analyze_max_visits=<visits>] [--analyze_max_time_s=<s>]
           [--verbose]

Flags:
  --model_path  TensorRT engine (.trt) to play with. Required.
  --config      Player config: one `key: value` per line; lines starting
                with `#` are comments.
                Unset keys keep their defaults; unknown keys are ignored.
                All keys are in cc/eval/player_config.h. Common ones:
                  time_ms: auto | 0 | <ms>  auto = budget from the GTP clock
                                            (time_settings / time_left;
                                            1 s per move without one);
                                            0 = fixed visit budget `n`;
                                            <ms> = fixed time per move
                  n: <int>                  visit budget per move when
                                            time_ms is 0
                  n_counts_reused_visits: true | false
                                            true: stop at n root visits,
                                            counting the reused tree;
                                            false: n new playouts
                  num_threads_per_game: <int>
                  enable_pondering: true | false
                  ponder_max_tree_mb: <int> 0 = size to fit in RAM
                  territory_komi: true | false
                                            true: GTP `komi` is territory
                                            komi; search with komi + 1
                                            (Japanese 6.5 -> area 7.5)
                  gtp_name: <string>        GTP `name` reply (default
                                            p3achygo). LizzieYzy uses
                                            kata-analyze if it starts
                                            with "kata".
  --n           Visit budget per move; overrides `n` in the config.
  --analyze_max_visits, --analyze_max_time_s
                lz-analyze / kata-analyze (e.g. LizzieYzy) stop at this
                many root visits or seconds, whichever comes first.
                Defaults 1000000 and 60; <= 0 disables a limit.
  --verbose     Log the config at startup, and each move's search
                details, to stderr.

Example:
  p3achygo --model_path=model.trt --config=tournament.cfg --verbose

Once running, send `list_commands` for the supported GTP commands.
--helpfull lists every flag from every linked library.
)";

bool WantsHelp(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-h") == 0 ||
        std::strcmp(argv[i], "--help") == 0 ||
        std::strcmp(argv[i], "-help") == 0) {
      return true;
    }
  }
  return false;
}
}  // namespace

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
  std::cerr << "  sampling_num_moves: " << cfg.sampling_num_moves
            << "  sampling_temperature: " << cfg.sampling_temperature << "\n";
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
  std::cerr << "  root_p_opt_weight:    " << cfg.root_p_opt_weight << "\n";
  std::cerr << "  p_opt_geometric:      " << cfg.p_opt_geometric << "\n";
  std::cerr << "  value_weight_exponent:" << cfg.value_weight_exponent << "\n";
  std::cerr << "  root_fpu:             " << cfg.root_fpu << "\n";
  std::cerr << "  round_free:           " << cfg.round_free << "\n";
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
  // absl's --help lists the flags of every linked library; ours is short.
  if (WantsHelp(argc, argv)) {
    std::cout << kUsage;
    return 0;
  }
  absl::ParseCommandLine(argc, argv);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    std::cerr << "No model path (--model_path) specified. See --help.\n";
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
  const int n = absl::GetFlag(FLAGS_n);
  if (n != 0) {
    cfg.n = absl::GetFlag(FLAGS_n);
  }
  cfg.analyze_max_visits = absl::GetFlag(FLAGS_analyze_max_visits);
  cfg.analyze_max_time_s = absl::GetFlag(FLAGS_analyze_max_time_s);

  if (absl::GetFlag(FLAGS_verbose)) {
    LogConfig(cfg);
  }

  std::unique_ptr<gtp::Client> client = std::make_unique<gtp::Client>();
  absl::Status status =
      client->Start(model_path, cfg, absl::GetFlag(FLAGS_verbose));
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }

  std::thread input_thread = std::thread(&InputLoop, client.get());
  input_thread.join();
  return 0;
}
