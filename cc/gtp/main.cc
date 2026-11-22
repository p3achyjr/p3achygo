#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "cc/eval/player_config.h"
#include "cc/gtp/client.h"
#include "cc/gtp/command.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(std::string, config, "", "Path to player config file.");
ABSL_FLAG(int, n, 0, "Number of visits (overrides config).");
ABSL_FLAG(bool, verbose, false, "Log config on startup.");
ABSL_FLAG(bool, batch, false,
          "Batch GTP mode: the GTP id of each command is a game id. Commands "
          "for different games run concurrently and share one batched NN "
          "engine; commands without an id are global (name, version, "
          "protocol_version, list_commands, known_command, quit).");
ABSL_FLAG(int, batch_games, 16, "Max concurrent games in --batch mode.");
ABSL_FLAG(int, nn_timeout_us, 1000,
          "--batch mode: NN batch timeout in microseconds.");

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

// Batch GTP mode. One shared NNInterface sized for `max_games` games; one
// Service + Client per game id, created on first use; responses are written
// atomically through a shared sink. A game's `quit` retires that game only;
// an id-less `quit` exits. Streaming analysis commands are rejected.
int BatchLoop(const std::string& model_path, eval::PlayerSearchConfig cfg,
              bool verbose, int max_games, int64_t nn_timeout_us) {
  cfg.enable_pondering = false;

  const int threads_per_game = cfg.num_threads_per_game;
  const bool uses_psearch = threads_per_game > 1 || cfg.time_ms != 0;
  const int num_threads = max_games * threads_per_game;
  std::unique_ptr<nn::Engine> engine =
      nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                       num_threads, nn::GetVersionFromModelPath(model_path));
  // Mirrors cc/eval: kExplicit with one shared search task per game for
  // parallel search, kAuto for serial Gumbel/PUCT.
  std::unique_ptr<nn::NNInterface> nn_interface =
      uses_psearch ? std::make_unique<nn::NNInterface>(
                         num_threads, nn_timeout_us, 16384, std::move(engine),
                         nn::NNInterface::SignalKind::kExplicit, max_games)
                   : std::make_unique<nn::NNInterface>(
                         num_threads, nn_timeout_us, 16384, std::move(engine));
  nn_interface->SetDegenerateReeval(cfg.nn_reeval_max_logit,
                                    cfg.nn_reeval_value_disagree,
                                    cfg.nn_reeval_max_retries);
  if (!uses_psearch) {
    // Games register their thread only while searching (see
    // Service::CreateSharedService), so start with nothing registered.
    for (int i = 0; i < num_threads; ++i) nn_interface->UnregisterThread(i);
  }

  std::mutex out_mu;
  auto sink = [&out_mu](const std::string& s) {
    std::lock_guard<std::mutex> l(out_mu);
    std::cout << s << std::flush;
  };

  struct Game {
    std::unique_ptr<gtp::Client> client;
    int slot;
  };
  std::map<int, Game> games;
  // Games whose quit has been queued; their slots are freed once the client
  // has processed it.
  std::vector<Game> retired;
  std::vector<int> free_slots;
  for (int i = max_games - 1; i >= 0; --i) free_slots.push_back(i);
  auto release = [&](Game& g) {
    g.client.reset();  // joins the client's threads
    free_slots.push_back(g.slot);
  };
  auto reap_retired = [&]() {
    std::vector<Game> keep;
    for (Game& g : retired) {
      if (!g.client->Running()) {
        release(g);
      } else {
        keep.push_back(std::move(g));
      }
    }
    retired = std::move(keep);
  };
  // Frees a slot, waiting for the oldest retiring game to finish its quit if
  // none is free yet. Returns false if there is no retiring game either.
  auto ensure_free_slot = [&]() {
    reap_retired();
    if (!free_slots.empty()) return true;
    if (retired.empty()) return false;
    release(retired.front());  // blocks until its queued quit is processed
    retired.erase(retired.begin());
    return true;
  };

  std::string line;
  while (std::getline(std::cin, line)) {
    absl::string_view sv = absl::StripAsciiWhitespace(line);
    if (sv.empty() || sv[0] == '#') continue;
    std::vector<absl::string_view> tokens =
        absl::StrSplit(sv, absl::ByAnyChar(" \t"), absl::SkipEmpty());
    int id;
    const bool has_id = absl::SimpleAtoi(tokens[0], &id);

    if (!has_id) {
      gtp::GTPCode code = gtp::StringToGTPCode(std::string(tokens[0]));
      switch (code) {
        case gtp::GTPCode::kQuit:
          sink("= \n\n");
          goto done;
        case gtp::GTPCode::kProtocolVersion:
          sink("= 2\n\n");
          break;
        case gtp::GTPCode::kName:
          sink("= p3achygo\n\n");
          break;
        case gtp::GTPCode::kVersion:
          sink("= \n\n");
          break;
        case gtp::GTPCode::kKnownCommand: {
          const bool known = tokens.size() == 2 &&
                             gtp::StringToGTPCode(std::string(tokens[1])) !=
                                 gtp::GTPCode::kUnknownCommand;
          sink(known ? "= true\n\n" : "= false\n\n");
          break;
        }
        case gtp::GTPCode::kListCommands: {
          std::string ls;
          for (const auto& c : gtp::kSupportedCommands) {
            if (!ls.empty()) ls += "\n";
            ls += gtp::GTPCodeToString(c);
          }
          sink("= " + ls + "\n\n");
          break;
        }
        default:
          sink("? batch mode: command needs a game id\n\n");
      }
      continue;
    }

    const std::string id_str(tokens[0]);
    if (tokens.size() < 2) {
      sink("?" + id_str + " syntax error\n\n");
      continue;
    }
    gtp::GTPCode code = gtp::StringToGTPCode(std::string(tokens[1]));
    if (code == gtp::GTPCode::kAnalyze ||
        code == gtp::GTPCode::kGenMoveAnalyze ||
        code == gtp::GTPCode::kCgosGenmoveAnalyze) {
      sink("?" + id_str + " command not supported in batch mode\n\n");
      continue;
    }

    auto it = games.find(id);
    if (it == games.end()) {
      if (!ensure_free_slot()) {
        sink("?" + id_str + " too many games (--batch_games " +
             std::to_string(max_games) + ")\n\n");
        continue;
      }
      Game g;
      g.slot = free_slots.back();
      free_slots.pop_back();
      g.client = std::make_unique<gtp::Client>();
      g.client->Start(gtp::Service::CreateSharedService(
                          nn_interface.get(), g.slot * threads_per_game, cfg,
                          verbose, "BatchGame " + id_str),
                      sink);
      it = games.emplace(id, std::move(g)).first;
    }
    gtp::InputLoopStatus status =
        it->second.client->ParseAndAddCommand(std::string(sv));
    if (status == gtp::InputLoopStatus::kStop) {
      // Retire now so a later command with this id starts a fresh game.
      retired.push_back(std::move(it->second));
      games.erase(it);
    }
  }
done:
  // Live games never got a quit: stop them without a response.
  for (auto& kv : games) {
    kv.second.client->Stop();
    release(kv.second);
  }
  games.clear();
  for (Game& g : retired) release(g);
  retired.clear();
  return 0;
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
  const int n = absl::GetFlag(FLAGS_n);
  if (n != 0) {
    cfg.n = absl::GetFlag(FLAGS_n);
  }

  if (absl::GetFlag(FLAGS_verbose)) {
    LogConfig(cfg);
  }

  if (absl::GetFlag(FLAGS_batch)) {
    return BatchLoop(model_path, cfg, absl::GetFlag(FLAGS_verbose),
                     absl::GetFlag(FLAGS_batch_games),
                     absl::GetFlag(FLAGS_nn_timeout_us));
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
