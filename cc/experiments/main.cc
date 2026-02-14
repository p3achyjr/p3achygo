#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "cc/experiments/play_games.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(int, num_games, 1, "Number of games");
ABSL_FLAG(int, visit_count, 400, "Visits Per Move");

constexpr size_t kCacheSize = 16384;
static constexpr int64_t kTimeoutUs = 4000;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path == "") {
    LOG(ERROR) << "--model_path Not Specified.";
    return 1;
  }

  int num_games = absl::GetFlag(FLAGS_num_games);
  int visit_count = absl::GetFlag(FLAGS_visit_count);

  std::unique_ptr<nn::Engine> engine =
      nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                       num_games, nn::GetVersionFromModelPath(model_path));
  std::unique_ptr<nn::NNInterface> nn_interface =
      std::make_unique<nn::NNInterface>(num_games, kTimeoutUs, kCacheSize,
                                        std::move(engine));

  // callbacks
  BiasCallback bias_cb;
  std::vector<Callback*> cbs = {&bias_cb};
  PlayGames(nn_interface.get(), num_games, visit_count, cbs);
}
