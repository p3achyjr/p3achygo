#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search.h"
#include "cc/mcts/search_policy.h"
#include "cc/mcts/tree.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/engine/go_features.h"
#include "cc/nn/nn_interface.h"

// Paths to TRT engines compiled for each batch size.
// If empty, a NullEngine (uniform output stub) is used instead.
ABSL_FLAG(std::string, engine_bs1, "",
          "Path to TRT engine compiled for batch_size=1.");
ABSL_FLAG(std::string, engine_bs2, "",
          "Path to TRT engine compiled for batch_size=2.");
ABSL_FLAG(std::string, engine_bs4, "",
          "Path to TRT engine compiled for batch_size=4.");
ABSL_FLAG(std::string, engine_bs8, "",
          "Path to TRT engine compiled for batch_size=8.");
ABSL_FLAG(std::string, engine_bs16, "",
          "Path to TRT engine compiled for batch_size=16.");
ABSL_FLAG(int, num_trials, 3, "Number of trials per benchmark configuration.");

namespace {

using game::Game;
using mcts::GumbelEvaluator;
using mcts::MctsNodeTable;
using mcts::PuctParams;
using mcts::PuctRootSelectionPolicy;
using mcts::Search;
using mcts::TreeNode;
using nn::NNInterface;

// Engine stub: returns uniform probabilities and neutral value estimates.
// Eliminates NN inference cost so the benchmark isolates search overhead.
class NullEngine final : public nn::Engine {
 public:
  Kind kind() override { return Kind::kUnknown; }
  std::string path() override { return "null"; }
  void LoadBatch(int, const nn::GoFeatures&) override {}
  void RunInference() override {}
  void GetBatch(int, nn::NNInferResult& result) override {
    result.move_probs.fill(1.0f / constants::kMaxMovesPerPosition);
    result.move_logits.fill(0.0f);
    result.value_probs.fill(1.0f / constants::kNumValueLogits);
    result.score_probs.fill(1.0f / constants::kNumScoreLogits);
  }
  void GetOwnership(int,
                    std::array<float, constants::kNumBoardLocs>& own) override {
    own.fill(0.0f);
  }
};

std::string EnginePathForBatchSize(int batch_size) {
  switch (batch_size) {
    case 1:
      return absl::GetFlag(FLAGS_engine_bs1);
    case 2:
      return absl::GetFlag(FLAGS_engine_bs2);
    case 4:
      return absl::GetFlag(FLAGS_engine_bs4);
    case 8:
      return absl::GetFlag(FLAGS_engine_bs8);
    case 16:
      return absl::GetFlag(FLAGS_engine_bs16);
    default:
      return "";
  }
}

// Returns a real TRT engine if the flag for this batch size is set and the
// file exists, otherwise falls back to NullEngine with a warning.
std::unique_ptr<nn::Engine> MakeEngine(int batch_size) {
  std::string path = EnginePathForBatchSize(batch_size);
  if (path.empty()) {
    LOG(WARNING) << "No engine path provided for batch_size=" << batch_size
                 << "; using NullEngine (no real NN inference).";
    return std::make_unique<NullEngine>();
  }
  if (!std::filesystem::exists(path)) {
    LOG(WARNING) << "Engine path does not exist: " << path
                 << " for batch_size=" << batch_size
                 << "; using NullEngine (no real NN inference).";
    return std::make_unique<NullEngine>();
  }
  nn::Engine::Kind kind = nn::KindFromEnginePath(path);
  int version = nn::GetVersionFromModelPath(path);
  return nn::CreateEngine(kind, path, batch_size, version);
}

Search::Params MakeParams(int num_threads, int visit_budget) {
  return Search::Params{
      .num_threads = num_threads,
      .total_visit_budget = visit_budget,
      .total_visit_time_ms = 0,
      .puct_params = PuctParams{PuctRootSelectionPolicy::kVisitCount},
      .q_fn_kind = mcts::QFnKind::kIdentity,
      .n_fn_kind = mcts::NFnKind::kVirtualVisit,
      .descent_policy_kind = mcts::DescentPolicyKind::kDeterministic,
      .collision_policy_kind = mcts::CollisionPolicyKind::kAbort,
  };
}

std::string LocStr(game::Loc loc) {
  if (loc == game::kPassLoc) return "pass";
  if (loc == game::kNoopLoc) return "none";
  return "(" + std::to_string(loc.i) + "," + std::to_string(loc.j) + ")";
}

using Clock = std::chrono::steady_clock;

// Returns elapsed wall-time in milliseconds.
template <typename F>
double TimeMs(F&& fn) {
  auto t0 = Clock::now();
  fn();
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void PrintHeader() {
  std::cout << "\n"
            << std::left << std::setw(36) << "Benchmark" << std::right
            << std::setw(8) << "visits" << std::setw(8) << "trials"
            << std::setw(10) << "avg_ms" << std::setw(14) << "visits/s"
            << std::setw(10) << "move" << std::setw(10) << "aborts"
            << "\n"
            << std::string(96, '-') << "\n";
}

void PrintResult(const std::string& name, int visits, int trials, double avg_ms,
                 game::Loc move, int num_aborts) {
  double visits_per_sec = visits / (avg_ms / 1000.0);
  std::cout << std::left << std::setw(36) << name << std::right << std::setw(8)
            << visits << std::setw(8) << trials << std::setw(10) << std::fixed
            << std::setprecision(1) << avg_ms << std::setw(14) << std::fixed
            << std::setprecision(0) << visits_per_sec << std::setw(10)
            << LocStr(move) << std::setw(10) << num_aborts << "\n";
}

// Serial baseline: GumbelEvaluator::SearchRootPuct on a single thread.
void BenchSerialPuct(int visit_budget, int num_trials) {
  NNInterface nn_interface(/*num_threads=*/1, /*timeout=*/0, /*cache_size=*/0,
                           MakeEngine(/*batch_size=*/1));
  GumbelEvaluator evaluator(&nn_interface, /*thread_id=*/0);

  double total_ms = 0.0;
  game::Loc last_move = game::kNoopLoc;
  for (int t = 0; t < num_trials; ++t) {
    MctsNodeTable node_table;
    Game game;
    core::Probability probability;
    TreeNode* root = node_table.GetOrCreate(game.board().hash(), BLACK,
                                            /*is_terminal=*/false);
    mcts::GumbelResult result;
    total_ms += TimeMs([&] {
      result = evaluator.SearchRootPuct(
          probability, game, &node_table, root, BLACK, visit_budget,
          PuctParams{PuctRootSelectionPolicy::kLcb});
    });
    last_move = result.mcts_move;
  }

  std::string name = "SerialPuct";
  PrintResult(name, visit_budget, num_trials, total_ms / num_trials, last_move,
              0);
}

// Parallel search: Search::Run with N worker threads.
void BenchParallelSearch(int num_threads, int visit_budget, int num_trials) {
  NNInterface nn_interface(num_threads, /*timeout=*/0, /*cache_size=*/0,
                           MakeEngine(/*batch_size=*/num_threads),
                           NNInterface::SignalKind::kExplicit,
                           /*num_shared_search_tasks=*/1);
  NNInterface::Slot slot = nn_interface.MakeSlot(0);
  Search search(slot);

  double total_ms = 0.0;
  game::Loc last_move = game::kNoopLoc;
  int num_aborts = 0;
  for (int t = 0; t < num_trials; ++t) {
    MctsNodeTable node_table;
    Game game;
    core::Probability probability;
    TreeNode* root =
        node_table.GetOrCreateGuarded(game.board().hash(), BLACK, false);
    Search::Result result;
    total_ms += TimeMs([&] {
      result = search.Run(probability, game, &node_table, root, BLACK,
                          MakeParams(num_threads, visit_budget));
    });
    last_move = result.move;
    num_aborts += result.num_aborted;
  }

  std::string name = "ParallelSearch threads=" + std::to_string(num_threads);
  PrintResult(name, visit_budget, num_trials, total_ms / num_trials, last_move,
              num_aborts / num_trials);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const int num_trials = absl::GetFlag(FLAGS_num_trials);

  PrintHeader();

  for (int visits : {1000, 10000}) {
    BenchSerialPuct(visits, num_trials);
  }

  for (int threads : {1, 2, 4, 8, 16}) {
    for (int visits : {1000, 10000}) {
      BenchParallelSearch(threads, visits, num_trials);
    }
  }

  return 0;
}
