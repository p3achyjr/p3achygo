// Define DOCTEST_CONFIG_IMPLEMENT so we can provide a custom main that parses
// absl flags before running tests.
#define DOCTEST_CONFIG_IMPLEMENT
#include "cc/mcts/search.h"

#include <filesystem>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
#include "cc/mcts/tree.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/engine/go_features.h"
#include "cc/nn/nn_interface.h"
#include "doctest/doctest.h"

// Filters to a single test case by name, e.g.:
//   --test_case="ParallelSearch - real engine"
ABSL_FLAG(std::string, test_case, "", "Run only the test case with this name.");

// Exactly one of these should be set when testing with a real engine.
// The batch size of the provided engine also determines the thread count used
// in the real-engine test case.
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

namespace mcts {
namespace {

using game::Game;
using nn::NNInterface;

// Engine stub: returns uniform probabilities and neutral value estimates.
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

// Returns a pre-evaluated root node satisfying the parallel search invariant
// (root is always kNnEvaluated before Run() is called).
TreeNode* MakeEvaluatedRoot(NodeTable& node_table) {
  TreeNode* root = node_table.GetOrCreate(/*board_hash=*/0, BLACK,
                                          /*is_terminal=*/false);
  root->state.store(TreeNodeState::kNnEvaluated, std::memory_order_release);
  root->color_to_move = BLACK;
  root->n = 1;
  root->w = 0.0f;
  root->v = 0.0f;
  root->init_util_est = 0.0f;
  root->init_outcome_est = 0.0f;
  root->init_score_est = 0.0f;
  root->move_probs.fill(1.0f / constants::kMaxMovesPerPosition);
  root->move_logits.fill(0.0f);
  return root;
}

Search::Params MakeParams(int num_threads, int visit_budget) {
  return Search::Params{
      .num_threads = num_threads,
      .total_visit_budget = visit_budget,
      .total_visit_time_ms = 0,
      .puct_params = PuctParams::Builder()
                        .set_kind(PuctRootSelectionPolicy::kLcb)
                        .build(),
      .q_fn_kind = QFnKind::kIdentity,
      .n_fn_kind = NFnKind::kVirtualVisit,
      .descent_policy_kind = DescentPolicyKind::kDeterministic,
      .collision_policy_kind = CollisionPolicyKind::kSmartRetry,
  };
}

struct RealEngineConfig {
  std::unique_ptr<nn::Engine> engine;
  int num_threads;  // 0 means no real engine was found.
};

// Scans the engine flags and returns the first valid one. At most one should
// be set; if multiple are set, the smallest batch size wins.
RealEngineConfig DetectRealEngine() {
  const std::pair<int, std::string> candidates[] = {
      {1, absl::GetFlag(FLAGS_engine_bs1)},
      {2, absl::GetFlag(FLAGS_engine_bs2)},
      {4, absl::GetFlag(FLAGS_engine_bs4)},
      {8, absl::GetFlag(FLAGS_engine_bs8)},
      {16, absl::GetFlag(FLAGS_engine_bs16)},
  };
  for (auto& [bs, path] : candidates) {
    if (path.empty()) continue;
    if (!std::filesystem::exists(path)) {
      LOG(WARNING) << "Engine path does not exist: " << path;
      continue;
    }
    nn::Engine::Kind kind = nn::KindFromEnginePath(path);
    int version = nn::GetVersionFromModelPath(path);
    return {nn::CreateEngine(kind, path, bs, version), bs};
  }
  return {nullptr, 0};
}

}  // namespace

TEST_CASE("ParallelSearch - terminates and visit count reaches budget") {
  INFO("Running Search 4 Threads 32 Visits");
  static constexpr int kNumThreads = 4;
  static constexpr int kVisitBudget = 32;

  NNInterface nn_interface(kNumThreads, std::make_unique<NullEngine>(),
                           NNInterface::SignalKind::kExplicit,
                           /*num_shared_search_tasks=*/1);
  Search search(nn_interface.MakeSlot(/*task_offset=*/0));
  MctsNodeTable node_table;
  Game game;
  core::Probability probability;

  TreeNode* root = MakeEvaluatedRoot(node_table);
  Search::Result result =
      search.Run(probability, game, &node_table, root, BLACK,
                 MakeParams(kNumThreads, kVisitBudget));

  CHECK(result.num_visits >= kVisitBudget);
  CHECK(result.num_visits < (kVisitBudget + kNumThreads));
  CHECK(result.move != game::kNoopLoc);
  CHECK(root->n > 1);
}

TEST_CASE("ParallelSearch - single thread") {
  static constexpr int kNumThreads = 1;
  static constexpr int kVisitBudget = 16;

  INFO("Running Search 1 Thread 16 Visits");
  NNInterface nn_interface(kNumThreads, std::make_unique<NullEngine>(),
                           NNInterface::SignalKind::kExplicit,
                           /*num_shared_search_tasks=*/1);
  Search search(nn_interface.MakeSlot(/*task_offset=*/0));
  MctsNodeTable node_table;
  Game game;
  core::Probability probability;

  TreeNode* root = MakeEvaluatedRoot(node_table);
  Search::Result result =
      search.Run(probability, game, &node_table, root, BLACK,
                 MakeParams(kNumThreads, kVisitBudget));

  CHECK(result.num_visits >= kVisitBudget);
  CHECK(result.num_visits < (kVisitBudget + kNumThreads));
  CHECK(root->n > 1);
}

TEST_CASE("ParallelSearch - visit budget respected across thread counts") {
  static constexpr int kVisitBudget = 24;

  for (int num_threads : {1, 2, 4}) {
    INFO("Running Search " << num_threads << " Threads 24 Visits");
    NNInterface nn_interface(num_threads, std::make_unique<NullEngine>(),
                             NNInterface::SignalKind::kExplicit,
                             /*num_shared_search_tasks=*/1);
    Search search(nn_interface.MakeSlot(/*task_offset=*/0));
    MctsNodeTable node_table;
    Game game;
    core::Probability probability;

    TreeNode* root = MakeEvaluatedRoot(node_table);
    Search::Result result =
        search.Run(probability, game, &node_table, root, BLACK,
                   MakeParams(num_threads, kVisitBudget));

    CHECK(result.num_visits >= kVisitBudget);
    CHECK(result.num_visits < (kVisitBudget + num_threads));
  }
}

TEST_CASE("ParallelSearch - big budget") {
  INFO("Running Search 16 Threads 10000 Visits");
  static constexpr int kNumThreads = 16;
  static constexpr int kVisitBudget = 10000;

  NNInterface nn_interface(kNumThreads, std::make_unique<NullEngine>(),
                           NNInterface::SignalKind::kExplicit,
                           /*num_shared_search_tasks=*/1);
  Search search(nn_interface.MakeSlot(/*task_offset=*/0));
  MctsNodeTable node_table;
  Game game;
  core::Probability probability;

  TreeNode* root = MakeEvaluatedRoot(node_table);
  Search::Result result =
      search.Run(probability, game, &node_table, root, BLACK,
                 MakeParams(kNumThreads, kVisitBudget));

  CHECK(result.num_visits >= kVisitBudget);
  CHECK(result.num_visits < (kVisitBudget + kNumThreads));
  CHECK(result.move != game::kNoopLoc);
  CHECK(root->n > 1);
}

TEST_CASE("ParallelSearch - real engine") {
  auto [engine, num_threads] = DetectRealEngine();
  if (!engine) {
    // No real engine flag provided; skipping.
    return;
  }

  INFO("Running Search with real engine, " << num_threads << " threads");
  static constexpr int kVisitBudget = 2000;

  NNInterface nn_interface(num_threads, /*timeout=*/0, /*cache_size=*/0,
                           std::move(engine),
                           NNInterface::SignalKind::kExplicit,
                           /*num_shared_search_tasks=*/1);
  Search search(nn_interface.MakeSlot(/*task_offset=*/0));
  MctsNodeTable node_table;
  Game game;
  core::Probability probability;

  TreeNode* root = node_table.GetOrCreate(/*board_hash=*/0, BLACK,
                                          /*is_terminal=*/false);
  Search::Result result =
      search.Run(probability, game, &node_table, root, BLACK,
                 MakeParams(num_threads, kVisitBudget));

  LOG(INFO) << "Selected Move: " << result.move
            << ", Visits: " << result.num_visits
            << ", Aborts: " << result.num_aborted
            << ", Time: " << result.time_ms << "ms";
  CHECK(result.num_visits >= kVisitBudget);
  CHECK(result.num_visits < (kVisitBudget + num_threads));
  CHECK(result.move != game::kNoopLoc);
  CHECK(root->n > 1);
  // Value should be in a plausible range for a real model.
  CHECK(root->v >= -1.5f);
  CHECK(root->v <= 1.5f);
}

}  // namespace mcts

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  doctest::Context ctx;
  std::string tc = absl::GetFlag(FLAGS_test_case);
  if (!tc.empty()) {
    ctx.addFilter("test-case", tc.c_str());
  }
  return ctx.run();
}
