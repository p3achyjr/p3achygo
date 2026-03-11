#include <filesystem>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search.h"
#include "cc/mcts/search_policy.h"
#include "cc/mcts/tree.h"
#include "cc/nn/engine/engine.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/engine/go_features.h"
#include "cc/nn/nn_interface.h"

ABSL_FLAG(std::string, engine_p0, "",
          "Path to TRT engine for player 0 (black). Empty -> NullEngine.");
ABSL_FLAG(std::string, engine_p1, "",
          "Path to TRT engine for player 1 (white). Empty -> NullEngine.");
ABSL_FLAG(int, threads_p0, 4, "Thread count for player 0.");
ABSL_FLAG(int, threads_p1, 4, "Thread count for player 1.");
ABSL_FLAG(int, time_ms, 3000, "Time per move in milliseconds.");
ABSL_FLAG(std::string, q_fn, "virtual_loss",
          "Q function: identity | virtual_loss | virtual_loss_soft");
ABSL_FLAG(std::string, n_fn, "virtual_visit",
          "N function: identity | virtual_visit");
ABSL_FLAG(std::string, collision_policy, "abort",
          "Collision policy: abort | retry | smart_retry");

namespace {

using namespace ::mcts;
using namespace ::nn;

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

std::unique_ptr<nn::Engine> MakeEngine(const std::string& path,
                                       int batch_size) {
  if (path.empty()) {
    LOG(WARNING) << "No engine path provided; using NullEngine.";
    return std::make_unique<NullEngine>();
  }
  if (!std::filesystem::exists(path)) {
    LOG(WARNING) << "Engine path not found: " << path << "; using NullEngine.";
    return std::make_unique<NullEngine>();
  }
  nn::Engine::Kind kind = nn::KindFromEnginePath(path);
  int version = nn::GetVersionFromModelPath(path);
  return nn::CreateEngine(kind, path, batch_size, version);
}

mcts::QFnKind ParseQFnKind(const std::string& s) {
  if (s == "identity") return mcts::QFnKind::kIdentity;
  if (s == "virtual_loss_soft") return mcts::QFnKind::kVirtualLossSoft;
  return mcts::QFnKind::kVirtualLoss;
}

mcts::NFnKind ParseNFnKind(const std::string& s) {
  if (s == "identity") return mcts::NFnKind::kIdentity;
  return mcts::NFnKind::kVirtualVisit;
}

mcts::CollisionPolicyKind ParseCollisionPolicyKind(const std::string& s) {
  if (s == "retry") return mcts::CollisionPolicyKind::kRetry;
  if (s == "smart_retry") return mcts::CollisionPolicyKind::kSmartRetry;
  return mcts::CollisionPolicyKind::kAbort;
}

Search::Params MakeParams(int num_threads, int time_ms, mcts::QFnKind q_fn_kind,
                          mcts::NFnKind n_fn_kind,
                          mcts::CollisionPolicyKind collision_policy_kind) {
  return Search::Params{
      .num_threads = num_threads,
      .total_visit_budget = 1 << 20,  // large ceiling; time is the real limit
      .total_visit_time_ms = time_ms,
      .puct_params = PuctParams{PuctRootSelectionPolicy::kVisitCount},
      .q_fn_kind = q_fn_kind,
      .n_fn_kind = n_fn_kind,
      .descent_policy_kind = mcts::DescentPolicyKind::kDeterministic,
      .collision_policy_kind = collision_policy_kind,
      .collision_detector_kind = mcts::CollisionDetectorKind::kNoOp,
  };
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const int threads_p0 = absl::GetFlag(FLAGS_threads_p0);
  const int threads_p1 = absl::GetFlag(FLAGS_threads_p1);
  const int time_ms = absl::GetFlag(FLAGS_time_ms);
  const mcts::QFnKind q_fn_kind = ParseQFnKind(absl::GetFlag(FLAGS_q_fn));
  const mcts::NFnKind n_fn_kind = ParseNFnKind(absl::GetFlag(FLAGS_n_fn));
  const mcts::CollisionPolicyKind collision_policy_kind =
      ParseCollisionPolicyKind(absl::GetFlag(FLAGS_collision_policy));

  NNInterface nn_p0(threads_p0, /*timeout=*/0, /*cache_size=*/0,
                    MakeEngine(absl::GetFlag(FLAGS_engine_p0), threads_p0),
                    NNInterface::SignalKind::kExplicit,
                    /*num_shared_search_tasks=*/1);
  NNInterface nn_p1(threads_p1, /*timeout=*/0, /*cache_size=*/0,
                    MakeEngine(absl::GetFlag(FLAGS_engine_p1), threads_p1),
                    NNInterface::SignalKind::kExplicit,
                    /*num_shared_search_tasks=*/1);

  Search search_p0(nn_p0.MakeSlot(0));
  Search search_p1(nn_p1.MakeSlot(0));

  game::Game game;
  core::Probability probability;
  McgsNodeTable table_p0;
  McgsNodeTable table_p1;
  game::Color color = BLACK;

  TreeNode* root_p0 =
      table_p0.GetOrCreateGuarded(game.board().hash(), BLACK, false);
  TreeNode* root_p1 =
      table_p1.GetOrCreateGuarded(game.board().hash(), BLACK, false);

  LOG(INFO) << "Starting game: p0 (BLACK) vs p1 (WHITE)\n"
            << "  threads: " << threads_p0 << " vs " << threads_p1 << "\n"
            << "  time per move: " << time_ms << "ms\n"
            << "  q_fn=" << absl::GetFlag(FLAGS_q_fn)
            << "  n_fn=" << absl::GetFlag(FLAGS_n_fn)
            << "  collision=" << absl::GetFlag(FLAGS_collision_policy)
            << "\n\n";

  int move_num = 0;
  while (!game.IsGameOver()) {
    ++move_num;

    Search& active_search = (color == BLACK) ? search_p0 : search_p1;
    TreeNode* active_root = (color == BLACK) ? root_p0 : root_p1;
    mcts::NodeTable* active_table =
        (color == BLACK) ? static_cast<mcts::NodeTable*>(&table_p0)
                         : static_cast<mcts::NodeTable*>(&table_p1);

    Search::Result result = active_search.Run(
        probability, game, active_table, active_root, color,
        MakeParams((color == BLACK) ? threads_p0 : threads_p1, time_ms,
                   q_fn_kind, n_fn_kind, collision_policy_kind));

    game::Loc move = result.move;
    game.PlayMove(move, color);
    LOG(INFO) << "Move " << move_num << " " << (color == BLACK ? "B" : "W")
              << ": " << move << "  visits=" << result.num_visits
              << "  aborted=" << result.num_aborted
              << "  time=" << result.time_ms << "ms"
              << "  v=" << active_root->v << ", board:\n"
              << game::ToString(game.board().position()) << "\n";
    color = game::OppositeColor(color);

    // Advance both trees to the new position.
    TreeNode* next_p0 = root_p0->children[move].load();
    if (!next_p0) {
      next_p0 = table_p0.GetOrCreateGuarded(game.board().hash(), color, false);
    }
    TreeNode* next_p1 = root_p1->children[move].load();
    if (!next_p1) {
      next_p1 = table_p1.GetOrCreateGuarded(game.board().hash(), color, false);
    }

    table_p0.Reap(next_p0);
    table_p1.Reap(next_p1);
    root_p0 = next_p0;
    root_p1 = next_p1;
  }

  LOG(INFO) << "\nGame over after " << move_num << " moves.\n";
  game.WriteResult();

  const game::Game::Result result = game.result();
  LOG(INFO) << "Result: " << (result.winner == BLACK ? "B" : "W") << "+"
            << (result.by_resign
                    ? "Resign"
                    : std::to_string(std::abs(result.bscore - result.wscore)))
            << "  (B=" << result.bscore << " W=" << result.wscore << ")\n";
  return 0;
}
