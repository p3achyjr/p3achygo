#include <limits>
// Define DOCTEST_CONFIG_IMPLEMENT so we can provide a custom main that parses
// absl flags before running tests.
#define DOCTEST_CONFIG_IMPLEMENT
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
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
      .puct_params =
          PuctParams::Builder().set_kind(PuctRootSelectionPolicy::kLcb).build(),
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

// INST pipelined search: deterministic position-dependent fake net. Policy,
// value, score and error are pseudo-random functions of (board, color), so
// subtrees differ and the bottom-up stat check is not trivially uniform.
class HashEngine final : public nn::Engine {
 public:
  // Policy logits are 4 * U[0, 1), plus `logit_scale` on one hashed move;
  // larger values concentrate the policy (more threads on the same path,
  // more collisions).
  HashEngine(int batch_size, float logit_scale)
      : h_(batch_size), logit_scale_(logit_scale) {}
  Kind kind() override { return Kind::kUnknown; }
  std::string path() override { return "hash"; }
  void LoadBatch(int i, const nn::GoFeatures& f) override {
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](uint64_t x) {
      h ^= x;
      h *= 1099511628211ull;
    };
    for (const auto c : f.board) mix(static_cast<uint64_t>(c + 3));
    mix(static_cast<uint64_t>(f.color + 3));
    h_[i] = h | 1;
  }
  void RunInference() override {}
  void GetBatch(int i, nn::NNInferResult& r) override {
    uint64_t st = h_[i];
    const auto next = [&st]() {
      st ^= st << 13;
      st ^= st >> 7;
      st ^= st << 17;
      return static_cast<float>((st >> 11) * (1.0 / 9007199254740992.0));
    };
    float sum = 0;
    const int hot =
        int(next() * BOARD_LEN * BOARD_LEN) % (BOARD_LEN * BOARD_LEN);
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      r.move_logits[a] = 4.0f * next();
      if (a == hot) r.move_logits[a] += logit_scale_;
      r.move_probs[a] = std::exp(r.move_logits[a]);
      sum += r.move_probs[a];
    }
    for (auto& p : r.move_probs) p /= sum;
    r.opt_move_probs = r.move_probs;
    const float u = 0.05f + 0.9f * next();
    r.value_probs.fill(0.0f);
    r.value_probs[0] = u;
    r.value_probs[constants::kNumValueLogits - 1] += 1.0f - u;
    r.score_probs.fill(0.0f);
    r.score_probs[std::min(constants::kNumScoreLogits - 1,
                           int(next() * constants::kNumScoreLogits))] = 1.0f;
    r.err2_outcome = 0.1f * next();
    r.mcts_value_dist.fill(1.0f / constants::kNumVBuckets);
  }
  void GetOwnership(int,
                    std::array<float, constants::kNumBoardLocs>& own) override {
    own.fill(0.0f);
  }

 private:
  std::vector<uint64_t> h_;
  const float logit_scale_;
};

// Reachable nodes in post-order (children before parents).
void PostOrder(TreeNode* node, std::unordered_set<TreeNode*>* seen,
               std::vector<TreeNode*>* out) {
  if (node == nullptr || !seen->insert(node).second) return;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    PostOrder(node->child(a), seen, out);
  }
  out->push_back(node);
}

struct TreeCheck {
  int nodes = 0, bad_in_flight = 0, bad_pending = 0, bad_n = 0, bad_vcat = 0;
  int64_t stat_nodes = 0;
  double max_dv = 0, sum_dv = 0, max_dvo = 0, sum_dvo = 0, max_ds = 0,
         sum_ds = 0, max_de = 0, sum_de = 0;
};

// Structural invariants after a search quiesces, plus an exact bottom-up
// recompute of v / v_outcome / score / v_err (same float formula and order
// as RecomputeNodeStats with obs_bias = 0) compared against the stored
// values. `mcts`: tree (not graph), so n == 1 + sum(child_visits) exactly.
TreeCheck CheckTree(TreeNode* root, bool mcts) {
  std::unordered_set<TreeNode*> seen;
  std::vector<TreeNode*> order;
  PostOrder(root, &seen, &order);
  struct Exp {
    float v, vo, s, e;
  };
  std::unordered_map<TreeNode*, Exp> exp;
  TreeCheck c;
  for (TreeNode* node : order) {
    c.nodes++;
    if (node->n_in_flight.load() != 0 ||
        node->n_in_flight_group[0].load() != 0 ||
        node->n_in_flight_group[1].load() != 0) {
      c.bad_in_flight++;
    }
    const TreeNodeState st = node->state.load();
    if (st == TreeNodeState::kPending && !node->is_terminal) c.bad_pending++;
    if (node->n == 0) {
      exp[node] = {node->v, node->v_outcome, node->score, node->v_err};
      continue;
    }
    int64_t sum_cv = 0;
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      sum_cv += node->child_visits[a];
    }
    if (!node->is_terminal) {
      if (mcts ? (node->n != 1 + sum_cv) : (node->n < 1 + sum_cv)) c.bad_n++;
      int64_t sum_vcat = 0;
      for (auto x : node->v_categorical) sum_vcat += x;
      // Every backup step through the node adds one bucket count; the root's
      // own eval (FetchRootEval, n = 1) adds none.
      if (sum_vcat != node->n - (node == root ? 1 : 0)) c.bad_vcat++;
    }
    if (node->is_terminal) {
      exp[node] = {node->v, node->v_outcome, node->score, node->v_err};
      continue;
    }
    Exp e;
    if (sum_cv == 0) {
      // BackupStep's leaf branch does not write `score` (stays 0).
      e = {node->init_util_est, node->init_outcome_est, 0.0f,
           node->init_err_est};
    } else {
      float w = node->init_util_est, w_outcome = node->init_outcome_est,
            total_score = node->init_score_est, w_err = node->init_err_est;
      for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
        const TreeNode* child = node->child(a);
        if (child == nullptr) continue;
        const Exp& ce = exp.at(const_cast<TreeNode*>(child));
        w -= (node->child_visits[a] * ce.v);
        w_outcome -= (node->child_visits[a] * ce.vo);
        total_score -= (node->child_visits[a] * ce.s);
        w_err += node->child_visits[a] * ce.e;
      }
      e = {w / node->n, w_outcome / node->n, total_score / node->n,
           w_err / node->n};
    }
    exp[node] = e;
    const double dv = std::abs(double(node->v) - e.v),
                 dvo = std::abs(double(node->v_outcome) - e.vo),
                 ds = std::abs(double(node->score) - e.s),
                 de = std::abs(double(node->v_err) - e.e);
    c.stat_nodes++;
    c.max_dv = std::max(c.max_dv, dv);
    c.sum_dv += dv;
    c.max_dvo = std::max(c.max_dvo, dvo);
    c.sum_dvo += dvo;
    c.max_ds = std::max(c.max_ds, ds);
    c.sum_ds += ds;
    c.max_de = std::max(c.max_de, de);
    c.sum_de += de;
  }
  return c;
}

// Plays `num_moves` seeded random legal moves (no passes).
void RandomOpening(Game& game, int num_moves, uint32_t seed,
                   game::Color* to_move) {
  std::mt19937 rng(seed);
  game::Color color = BLACK;
  for (int i = 0; i < num_moves; ++i) {
    std::vector<int> legal;
    for (int a = 0; a < BOARD_LEN * BOARD_LEN; ++a) {
      if (game.IsValidMove(a, color)) legal.push_back(a);
    }
    if (legal.empty()) break;
    const int a = legal[rng() % legal.size()];
    CHECK(game.PlayMove(game::AsLoc(a), color));
    color = game::OppositeColor(color);
  }
  *to_move = color;
}

struct PipeSetup {
  std::unique_ptr<NNInterface> nn1, nn2;
  std::unique_ptr<Search> search;
};

// pipelined: two NNInterfaces of num_threads / 2 rows (as the GTP service
// builds them with P3_PIPELINE=1); otherwise one of num_threads rows.
// shared_infer_threads > 0 (pipelined only): one shared NNInterface with two
// waves of num_threads / 2 rows and that many inference threads (EXP, as
// P3_SHARED_NN=1).
PipeSetup MakePipeSetup(int num_threads, bool pipelined,
                        NNInterface::WakeStrategy wake, bool hash_engine,
                        float logit_scale = 0.0f, int shared_infer_threads = 0,
                        int rf_batch = 0, int rf_min_fill = 1,
                        int64_t rf_timeout_us = 0) {
  PipeSetup p;
  // EXP round-free: one queue-mode interface, engine batch rf_batch.
  if (rf_batch > 0) {
    std::unique_ptr<nn::Engine> engine =
        hash_engine
            ? std::unique_ptr<nn::Engine>(
                  std::make_unique<HashEngine>(rf_batch, logit_scale))
            : std::unique_ptr<nn::Engine>(std::make_unique<NullEngine>());
    p.nn1 = std::make_unique<NNInterface>(num_threads, 512 * num_threads,
                                          std::move(engine), rf_batch,
                                          rf_min_fill, rf_timeout_us, wake);
    // EXP cache variants, as the GTP env flags.
    const char* key = std::getenv("P3_NN_KEY");
    if (key != nullptr && std::string(key) == "hashko") {
      p.nn1->SetNumCacheLastMoves(0);
      p.nn1->SetCacheKeyKo(true);
    }
    const char* shared = std::getenv("P3_SHARED_CACHE");
    if (shared != nullptr && std::string(shared) == "1") {
      p.nn1->EnableSharedCache(512 * num_threads);
    }
    if (shared != nullptr && std::string(shared) == "kata") {
      const char* l2 = std::getenv("P3_KCACHE_LOG2");
      const char* m2 = std::getenv("P3_KCACHE_MUTEX_LOG2");
      p.nn1->EnableKataCache(l2 ? std::atoi(l2) : 20, m2 ? std::atoi(m2) : 16);
    }
    p.search = std::make_unique<Search>(p.nn1->MakeSlot(0));
    return p;
  }
  const int rows = pipelined ? num_threads / 2 : num_threads;
  const auto make_engine = [&]() {
    return hash_engine
               ? std::unique_ptr<nn::Engine>(
                     std::make_unique<HashEngine>(rows, logit_scale))
               : std::unique_ptr<nn::Engine>(std::make_unique<NullEngine>());
  };
  if (pipelined && shared_infer_threads > 0) {
    std::vector<std::unique_ptr<nn::Engine>> engines;
    engines.push_back(make_engine());
    engines.push_back(make_engine());
    p.nn1 = std::make_unique<NNInterface>(
        num_threads, std::numeric_limits<int64_t>::max(), 2 * 16384,
        std::move(engines), NNInterface::SignalKind::kExplicit,
        /*num_shared_search_tasks=*/1, wake, shared_infer_threads);
    p.search = std::make_unique<Search>(p.nn1->MakeSlot(0));
    p.search->SetPipelineSlot(p.nn1->MakeSlot(rows));
    return p;
  }
  const auto make = [&]() {
    std::unique_ptr<nn::Engine> engine = make_engine();
    return std::make_unique<NNInterface>(
        rows, std::numeric_limits<int64_t>::max(), 16384, std::move(engine),
        NNInterface::SignalKind::kExplicit, /*num_shared_search_tasks=*/1,
        wake);
  };
  p.nn1 = make();
  p.search = std::make_unique<Search>(p.nn1->MakeSlot(0));
  if (pipelined) {
    p.nn2 = make();
    p.search->SetPipelineSlot(p.nn2->MakeSlot(0));
  }
  return p;
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

// INST: stress for the barrier / wake experiments. Mirrors the GTP setup
// (infinite batching timeout, 16384-entry cache, kExplicit, one search task),
// sweeps thread counts, collision policies, wake strategies and stop modes.
// Run with P3_BARRIER=gen and without. Invariants per search:
//  - every round consumes exactly one slot per worker: (visits + aborted) % n
//  == 0
//  - every visit backs up through the root exactly once: root->n == 1 + visits
//  - no virtual loss left behind: root->n_in_flight == 0
TEST_CASE("ParallelSearch - INST barrier stress") {
  for (int num_threads : {2, 7, 32, 48, 64}) {
    for (auto wake : {NNInterface::WakeStrategy::kMutex,
                      NNInterface::WakeStrategy::kGenCounter}) {
      for (auto cp :
           {CollisionPolicyKind::kAbort, CollisionPolicyKind::kSmartRetry}) {
        for (int time_ms : {0, 30}) {
          for (int rep = 0; rep < 2; ++rep) {
            NNInterface nn_interface(num_threads,
                                     std::numeric_limits<int64_t>::max(), 16384,
                                     std::make_unique<NullEngine>(),
                                     NNInterface::SignalKind::kExplicit,
                                     /*num_shared_search_tasks=*/1, wake);
            Search search(nn_interface.MakeSlot(/*task_offset=*/0));
            MctsNodeTable node_table;
            Game game;
            core::Probability probability;
            TreeNode* root = MakeEvaluatedRoot(node_table);
            Search::Params params =
                MakeParams(num_threads, time_ms > 0 ? (1 << 30) : 3000);
            params.total_visit_time_ms = time_ms;
            params.q_fn_kind = QFnKind::kVirtualLossSoft;
            params.collision_policy_kind = cp;
            Search::Result result =
                search.Run(probability, game, &node_table, root, BLACK, params);
            INFO("threads=" << num_threads << " wake=" << int(wake)
                            << " cp=" << int(cp) << " time_ms=" << time_ms
                            << " visits=" << result.num_visits
                            << " aborted=" << result.num_aborted);
            CHECK(result.num_visits > 0);
            CHECK((result.num_visits + result.num_aborted) % num_threads == 0);
            CHECK(root->n == 1 + (int)result.num_visits);
            CHECK(root->n_in_flight.load() == 0);
          }
        }
      }
    }
  }
}

// INST pipelined search stress: group sizes 1..32 (threads 2..64), both
// wakers, MCTS and MCGS tables, stop by visit budget, by time, and by an
// external StopSearch() from another thread (as ponder stop does). Checks
// that every search returns (no deadlock) and the structural invariants.
TEST_CASE("ParallelSearch - INST pipeline stress") {
  const char* reps_env = std::getenv("P3_PIPE_REPS");
  const int reps = reps_env ? std::atoi(reps_env) : 2;
  int64_t searches = 0, total_visits = 0, total_aborted = 0,
          vcat_mismatch_nodes = 0;
  for (int group : {1, 4, 16, 32}) {
    const int num_threads = 2 * group;
    for (int shared : {0, 1, 2}) {
      // A shared interface needs more than one row per wave.
      if (shared > 0 && group == 1) continue;
      for (auto wake : {NNInterface::WakeStrategy::kMutex,
                        NNInterface::WakeStrategy::kGenCounter}) {
        for (bool mcgs : {false, true}) {
          for (int stop_mode = 0; stop_mode < 3; ++stop_mode) {
            for (int rep = 0; rep < reps; ++rep) {
              PipeSetup p = MakePipeSetup(num_threads, /*pipelined=*/true, wake,
                                          /*hash_engine=*/true,
                                          /*logit_scale=*/rep % 2 ? 0.0f : 8.0f,
                                          /*shared_infer_threads=*/shared);
              std::unique_ptr<NodeTable> node_table =
                  mcgs ? std::unique_ptr<NodeTable>(
                             std::make_unique<McgsNodeTable>())
                       : std::unique_ptr<NodeTable>(
                             std::make_unique<MctsNodeTable>());
              Game game;
              game::Color to_move;
              RandomOpening(game, 10 + rep, 1000 * group + rep, &to_move);
              core::Probability probability;
              TreeNode* root =
                  node_table->GetOrCreate(game.board().hash(), to_move, false);
              const int budget = stop_mode == 0 ? 2000 + 37 * rep : (1 << 30);
              Search::Params params = MakeParams(num_threads, budget);
              params.total_visit_time_ms = stop_mode == 1 ? 20 : 0;
              params.q_fn_kind = QFnKind::kVirtualLossSoft;
              params.collision_policy_kind = CollisionPolicyKind::kAbort;
              std::thread stopper;
              if (stop_mode == 2) {
                const int delay_us = (rep * 7919) % 20000;
                stopper = std::thread([&p, delay_us]() {
                  std::this_thread::sleep_for(
                      std::chrono::microseconds(delay_us));
                  p.search->StopSearch();
                });
              }
              Search::Result result = p.search->Run(
                  probability, game, node_table.get(), root, to_move, params);
              if (stopper.joinable()) stopper.join();
              searches++;
              total_visits += result.num_visits;
              total_aborted += result.num_aborted;
              const TreeCheck tc = CheckTree(root, !mcgs);
              INFO("group=" << group << " shared=" << shared
                            << " wake=" << int(wake) << " mcgs=" << mcgs
                            << " stop_mode=" << stop_mode << " rep=" << rep
                            << " visits=" << result.num_visits
                            << " aborted=" << result.num_aborted);
              if (stop_mode != 2) CHECK(result.num_visits > 0);
              CHECK((result.num_visits + result.num_aborted) % group == 0);
              CHECK(root->n == 1 + (int)result.num_visits);
              if (stop_mode == 0) {
                CHECK(result.num_visits >= (size_t)budget);
                CHECK(result.num_visits < (size_t)(budget + num_threads));
              }
              CHECK(tc.bad_in_flight == 0);
              CHECK(tc.bad_pending == 0);
              CHECK(tc.bad_n == 0);
              // Not CHECKed: UpdateVCategorial runs outside node->mu (tree.h),
              // so concurrent backups can lose bucket increments with or
              // without pipelining. Reported in PIPE_STRESS.
              if (!mcgs) vcat_mismatch_nodes += tc.bad_vcat;
            }
          }
        }
      }
    }
  }
  fprintf(stderr,
          "PIPE_STRESS searches=%ld visits=%ld aborted=%ld "
          "mcts_vcat_mismatch_nodes=%ld\n",
          (long)searches, (long)total_visits, (long)total_aborted,
          (long)vcat_mismatch_nodes);
}

// EXP round-free stress: queue-mode interface, workers without rounds.
// Thread / batch / min-fill combinations, both wakers, MCTS and MCGS, stop by
// visit budget, by time, and by an external StopSearch(). Checks that every
// search returns and the structural invariants.
TEST_CASE("ParallelSearch - EXP round-free stress") {
  const char* reps_env = std::getenv("P3_PIPE_REPS");
  const int reps = reps_env ? std::atoi(reps_env) : 2;
  struct RfCfg {
    int threads, batch, min_fill;
    int64_t timeout_us;
  };
  int64_t searches = 0, total_visits = 0, total_aborted = 0;
  const char* repeat_env = std::getenv("P3_RF_REPEAT");
  const bool kRepeat = repeat_env != nullptr && std::string(repeat_env) == "1";
  for (const RfCfg& c :
       {RfCfg{2, 2, 1, 0}, RfCfg{4, 4, 1, 0}, RfCfg{32, 16, 1, 0},
        RfCfg{32, 32, 1, 0}, RfCfg{64, 32, 1, 0}, RfCfg{32, 32, 16, 200},
        RfCfg{64, 32, 32, 100}}) {
    for (auto wake : {NNInterface::WakeStrategy::kMutex,
                      NNInterface::WakeStrategy::kGenCounter}) {
      for (bool mcgs : {false, true}) {
        for (int stop_mode = 0; stop_mode < 3; ++stop_mode) {
          for (int rep = 0; rep < reps; ++rep) {
            PipeSetup p = MakePipeSetup(
                c.threads, /*pipelined=*/false, wake, /*hash_engine=*/true,
                /*logit_scale=*/rep % 2 ? 0.0f : 8.0f,
                /*shared_infer_threads=*/0, c.batch, c.min_fill, c.timeout_us);
            // P3_RF_REPEAT=1: search the same position again on the same
            // interface (fresh tree), so the second pass hits the NN cache
            // while other workers insert / evict.
            for (int pass = 0; pass < (kRepeat ? 2 : 1); ++pass) {
              std::unique_ptr<NodeTable> node_table =
                  mcgs ? std::unique_ptr<NodeTable>(
                             std::make_unique<McgsNodeTable>())
                       : std::unique_ptr<NodeTable>(
                             std::make_unique<MctsNodeTable>());
              Game game;
              game::Color to_move;
              RandomOpening(game, 10 + rep, 1000 * c.threads + rep, &to_move);
              core::Probability probability;
              TreeNode* root =
                  node_table->GetOrCreate(game.board().hash(), to_move, false);
              const int budget = stop_mode == 0 ? 2000 + 37 * rep : (1 << 30);
              Search::Params params = MakeParams(c.threads, budget);
              params.total_visit_time_ms = stop_mode == 1 ? 20 : 0;
              params.q_fn_kind = QFnKind::kVirtualLossSoft;
              params.collision_policy_kind = CollisionPolicyKind::kAbort;
              std::thread stopper;
              if (stop_mode == 2) {
                const int delay_us = (rep * 7919) % 20000;
                stopper = std::thread([&p, delay_us]() {
                  std::this_thread::sleep_for(
                      std::chrono::microseconds(delay_us));
                  p.search->StopSearch();
                });
              }
              Search::Result result = p.search->Run(
                  probability, game, node_table.get(), root, to_move, params);
              if (stopper.joinable()) stopper.join();
              searches++;
              total_visits += result.num_visits;
              total_aborted += result.num_aborted;
              const TreeCheck tc = CheckTree(root, !mcgs);
              INFO("threads=" << c.threads << " batch=" << c.batch << " min_fill="
                              << c.min_fill << " wake=" << int(wake)
                              << " mcgs=" << mcgs << " stop_mode=" << stop_mode
                              << " rep=" << rep << " visits=" << result.num_visits
                              << " aborted=" << result.num_aborted);
              if (stop_mode != 2) CHECK(result.num_visits > 0);
              CHECK(root->n == 1 + (int)result.num_visits);
              if (stop_mode == 0) {
                CHECK(result.num_visits >= (size_t)budget);
                CHECK(result.num_visits < (size_t)(budget + c.threads));
              }
              CHECK(tc.bad_in_flight == 0);
              CHECK(tc.bad_pending == 0);
              CHECK(tc.bad_n == 0);
              if (!mcgs) CHECK(tc.bad_vcat == 0);
            }
          }
        }
      }
    }
  }
  // Per-search NN cache counts: the search's "NNC" stderr lines.
  fprintf(stderr, "RF_STRESS searches=%ld visits=%ld aborted=%ld\n",
          (long)searches, (long)total_visits, (long)total_aborted);
}

// INST pipelined search accuracy: exact bottom-up recompute of every node
// after the search quiesces (MCTS table, hash engine, abort), pipelined vs
// non-pipelined at the same total thread count. Prints STATDEV lines; the
// structural invariants are CHECKed.
TEST_CASE("ParallelSearch - INST pipeline stat recompute") {
  const char* reps_env = std::getenv("P3_PIPE_REPS");
  const int reps = reps_env ? std::atoi(reps_env) : 2;
  struct Cfg {
    const char* name;
    int threads;
    bool pipelined;
    int shared_infer_threads = 0;
    int rf_batch = 0;
  };
  for (const float logit_scale : {0.0f, 8.0f})
    for (const Cfg& cfg :
         {Cfg{"base_t1", 1, false}, Cfg{"base_t32", 32, false},
          Cfg{"base_t64", 64, false}, Cfg{"pipe_2x16", 32, true},
          Cfg{"pipe_2x32", 64, true}, Cfg{"pipe_2x16_sh1", 32, true, 1},
          Cfg{"pipe_2x16_sh2", 32, true, 2}, Cfg{"pipe_2x32_sh1", 64, true, 1},
          Cfg{"rf_t32_b32", 32, false, 0, 32},
          Cfg{"rf_t64_b32", 64, false, 0, 32},
          Cfg{"rf_t32_b16", 32, false, 0, 16}}) {
      TreeCheck tot;
      int64_t visits = 0, aborted = 0;
      for (int pos = 0; pos < 8; ++pos) {
        for (int rep = 0; rep < reps; ++rep) {
          PipeSetup p = MakePipeSetup(cfg.threads, cfg.pipelined,
                                      NNInterface::WakeStrategy::kGenCounter,
                                      /*hash_engine=*/true, logit_scale,
                                      cfg.shared_infer_threads, cfg.rf_batch);
          MctsNodeTable node_table;
          Game game;
          game::Color to_move;
          RandomOpening(game, 20 + 3 * pos, 77 + pos, &to_move);
          core::Probability probability;
          TreeNode* root =
              node_table.GetOrCreate(game.board().hash(), to_move, false);
          Search::Params params = MakeParams(cfg.threads, 4000);
          params.q_fn_kind = QFnKind::kVirtualLossSoft;
          params.collision_policy_kind = CollisionPolicyKind::kAbort;
          Search::Result result = p.search->Run(probability, game, &node_table,
                                                root, to_move, params);
          visits += result.num_visits;
          aborted += result.num_aborted;
          const TreeCheck c = CheckTree(root, /*mcts=*/true);
          INFO("cfg=" << cfg.name << " pos=" << pos << " rep=" << rep);
          CHECK(c.bad_in_flight == 0);
          CHECK(c.bad_pending == 0);
          CHECK(c.bad_n == 0);
          CHECK(root->n == 1 + (int)result.num_visits);
          tot.nodes += c.nodes;
          tot.bad_vcat += c.bad_vcat;
          tot.bad_n += c.bad_n;
          tot.stat_nodes += c.stat_nodes;
          tot.max_dv = std::max(tot.max_dv, c.max_dv);
          tot.max_dvo = std::max(tot.max_dvo, c.max_dvo);
          tot.max_ds = std::max(tot.max_ds, c.max_ds);
          tot.max_de = std::max(tot.max_de, c.max_de);
          tot.sum_dv += c.sum_dv;
          tot.sum_dvo += c.sum_dvo;
          tot.sum_ds += c.sum_ds;
          tot.sum_de += c.sum_de;
        }
      }
      const double sn = std::max<int64_t>(tot.stat_nodes, 1);
      fprintf(stderr,
              "STATDEV logit_scale=%.0f cfg=%s searches=%d visits=%ld "
              "aborted=%ld nodes=%d "
              "stat_nodes=%ld bad_n=%d vcat_mismatch_nodes=%d v_max=%.3g "
              "v_mean=%.3g vo_max=%.3g vo_mean=%.3g score_max=%.3g "
              "score_mean=%.3g err_max=%.3g err_mean=%.3g\n",
              logit_scale, cfg.name, 8 * reps, (long)visits, (long)aborted,
              tot.nodes, (long)tot.stat_nodes, tot.bad_n, tot.bad_vcat,
              tot.max_dv, tot.sum_dv / sn, tot.max_dvo, tot.sum_dvo / sn,
              tot.max_ds, tot.sum_ds / sn, tot.max_de, tot.sum_de / sn);
    }
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
