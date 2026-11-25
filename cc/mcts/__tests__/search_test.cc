// Define DOCTEST_CONFIG_IMPLEMENT so we can provide a custom main that parses
// absl flags before running tests.
#define DOCTEST_CONFIG_IMPLEMENT
#include "cc/mcts/search.h"

#include <filesystem>
#include <limits>
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
#include "cc/mcts/bias_cache.h"
#include "cc/mcts/gumbel.h"
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

// Deterministic position-dependent fake net. Policy, value, score and error
// are pseudo-random functions of (board, color), so subtrees differ and the
// bottom-up stat check is not trivially uniform. Rows are engine batch rows.
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
  double max_dv = 0, max_dvo = 0, max_ds = 0, max_de = 0;
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
    if (node->n_in_flight.load() != 0) c.bad_in_flight++;
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
      // Leaf backup: every stat is the node's own NN estimate.
      e = {node->init_util_est, node->init_outcome_est, node->init_score_est,
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
    c.stat_nodes++;
    c.max_dv = std::max(c.max_dv, std::abs(double(node->v) - e.v));
    c.max_dvo = std::max(c.max_dvo, std::abs(double(node->v_outcome) - e.vo));
    c.max_ds = std::max(c.max_ds, std::abs(double(node->score) - e.s));
    c.max_de = std::max(c.max_de, std::abs(double(node->v_err) - e.e));
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

// Round-free search (SignalKind::kRoundFree interface): structural invariants
// and visit accounting across thread counts, engine batches (<= threads),
// wake strategies, MCTS / MCGS tables, stop modes (budget, timer,
// StopSearch from another thread), collision policies and policy
// concentration. A second search on the same interface (fresh tree) hits the
// shared NN cache.
TEST_CASE("RoundFreeSearch - stress") {
  struct RfCfg {
    int threads, batch;
  };
  int64_t searches = 0, total_visits = 0, total_aborted = 0;
  for (const RfCfg& c : {RfCfg{2, 1}, RfCfg{2, 2}, RfCfg{4, 4}, RfCfg{32, 16},
                         RfCfg{32, 32}, RfCfg{64, 32}}) {
    for (auto wake : {NNInterface::WakeStrategy::kMutex,
                      NNInterface::WakeStrategy::kGenCounter}) {
      for (bool mcgs : {false, true}) {
        for (auto cp : {CollisionPolicyKind::kAbort,
                        CollisionPolicyKind::kSmartRetryDepth}) {
          for (int stop_mode = 0; stop_mode < 3; ++stop_mode) {
            for (int rep = 0; rep < 2; ++rep) {
              NNInterface nn_interface(
                  c.threads, /*cache_size=*/4096,
                  std::make_unique<HashEngine>(c.batch, rep % 2 ? 0.0f : 8.0f),
                  c.batch, wake);
              nn_interface.SetNumCacheLastMoves(0);
              nn_interface.SetCacheKeyKo(true);
              Search search(nn_interface.MakeSlot(0));
              for (int pass = 0; pass < 2; ++pass) {
                std::unique_ptr<NodeTable> node_table =
                    mcgs ? std::unique_ptr<NodeTable>(
                               std::make_unique<McgsNodeTable>())
                         : std::unique_ptr<NodeTable>(
                               std::make_unique<MctsNodeTable>());
                Game game;
                game::Color to_move;
                RandomOpening(game, 10 + rep, 1000 * c.threads + rep, &to_move);
                core::Probability probability;
                TreeNode* root = node_table->GetOrCreate(game.board().hash(),
                                                         to_move, false);
                const int budget = stop_mode == 0 ? 2000 + 37 * rep : (1 << 30);
                Search::Params params = MakeParams(c.threads, budget);
                params.total_visit_time_ms = stop_mode == 1 ? 20 : 0;
                params.q_fn_kind = QFnKind::kVirtualLossSoft;
                params.collision_policy_kind = cp;
                params.fork_depth_c = 0.8f;
                params.fork_depth_p = 0.5f;
                std::thread stopper;
                if (stop_mode == 2) {
                  const int delay_us = (rep * 7919) % 20000;
                  stopper = std::thread([&search, delay_us]() {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(delay_us));
                    search.StopSearch();
                  });
                }
                Search::Result result = search.Run(
                    probability, game, node_table.get(), root, to_move, params);
                if (stopper.joinable()) stopper.join();
                searches++;
                total_visits += result.num_visits;
                total_aborted += result.num_aborted;
                const TreeCheck tc = CheckTree(root, !mcgs);
                INFO("threads=" << c.threads << " batch=" << c.batch
                                << " wake=" << int(wake) << " mcgs=" << mcgs
                                << " cp=" << int(cp)
                                << " stop_mode=" << stop_mode << " rep=" << rep
                                << " pass=" << pass
                                << " visits=" << result.num_visits
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
  }
  MESSAGE("round-free stress: searches=" << searches
                                         << " visits=" << total_visits
                                         << " aborted=" << total_aborted);
}

// Round-free search: after the search quiesces, every node's v / v_outcome /
// score / v_err equals an exact bottom-up recompute (MCTS table, hash engine).
TEST_CASE("RoundFreeSearch - exact stat recompute") {
  struct Cfg {
    int threads, batch;
  };
  for (const float logit_scale : {0.0f, 8.0f}) {
    for (const Cfg& cfg : {Cfg{4, 4}, Cfg{32, 32}, Cfg{64, 32}, Cfg{32, 16}}) {
      for (auto cp : {CollisionPolicyKind::kAbort,
                      CollisionPolicyKind::kSmartRetryDepth}) {
        for (int pos = 0; pos < 6; ++pos) {
          NNInterface nn_interface(
              cfg.threads, /*cache_size=*/4096,
              std::make_unique<HashEngine>(cfg.batch, logit_scale), cfg.batch,
              NNInterface::WakeStrategy::kGenCounter);
          Search search(nn_interface.MakeSlot(0));
          MctsNodeTable node_table;
          Game game;
          game::Color to_move;
          RandomOpening(game, 20 + 3 * pos, 77 + pos, &to_move);
          core::Probability probability;
          TreeNode* root =
              node_table.GetOrCreate(game.board().hash(), to_move, false);
          Search::Params params = MakeParams(cfg.threads, 4000);
          params.q_fn_kind = QFnKind::kVirtualLossSoft;
          params.collision_policy_kind = cp;
          params.fork_depth_c = 0.8f;
          params.fork_depth_p = 0.5f;
          Search::Result result =
              search.Run(probability, game, &node_table, root, to_move, params);
          const TreeCheck c = CheckTree(root, /*mcts=*/true);
          INFO("threads=" << cfg.threads << " batch=" << cfg.batch
                          << " logit_scale=" << logit_scale << " cp=" << int(cp)
                          << " pos=" << pos << " nodes=" << c.nodes
                          << " max_dv=" << c.max_dv << " max_dvo=" << c.max_dvo
                          << " max_ds=" << c.max_ds << " max_de=" << c.max_de);
          CHECK(root->n == 1 + (int)result.num_visits);
          CHECK(c.bad_in_flight == 0);
          CHECK(c.bad_pending == 0);
          CHECK(c.bad_n == 0);
          CHECK(c.bad_vcat == 0);
          CHECK(c.stat_nodes > 0);
          CHECK(c.max_dv == 0.0);
          CHECK(c.max_dvo == 0.0);
          CHECK(c.max_ds == 0.0);
          CHECK(c.max_de == 0.0);
        }
      }
    }
  }
}

// Single-threaded search (GumbelEvaluator, PUCT and Gumbel root): every
// node's backed-up score equals the bottom-up recompute, with incremental
// updates (no bias cache) and with idempotent recompute (bias cache on, as in
// self-play and GTP).
TEST_CASE("GumbelEvaluator - score recompute") {
  for (const bool gumbel_root : {false, true}) {
    for (const bool bias : {false, true}) {
      for (int pos = 0; pos < 4; ++pos) {
        NNInterface nn_interface(1, std::numeric_limits<int64_t>::max(),
                                 /*cache_size=*/4096,
                                 std::make_unique<HashEngine>(1, 8.0f));
        BiasCache bias_cache(0.85f, 0.45f);
        GumbelEvaluator gumbel(&nn_interface, 0, bias ? &bias_cache : nullptr);
        MctsNodeTable node_table;
        Game game;
        game::Color to_move;
        RandomOpening(game, 20 + 3 * pos, 91 + pos, &to_move);
        core::Probability probability;
        TreeNode* root =
            node_table.GetOrCreate(game.board().hash(), to_move, false);
        if (!gumbel_root) {
          gumbel.SearchRootPuct(probability, game, &node_table, root, to_move,
                                800, PuctParams{});
          const TreeCheck c = CheckTree(root, /*mcts=*/true);
          INFO("puct bias=" << bias << " pos=" << pos << " nodes=" << c.nodes
                            << " max_ds=" << c.max_ds);
          CHECK(c.stat_nodes > 0);
          CHECK(c.max_ds < 1e-3);
          continue;
        }
        // Gumbel root: every child subtree is exact; the root mixes its own
        // estimate with the selected child's backed-up score (like v).
        const GumbelResult res =
            gumbel.SearchRoot(probability, game, &node_table, root, to_move,
                              GumbelSearchParams{800, 16});
        double max_ds = 0;
        int subtrees = 0;
        for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
          TreeNode* child = root->child(a);
          if (child == nullptr || child->n == 0) continue;
          const TreeCheck c = CheckTree(child, /*mcts=*/true);
          max_ds = std::max(max_ds, c.max_ds);
          ++subtrees;
        }
        const TreeNode* sel = root->child(res.mcts_move);
        REQUIRE(sel != nullptr);
        const float n_sel =
            root->child_visits[game::AsIndex(res.mcts_move, BOARD_LEN)];
        const float want =
            (root->init_score_est + n_sel * -sel->score) / (1.0f + n_sel);
        INFO("gumbel bias=" << bias << " pos=" << pos
                            << " subtrees=" << subtrees << " max_ds=" << max_ds
                            << " root=" << root->score << " want=" << want);
        CHECK(subtrees > 1);
        CHECK(max_ds < 1e-3);
        CHECK(root->score == doctest::Approx(want).epsilon(1e-4));
      }
    }
  }
}

// GTP ownership in round-free mode: LoadAndGetOwnership returns the engine's
// ownership for this thread's row.
TEST_CASE("RoundFreeSearch - ownership") {
  NNInterface nn_interface(4, /*cache_size=*/0, std::make_unique<NullEngine>(),
                           /*engine_batch=*/2);
  Game game;
  std::array<float, constants::kNumBoardLocs> own =
      nn_interface.LoadAndGetOwnership(0, game, BLACK);
  for (float x : own) CHECK(x == 0.0f);
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
