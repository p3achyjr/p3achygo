// Experiment: Variance vs. (root Q, visit count) study.
//
// Plays self-play games in batch mode (one thread per game, shared
// NNInterface). For each turn, runs PUCT for `--n` visits with tree reuse,
// BiasCache(alpha=0.85, lambda=0.45), root_fpu=0.1. After each turn, traverses
// the searched tree and, for every node with n >= 3, accumulates v_var into a
// 2D histogram keyed by (root_q_bin, node_n_bin):
//   root_q_bin: floor(root->v / --q_bin_size), root->v ∈ [-1, 1].
//   node_n_bin: floor(node->n / --n_bin_size).
//
// At the end, aggregates across games and prints per-cell mean/variance of
// v_var to the console.
//
// Komi = 7.5; no move cap; tree reuse on.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/mcts/bias_cache.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
#include "cc/mcts/tree.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(int, num_games, 0, "Number of self-play games.");
ABSL_FLAG(int, n, 1600, "PUCT visits per turn.");
ABSL_FLAG(int, n_bin_size, 10, "Linear bin size for node visit-count axis.");
ABSL_FLAG(float, q_bin_size, 0.05f,
          "Linear bin size for root-Q axis (Q in [-1, 1]).");
ABSL_FLAG(int, cache_size, constants::kDefaultNNCacheSize / 2,
          "NNInterface cache size.");
ABSL_FLAG(int, timeout, 200, "NNInterface timeout in us.");

namespace {

using ::core::Probability;
using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::game::OppositeColor;
using ::mcts::BiasCache;
using ::mcts::GumbelEvaluator;
using ::mcts::MctsNodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::TreeNode;

constexpr float kBiasCacheAlpha = 0.85f;
constexpr float kBiasCacheLambda = 0.45f;
constexpr float kRootFpu = 0.1f;
constexpr float kKomi = 7.5f;

struct Accumulator {
  uint64_t count = 0;
  double sum = 0.0;
  double sum_sq = 0.0;

  void Add(double x) {
    count += 1;
    sum += x;
    sum_sq += x * x;
  }

  void Merge(const Accumulator& o) {
    count += o.count;
    sum += o.sum;
    sum_sq += o.sum_sq;
  }

  double Mean() const {
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }
  double Var() const {
    if (count < 2) return 0.0;
    const double m = Mean();
    const double v = sum_sq / static_cast<double>(count) - m * m;
    return v < 0.0 ? 0.0 : v;
  }
};

// Pack (root_q_bin, n_bin) into one int64 key. root_q_bin is signed.
inline int64_t PackKey(int root_q_bin, int n_bin) {
  return (static_cast<int64_t>(root_q_bin) << 32) |
         static_cast<uint32_t>(n_bin);
}
inline int UnpackQBin(int64_t k) { return static_cast<int>(k >> 32); }
inline int UnpackNBin(int64_t k) { return static_cast<int>(k & 0xFFFFFFFFLL); }

inline int QBinOf(float q, float q_bin_size) {
  return static_cast<int>(std::floor(q / q_bin_size));
}

using BinMap = std::unordered_map<int64_t, Accumulator>;

void TraverseAndAccumulate(
    const TreeNode* root, int root_q_bin, int n_bin_size, BinMap* m,
    std::vector<std::pair<int, float>>* samples_out = nullptr) {
  if (root == nullptr) return;
  std::vector<const TreeNode*> stack;
  stack.reserve(2048);
  stack.push_back(root);
  while (!stack.empty()) {
    const TreeNode* node = stack.back();
    stack.pop_back();
    if (node == nullptr) continue;
    if (node->n >= 3) {
      const int n_bin = node->n / n_bin_size;
      (*m)[PackKey(root_q_bin, n_bin)].Add(static_cast<double>(node->v_var));
      if (samples_out) samples_out->emplace_back(node->n, node->v_var);
    }
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      const TreeNode* c = node->child(a);
      if (c != nullptr) stack.push_back(c);
    }
  }
}

void PlaySelfPlayGame(size_t seed, int game_id, nn::NNInterface* nn,
                      int n_visits, int n_bin_size, float q_bin_size,
                      std::promise<BinMap> result) {
  Probability probability(seed);
  Game game;
  game.SetKomi(kKomi);
  Color color_to_move = BLACK;

  auto node_table = std::make_unique<MctsNodeTable>();
  TreeNode* root = node_table->GetOrCreate(game.board().hash(), color_to_move,
                                           /*is_terminal=*/false);

  BiasCache bias_cache(kBiasCacheAlpha, kBiasCacheLambda);
  GumbelEvaluator evaluator(nn, game_id, &bias_cache);

  const bool log_this_game = (game_id == 0);

  BinMap acc;

  while (!game.IsGameOver()) {
    auto search_result = evaluator.SearchRootPuct(
        probability, game, node_table.get(), root, color_to_move, n_visits,
        PuctParams::Builder()
            .set_kind(PuctRootSelectionPolicy::kLcb)
            .set_root_fpu(kRootFpu)
            .build());

    const int root_q_bin = QBinOf(root->v, q_bin_size);
    std::vector<std::pair<int, float>> samples;
    TraverseAndAccumulate(root, root_q_bin, n_bin_size, &acc,
                          log_this_game ? &samples : nullptr);

    if (log_this_game) {
      double sum = 0.0, sum_sq = 0.0;
      for (const auto& [n, v] : samples) {
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
      }
      const size_t k = samples.size();
      const double mean = k > 0 ? sum / static_cast<double>(k) : 0.0;
      const double var =
          k > 1 ? std::max(0.0, sum_sq / static_cast<double>(k) - mean * mean)
                : 0.0;
      const double stddev = std::sqrt(var);
      std::stringstream s;
      s << "\n===== Game 0  Turn " << game.num_moves()
        << "  Color=" << (color_to_move == BLACK ? "B" : "W")
        << "  root_q=" << absl::StrFormat("%.4f", root->v)
        << "  root_q_bin=" << root_q_bin << "  samples=" << k
        << "  mean(v_var)=" << absl::StrFormat("%.6f", mean)
        << "  std(v_var)=" << absl::StrFormat("%.6f", stddev) << " =====\n";
      s << game.board() << "\n";
      printf("%s", s.str().c_str());
      fflush(stdout);
    }

    Loc move = search_result.mcts_move;
    game.PlayMove(move, color_to_move);
    color_to_move = OppositeColor(color_to_move);

    TreeNode* next_root = root->children[move];
    if (next_root == nullptr) {
      next_root = node_table->GetOrCreate(game.board().hash(), color_to_move,
                                          game.IsGameOver());
    }
    node_table->Reap(next_root);
    bias_cache.PruneUnused();
    root = next_root;
  }

  game.WriteResult();
  nn->UnregisterThread(game_id);
  result.set_value(std::move(acc));
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const int num_games = absl::GetFlag(FLAGS_num_games);
  const int n_visits = absl::GetFlag(FLAGS_n);
  const int n_bin_size = absl::GetFlag(FLAGS_n_bin_size);
  const float q_bin_size = absl::GetFlag(FLAGS_q_bin_size);
  const int cache_size = absl::GetFlag(FLAGS_cache_size);
  const int timeout_us = absl::GetFlag(FLAGS_timeout);

  if (model_path.empty()) {
    LOG(ERROR) << "--model_path not specified.";
    return 1;
  }
  if (num_games <= 0) {
    LOG(ERROR) << "--num_games must be > 0.";
    return 1;
  }
  if (n_bin_size <= 0) {
    LOG(ERROR) << "--n_bin_size must be > 0.";
    return 1;
  }
  if (q_bin_size <= 0.0f) {
    LOG(ERROR) << "--q_bin_size must be > 0.";
    return 1;
  }

  auto engine = nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                                 /*batch_size=*/num_games,
                                 nn::GetVersionFromModelPath(model_path));
  auto nn_interface = std::make_unique<nn::NNInterface>(
      /*num_threads=*/num_games, timeout_us, cache_size, std::move(engine));

  const size_t time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();

  std::vector<std::thread> threads;
  std::vector<std::future<BinMap>> futures;
  threads.reserve(num_games);
  futures.reserve(num_games);

  for (int game_id = 0; game_id < num_games; ++game_id) {
    std::promise<BinMap> p;
    futures.emplace_back(p.get_future());
    const size_t seed = absl::HashOf(time, game_id);
    threads.emplace_back(PlaySelfPlayGame, seed, game_id, nn_interface.get(),
                         n_visits, n_bin_size, q_bin_size, std::move(p));
  }

  LOG(INFO) << "Playing " << num_games << " self-play games at n=" << n_visits
            << " visits/turn, n_bin_size=" << n_bin_size
            << ", q_bin_size=" << q_bin_size;

  for (auto& t : threads) t.join();

  // Aggregate.
  BinMap total;
  for (auto& f : futures) {
    BinMap m = f.get();
    for (auto& [k, a] : m) {
      total[k].Merge(a);
    }
  }

  // Sort by (root_q_bin, n_bin) ascending.
  std::vector<int64_t> keys;
  keys.reserve(total.size());
  for (const auto& [k, _] : total) keys.push_back(k);
  std::sort(keys.begin(), keys.end(), [](int64_t a, int64_t b) {
    const int qa = UnpackQBin(a), qb = UnpackQBin(b);
    if (qa != qb) return qa < qb;
    return UnpackNBin(a) < UnpackNBin(b);
  });

  uint64_t total_samples = 0;
  for (const auto& [_, a] : total) total_samples += a.count;

  printf("\nVariance vs. (Root Q, Visit Count) Study\n");
  printf("model: %s\n", model_path.c_str());
  printf("games: %d  n_visits/turn: %d  n_bin_size: %d  q_bin_size: %.4f\n",
         num_games, n_visits, n_bin_size, q_bin_size);
  printf("total samples (nodes with n>=3): %llu\n\n",
         static_cast<unsigned long long>(total_samples));

  printf("%-10s  %-10s  %-10s  %-10s  %-12s  %-14s  %-14s  %-14s\n", "q_lo",
         "q_hi", "n_lo", "n_hi", "count", "mean(v_var)", "var(v_var)",
         "std(v_var)");
  printf("%-10s  %-10s  %-10s  %-10s  %-12s  %-14s  %-14s  %-14s\n",
         "----------", "----------", "----------", "----------", "------------",
         "--------------", "--------------", "--------------");
  for (int64_t k : keys) {
    const Accumulator& a = total[k];
    const int q_bin = UnpackQBin(k);
    const int n_bin = UnpackNBin(k);
    const float q_lo = q_bin * q_bin_size;
    const float q_hi = (q_bin + 1) * q_bin_size;
    const int n_lo = n_bin * n_bin_size;
    const int n_hi = (n_bin + 1) * n_bin_size - 1;
    const double mean = a.Mean();
    const double var = a.Var();
    const double stddev = std::sqrt(var);
    printf(
        "%-10.4f  %-10.4f  %-10d  %-10d  %-12llu  %-14.6f  %-14.6f  %-14.6f\n",
        q_lo, q_hi, n_lo, n_hi, static_cast<unsigned long long>(a.count), mean,
        var, stddev);
  }

  return 0;
}
