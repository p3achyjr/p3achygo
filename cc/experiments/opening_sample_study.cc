// Experiment: Opening sampling method study.
//
// Plays N moves from the empty board using different sampling policies, then
// evaluates the resulting position with a raw NN forward pass.
//
// Sampling configs:
//   gumbel  : GumbelEvaluator::SearchRoot(N=1) — samples from raw policy
//   temp=T  : policy^(1/T) renormalized, multinomial sample
//
// For each (num_moves, sampling_config), runs --num_trials trials and reports:
//   mean(V_root), stddev(V_root), max(|V_root|)
// where V_root = init_outcome_est from a single NN eval at the final position.
// One example board is printed per (num_moves, config) tuple.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/game/symmetry.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(int, num_trials, 200, "Number of trials per (num_moves, config).");
ABSL_FLAG(std::string, n_values, "1,2,4,8,16",
          "Comma-separated move counts to evaluate.");
ABSL_FLAG(std::string, temperatures, "1.0,1.1,1.25,1.5,2.0",
          "Comma-separated base temperature diffs from 1.0. Each is divided "
          "by the per-N divisor.");

namespace {

using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::game::MoveOk;
using ::mcts::GumbelEvaluator;
using ::mcts::GumbelSearchParams;
using ::mcts::MctsNodeTable;

using Position = std::array<Color, constants::kNumBoardLocs>;

// Returns the lex-min form of pos under the 8 board symmetries × color-swap.
Position CanonicalPosition(const Position& pos) {
  Position best = pos;
  for (int s = 0; s < game::kSymUpperBound; ++s) {
    auto sym = static_cast<game::Symmetry>(s);
    Position t = game::ApplySymmetry(sym, pos, BOARD_LEN);
    if (t < best) best = t;
    // Color-swap: BLACK=1, WHITE=-1, EMPTY=0 → negate.
    Position tc = t;
    for (Color& c : tc) c = -c;
    if (tc < best) best = tc;
  }
  return best;
}

// Number of intersections differing in occupancy (occupied vs empty).
int DistOcc(const Position& a, const Position& b) {
  int d = 0;
  for (int i = 0; i < constants::kNumBoardLocs; ++i) {
    d += ((a[i] != EMPTY) != (b[i] != EMPTY)) ? 1 : 0;
  }
  return d;
}

// Number of intersections differing in color (BLACK / WHITE / EMPTY).
int DistColor(const Position& a, const Position& b) {
  int d = 0;
  for (int i = 0; i < constants::kNumBoardLocs; ++i) {
    d += (a[i] != b[i]) ? 1 : 0;
  }
  return d;
}

struct DistStats {
  float mean, min, p05, p95, max;
};

DistStats ComputeDistStats(std::vector<int>& vals) {
  if (vals.empty()) return {};
  std::sort(vals.begin(), vals.end());
  const int n = static_cast<int>(vals.size());
  float mean = 0.f;
  for (int v : vals) mean += v;
  mean /= n;
  auto pct = [&](float p) -> float {
    float idx = p * (n - 1);
    int lo = static_cast<int>(idx);
    int hi = std::min(lo + 1, n - 1);
    return vals[lo] + (idx - lo) * (vals[hi] - vals[lo]);
  };
  return {mean, static_cast<float>(vals.front()), pct(0.05f), pct(0.95f),
          static_cast<float>(vals.back())};
}

static constexpr int kGumbelK = 5;
static constexpr size_t kCacheSize = 32768;
static constexpr int64_t kTimeoutUs = 400;
static constexpr float kEps = 1e-9f;

std::vector<int> ParseIntList(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(std::stoi(tok));
  return out;
}

std::vector<float> ParseFloatList(const std::string& s) {
  std::vector<float> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(std::stof(tok));
  return out;
}

struct SamplingConfig {
  enum Kind { kGumbel, kTemp };
  Kind kind;
  float temperature;  // only used when kind == kTemp

  std::string label() const {
    if (kind == kGumbel) return "gumbel";
    char buf[32];
    snprintf(buf, sizeof(buf), "temp=%.2f", temperature);
    return buf;
  }
};

Game MakeEmptyGame() {
  game::Board board(/*komi=*/7.5f);
  absl::InlinedVector<game::Move, constants::kNumLastMoves> last_moves;
  for (int i = 0; i < constants::kNumLastMoves; ++i) {
    last_moves.push_back(game::Move{BLACK, game::kNoopLoc});
  }
  return Game(board, last_moves, /*init_mv_num=*/0);
}

// Calls SearchRoot(N=1) and returns the move chosen by Gumbel sampling.
Loc SampleGumbel(nn::NNInterface* nn, Game& game, Color color, uint64_t seed) {
  auto node_table = std::make_unique<MctsNodeTable>();
  auto* root =
      node_table->GetOrCreate(game.board().hash(), color, /*is_root=*/false);
  GumbelEvaluator evaluator(nn, /*thread_id=*/0);
  core::Probability prob(seed);
  auto params = GumbelSearchParams::Builder().set_n(1).set_k(kGumbelK).build();
  auto result =
      evaluator.SearchRoot(prob, game, node_table.get(), root, color, params);
  return result.mcts_move;
}

// Samples a move from policy^(1/T) via rejection sampling until legal.
Loc SampleTemp(nn::NNInterface* nn, Game& game, Color color, float temperature,
               std::mt19937& rng) {
  core::Probability prob;
  auto infer = nn->LoadAndGetInference(/*thread_id=*/0, game, color, prob);

  // Apply temperature.
  std::array<float, constants::kMaxMovesPerPosition> weights;
  float total = 0.f;
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    float p = infer.move_probs[i];
    float w = (p > kEps) ? std::pow(p, 1.0f / temperature) : 0.f;
    weights[i] = w;
    total += w;
  }
  if (total < kEps) return game::kPassLoc;

  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  while (true) {
    int idx = dist(rng);
    Loc move =
        (idx == constants::kNumBoardLocs) ? game::kPassLoc : game::AsLoc(idx);
    if (MoveOk(game.board().PlayMoveDry(move, color))) return move;
  }
}

// Returns init_outcome_est from a single NN eval at the position.
float EvalNN(nn::NNInterface* nn, Game& game, Color color) {
  auto node_table = std::make_unique<MctsNodeTable>();
  auto* root =
      node_table->GetOrCreate(game.board().hash(), color, /*is_root=*/false);
  GumbelEvaluator evaluator(nn, /*thread_id=*/0);
  core::Probability prob;
  auto params = GumbelSearchParams::Builder().set_n(1).set_k(kGumbelK).build();
  evaluator.SearchRoot(prob, game, node_table.get(), root, color, params);
  return root->init_outcome_est;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const int num_trials = absl::GetFlag(FLAGS_num_trials);
  const std::vector<int> n_values = ParseIntList(absl::GetFlag(FLAGS_n_values));
  const std::vector<float> temperatures =
      ParseFloatList(absl::GetFlag(FLAGS_temperatures));
  const int num_move_entries = static_cast<int>(n_values.size());

  if (model_path.empty()) {
    LOG(ERROR) << "--model_path not specified.";
    return 1;
  }

  auto engine = nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                                 /*batch_size=*/1,
                                 nn::GetVersionFromModelPath(model_path));
  auto nn_interface = std::make_unique<nn::NNInterface>(
      /*num_threads=*/1, kTimeoutUs, kCacheSize, std::move(engine));

  static constexpr int kNumExampleBoards = 5;

  printf("\nOpening Sample Study\n");
  printf("Trials per config: %d\n\n", num_trials);

  // configs_per_n[mi]: gumbel + temperatures (same for all N).
  std::vector<std::vector<SamplingConfig>> configs_per_n(num_move_entries);
  for (int mi = 0; mi < num_move_entries; ++mi) {
    configs_per_n[mi].push_back({SamplingConfig::kGumbel, 1.0f});
    for (float temp : temperatures) {
      configs_per_n[mi].push_back({SamplingConfig::kTemp, temp});
    }
  }
  const int num_configs = static_cast<int>(configs_per_n[0].size());

  // summary[mi][ci]
  struct Summary {
    float mean, stddev, max_abs, diversity;
    DistStats d_occ, d_color;
  };
  std::vector<std::vector<Summary>> summary(num_move_entries,
                                            std::vector<Summary>(num_configs));

  for (int mi = 0; mi < num_move_entries; ++mi) {
    const int num_moves = n_values[mi];
    const std::vector<SamplingConfig>& configs = configs_per_n[mi];

    for (int ci = 0; ci < num_configs; ++ci) {
      const SamplingConfig& cfg = configs[ci];

      std::vector<float> v_samples;
      v_samples.reserve(num_trials);
      std::vector<std::string> example_boards;
      std::vector<Position> canonical_positions;
      canonical_positions.reserve(num_trials);

      std::mt19937 rng(/*seed=*/42 ^ static_cast<uint32_t>(mi * 100 + ci));

      for (int trial = 0; trial < num_trials; ++trial) {
        Game game = MakeEmptyGame();
        Color color = BLACK;

        for (int move_idx = 0; move_idx < num_moves; ++move_idx) {
          Loc move;
          if (cfg.kind == SamplingConfig::kGumbel) {
            uint64_t seed = static_cast<uint64_t>(trial) * 1000003ULL +
                            static_cast<uint64_t>(move_idx) * 17ULL;
            move = SampleGumbel(nn_interface.get(), game, color, seed);
          } else {
            move = SampleTemp(nn_interface.get(), game, color, cfg.temperature,
                              rng);
          }
          game.PlayMove(move, color);
          color = game::OppositeColor(color);
        }

        float v = EvalNN(nn_interface.get(), game, color);
        v_samples.push_back(v);
        canonical_positions.push_back(
            CanonicalPosition(game.board().position()));

        if (static_cast<int>(example_boards.size()) < kNumExampleBoards) {
          example_boards.push_back(game::ToString(game.board().position()));
        }
      }

      // Unique positions.
      std::set<Position> unique_set(canonical_positions.begin(),
                                    canonical_positions.end());
      float diversity = static_cast<float>(unique_set.size()) / num_trials;

      // Pairwise distances.
      std::vector<int> occ_dists, color_dists;
      occ_dists.reserve(num_trials * (num_trials - 1) / 2);
      color_dists.reserve(num_trials * (num_trials - 1) / 2);
      for (int a = 0; a < num_trials; ++a) {
        for (int b = a + 1; b < num_trials; ++b) {
          occ_dists.push_back(
              DistOcc(canonical_positions[a], canonical_positions[b]));
          color_dists.push_back(
              DistColor(canonical_positions[a], canonical_positions[b]));
        }
      }
      DistStats d_occ = ComputeDistStats(occ_dists);
      DistStats d_color = ComputeDistStats(color_dists);

      // V stats.
      float mean = 0.f;
      for (float v : v_samples) mean += v;
      mean /= static_cast<float>(num_trials);

      float var = 0.f;
      for (float v : v_samples) var += (v - mean) * (v - mean);
      var /= static_cast<float>(num_trials);
      float stddev = std::sqrt(var);

      float max_abs = 0.f;
      for (float v : v_samples) max_abs = std::max(max_abs, std::abs(v));

      summary[mi][ci] = {mean, stddev, max_abs, diversity, d_occ, d_color};

      // Print example boards.
      printf(
          "\n=== num_moves=%d  config=%s  "
          "(mean=%.4f  std=%.4f  max|V|=%.4f  diversity=%.3f  "
          "d_occ mean=%.1f  d_color mean=%.1f) ===\n",
          num_moves, cfg.label().c_str(), mean, stddev, max_abs, diversity,
          d_occ.mean, d_color.mean);
      for (int bi = 0; bi < static_cast<int>(example_boards.size()); ++bi) {
        printf("--- trial %d ---\n%s\n", bi, example_boards[bi].c_str());
      }
    }

    printf("\n");
  }

  // Summary table.
  printf("\n%s\n", std::string(80, '=').c_str());
  printf("Summary\n");
  printf("%s\n\n", std::string(80, '=').c_str());

  // Header.
  printf(
      "%-10s  %-12s  %8s  %8s  %8s  %8s  "
      "%7s %7s %7s %7s %7s  "
      "%7s %7s %7s %7s %7s\n",
      "num_moves", "config", "mean(V)", "std(V)", "max|V|", "divers", "occ.mn",
      "occ.p05", "occ.p95", "occ.mn", "occ.mx", "col.mn", "col.p05", "col.p95",
      "col.mn", "col.mx");
  printf(
      "%-10s  %-12s  %8s  %8s  %8s  %8s  "
      "%7s %7s %7s %7s %7s  "
      "%7s %7s %7s %7s %7s\n",
      "----------", "------------", "--------", "--------", "--------",
      "--------", "-------", "-------", "-------", "-------", "-------",
      "-------", "-------", "-------", "-------", "-------");

  for (int mi = 0; mi < num_move_entries; ++mi) {
    for (int ci = 0; ci < num_configs; ++ci) {
      const Summary& s = summary[mi][ci];
      printf(
          "%-10d  %-12s  %8.4f  %8.4f  %8.4f  %8.3f  "
          "%7.1f %7.1f %7.1f %7.1f %7.1f  "
          "%7.1f %7.1f %7.1f %7.1f %7.1f\n",
          n_values[mi], configs_per_n[mi][ci].label().c_str(), s.mean, s.stddev,
          s.max_abs, s.diversity, s.d_occ.mean, s.d_occ.min, s.d_occ.p05,
          s.d_occ.p95, s.d_occ.max, s.d_color.mean, s.d_color.min,
          s.d_color.p05, s.d_color.p95, s.d_color.max);
    }
    printf("\n");
  }

  return 0;
}
