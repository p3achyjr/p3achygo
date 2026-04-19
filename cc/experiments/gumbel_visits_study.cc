// Experiment: Gumbel visit count study.
//
// Loads positions from a tfrecord chunk and, for each position, measures the
// KLD between the "ground-truth" improved policy (N=10000, K=16 Gumbel) and
// the improved policy at each N in [64, 100, 150, 200, 300, 400, 800].
//
// For each N-visit search, the tree is first seeded with --seed_visits PUCT
// visits (simulating tree reuse from self-play), then Gumbel(N, K=16) runs.
//
// KLD is computed as:
//   ComputeKLD(ComputeImprovedPolicy(gt_root, 0),
//              ComputeImprovedPolicy(n_root,  0))
//
// The same K=16 actions are used across ground-truth and all N-visit searches
// for a given position: a fixed per-position seed is passed to each SearchRoot
// call, ensuring the Gumbel noise (and thus the top-K action selection) is
// identical across all searches.
//
// Statistics reported per N value: mean, p75, p95, max KLD.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/core/probability.h"
#include "cc/data/tfrecord/record_reader.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/mcts/bias_cache.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"
#include "cc/proto/feature_util.h"
#include "example.pb.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(std::string, chunk_path, "",
          "Path to tfrecord chunk (.tfrecord.zz).");
ABSL_FLAG(int, num_examples, 500, "Number of examples to process.");
ABSL_FLAG(int, seed_visits, 30,
          "PUCT visits used to seed each tree before Gumbel search.");
ABSL_FLAG(bool, verbose, false,
          "If true, print per-example KLD values for each N.");
ABSL_FLAG(bool, use_gumbel, true,
          "If true (default), use Gumbel(N=10000,K=16) as ground truth and "
          "Gumbel for N-visit searches. If false, use PUCT(N=10000,fpu=0.1) "
          "and empirical visit distribution as ground truth.");

namespace {

using ::data::RecordReaderOptions;
using ::data::SequentialRecordReader;
using ::game::AsLoc;
using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::game::MoveOk;
using ::mcts::BiasCache;
using ::mcts::ComputeImprovedPolicy;
using ::mcts::ComputeKLD;
using ::mcts::GumbelEvaluator;
using ::mcts::GumbelSearchParams;
using ::mcts::MctsNodeTable;
using ::mcts::NodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::TreeNode;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

static constexpr int kGroundTruthVisits = 10000;
static constexpr int kGumbelK = 16;
static constexpr size_t kCacheSize = 32768;
static constexpr int64_t kTimeoutUs = 400;

static constexpr int kNValues[] = {64, 100, 150, 200, 270, 300, 400, 800};
static constexpr int kNumNValues =
    static_cast<int>(sizeof(kNValues) / sizeof(kNValues[0]));

template <typename T>
T ParseScalar(const std::string& s) {
  T val;
  memcpy(&val, s.data(), sizeof(T));
  return val;
}

template <typename T, size_t N>
std::array<T, N> ParseSequence(const std::string& s) {
  std::array<T, N> arr;
  memcpy(arr.data(), s.data(), sizeof(T) * N);
  return arr;
}

// Converts a Loc to board coordinates matching the custom ToString format:
// column letter (A=0) + row number (0=top).  E.g. Loc{3,4} -> "E3".
std::string LocToString(game::Loc loc) {
  if (loc == game::kPassLoc) return "pass";
  if (loc == game::kNoopLoc) return "noop";
  static constexpr char kCols[] = "ABCDEFGHIJKLMNOPQRST";
  return std::string(1, kCols[loc.j]) + std::to_string(loc.i);
}

// Samples the top-K valid action indices using logits + Gumbel(0,1) noise,
// matching the action-selection logic inside GumbelEvaluator::SearchRoot.
// Callers should pass a Probability with a fixed seed to make the sample
// reproducible across searches on the same position.
absl::InlinedVector<int, 16> SampleTopKActions(
    const std::array<float, constants::kMaxMovesPerPosition>& logits,
    const game::Game& game, game::Color color_to_move,
    core::Probability& probability, int k) {
  struct Entry {
    float score;  // logit + gumbel_noise
    int action;
  };

  absl::InlinedVector<Entry, constants::kMaxMovesPerPosition> entries;
  entries.reserve(constants::kMaxMovesPerPosition);

  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (!game.IsValidMove(a, color_to_move)) continue;
    entries.push_back({logits[a] + probability.GumbelSample(), a});
  }

  int k_actual = std::min(k, static_cast<int>(entries.size()));
  std::partial_sort(
      entries.begin(), entries.begin() + k_actual, entries.end(),
      [](const Entry& a, const Entry& b) { return a.score > b.score; });

  absl::InlinedVector<int, 16> top_k;
  for (int i = 0; i < k_actual; ++i) {
    top_k.push_back(entries[i].action);
  }
  return top_k;
}

// Builds a Board from a raw position array.
// Plays all BLACK stones first (valid since the board is empty, so no
// self-captures), then all WHITE stones (valid since every white stone in a
// legal Go position has at least one empty or same-color adjacent
// intersection—specifically, the not-yet-placed white stones fill those
// liberties).
game::Board BuildBoard(const std::array<Color, constants::kNumBoardLocs>& pos,
                       float komi) {
  game::Board board(komi);
  for (int i = 0; i < constants::kNumBoardLocs; ++i) {
    if (pos[i] == BLACK) {
      auto res = board.PlayMove(AsLoc(i), BLACK);
      CHECK(MoveOk(res)) << "Failed to place black stone at index " << i;
    }
  }
  for (int i = 0; i < constants::kNumBoardLocs; ++i) {
    if (pos[i] == WHITE) {
      auto res = board.PlayMove(AsLoc(i), WHITE);
      CHECK(MoveOk(res)) << "Failed to place white stone at index " << i;
    }
  }
  return board;
}

struct SearchResult {
  std::array<float, constants::kMaxMovesPerPosition> pi_improved;
  game::Loc best_move;
  uint32_t visits = 0;
  // Populated only by RunGroundTruth.
  std::array<float, constants::kMaxMovesPerPosition> prior_probs{};
  game::Loc prior_best_move = game::kNoopLoc;
};

// Runs a fresh ground-truth Gumbel search (no PUCT seed).
// Uses `Probability(pos_seed)` so that the top-K action selection matches
// RunSeededGumbel for the same position.
SearchResult RunGroundTruth(GumbelEvaluator& evaluator, Game& game,
                            Color color_to_move, uint64_t pos_seed) {
  auto node_table = std::make_unique<MctsNodeTable>();
  TreeNode* root =
      node_table->GetOrCreate(game.board().hash(), color_to_move, false);

  core::Probability prob(pos_seed);
  auto result =
      evaluator.SearchRoot(prob, game, node_table.get(), root, color_to_move,
                           GumbelSearchParams{kGroundTruthVisits, kGumbelK});

  // Find argmax of prior over valid moves.
  game::Loc prior_best = game::kPassLoc;
  float prior_best_prob = -1.f;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (game.IsValidMove(a, color_to_move) &&
        root->move_probs[a] > prior_best_prob) {
      prior_best_prob = root->move_probs[a];
      prior_best = AsLoc(a);
    }
  }

  SearchResult sr;
  sr.pi_improved = ComputeImprovedPolicy(root, 0);
  sr.best_move = result.mcts_move;
  sr.prior_probs = root->move_probs;
  sr.prior_best_move = prior_best;
  return sr;
}

// Seeds the tree with PUCT visits (simulating self-play tree reuse), then runs
// Gumbel(n, k) using the same fixed per-position seed.
SearchResult RunSeededGumbel(GumbelEvaluator& evaluator, Game& game,
                             Color color_to_move, int n, int k, int seed_visits,
                             uint64_t pos_seed, bool early_stopping = false) {
  auto node_table = std::make_unique<MctsNodeTable>();
  TreeNode* root =
      node_table->GetOrCreate(game.board().hash(), color_to_move, false);

  // PUCT seeding: use an independent probability object (we don't need this
  // to be reproducible; it just warms up the tree).
  if (seed_visits > 0) {
    core::Probability puct_prob;
    evaluator.SearchRootPuct(
        puct_prob, game, node_table.get(), root, color_to_move, seed_visits,
        PuctParams::Builder().set_kind(PuctRootSelectionPolicy::kLcb).build());
  }

  // Gumbel search: use the fixed per-position seed so that the Gumbel noise
  // (and thus the initial top-K action selection) matches RunGroundTruth.
  core::Probability gumbel_prob(pos_seed);
  auto params = GumbelSearchParams::Builder()
                    .set_n(n)
                    .set_k(k)
                    .set_early_stopping_enabled(early_stopping)
                    .build();
  auto result = evaluator.SearchRoot(gumbel_prob, game, node_table.get(), root,
                                     color_to_move, params);
  return {ComputeImprovedPolicy(root, 0), result.mcts_move, result.visits};
}

// Returns the normalized empirical visit distribution: child_visits[a] / sum.
std::array<float, constants::kMaxMovesPerPosition> EmpiricalVisitDist(
    const TreeNode* root) {
  std::array<float, constants::kMaxMovesPerPosition> dist{};
  int total = 0;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    total += root->child_visits[a];
  }
  if (total == 0) return dist;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    dist[a] = static_cast<float>(root->child_visits[a]) / total;
  }
  return dist;
}

static const PuctParams kStudyPuctParams =
    PuctParams::Builder()
        .set_root_fpu(0.1f)
        .set_p_opt_weight(1.0)
        .set_enable_var_scaling(true)
        .set_var_scale_prior_visits(10)
        .set_kind(PuctRootSelectionPolicy::kLcb)
        .build();

// Runs a PUCT search with n visits and returns the empirical visit
// distribution. Populates prior_probs and prior_best_move from the root after
// NN evaluation.
SearchResult RunPuct(GumbelEvaluator& evaluator, Game& game,
                     Color color_to_move, int n) {
  auto node_table = std::make_unique<MctsNodeTable>();
  TreeNode* root =
      node_table->GetOrCreate(game.board().hash(), color_to_move, false);

  core::Probability prob;
  auto result = evaluator.SearchRootPuct(prob, game, node_table.get(), root,
                                         color_to_move, n, kStudyPuctParams);

  game::Loc prior_best = game::kPassLoc;
  float prior_best_prob = -1.f;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (game.IsValidMove(a, color_to_move) &&
        root->move_probs[a] > prior_best_prob) {
      prior_best_prob = root->move_probs[a];
      prior_best = AsLoc(a);
    }
  }

  SearchResult sr;
  sr.pi_improved = EmpiricalVisitDist(root);
  sr.best_move = result.mcts_move;
  sr.prior_probs = root->move_probs;
  sr.prior_best_move = prior_best;
  return sr;
}

// Returns the top-k (loc, prob) pairs from a policy array, sorted by prob desc.
// Only considers valid moves.
absl::InlinedVector<std::pair<game::Loc, float>, 5> TopKMoves(
    const std::array<float, constants::kMaxMovesPerPosition>& probs,
    const game::Game& game, game::Color color_to_move, int k) {
  absl::InlinedVector<std::pair<game::Loc, float>,
                      constants::kMaxMovesPerPosition>
      entries;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (game.IsValidMove(a, color_to_move)) {
      entries.push_back({AsLoc(a), probs[a]});
    }
  }
  int k_actual = std::min(k, static_cast<int>(entries.size()));
  std::partial_sort(
      entries.begin(), entries.begin() + k_actual, entries.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });
  return {entries.begin(), entries.begin() + k_actual};
}

struct Stats {
  float mean;
  float p75;
  float p95;
  float max;
};

Stats ComputeStats(std::vector<float>& vals) {
  if (vals.empty()) return {0.f, 0.f, 0.f, 0.f};
  std::sort(vals.begin(), vals.end());
  const int n = static_cast<int>(vals.size());
  float mean =
      std::accumulate(vals.begin(), vals.end(), 0.f) / static_cast<float>(n);
  auto percentile = [&](float p) -> float {
    float idx = p * (n - 1);
    int lo = static_cast<int>(idx);
    int hi = std::min(lo + 1, n - 1);
    float frac = idx - lo;
    return vals[lo] + frac * (vals[hi] - vals[lo]);
  };
  return Stats{mean, percentile(0.75f), percentile(0.95f), vals.back()};
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string chunk_path = absl::GetFlag(FLAGS_chunk_path);
  const int num_examples = absl::GetFlag(FLAGS_num_examples);
  const int seed_visits = absl::GetFlag(FLAGS_seed_visits);
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  const bool use_gumbel = absl::GetFlag(FLAGS_use_gumbel);

  if (model_path.empty()) {
    LOG(ERROR) << "--model_path not specified.";
    return 1;
  }
  if (chunk_path.empty()) {
    LOG(ERROR) << "--chunk_path not specified.";
    return 1;
  }

  auto engine = nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                                 /*batch_size=*/1,
                                 nn::GetVersionFromModelPath(model_path));
  auto nn_interface = std::make_unique<nn::NNInterface>(
      /*num_threads=*/1, kTimeoutUs, kCacheSize, std::move(engine));

  // kld_samples[ni] = KLD(GT, dist_at_N) per example.
  std::vector<std::vector<float>> kld_samples(kNumNValues);
  // kld_prior_dist_samples[ni] = KLD(dist_at_N, prior) per example.
  std::vector<std::vector<float>> kld_prior_dist_samples(kNumNValues);
  // kld_prior_samples[i] = KLD(GT, nn_prior) for example i.
  std::vector<float> kld_prior_samples;
  // N=800 + early stopping.
  std::vector<float> kld_es_samples;
  std::vector<float> kld_prior_dist_es_samples;
  std::vector<float> visits_es_samples;

  SequentialRecordReader reader(chunk_path, RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok()) << "Failed to open: " << chunk_path;

  int processed = 0;
  int skipped = 0;
  std::string record;

  while (processed < num_examples) {
    auto status = reader.ReadRecord(&record);
    if (absl::IsOutOfRange(status)) {
      LOG(WARNING) << "Reached EOF after " << processed << " examples.";
      break;
    }
    CHECK(status.ok()) << "Read error: " << status;

    Example example;
    if (!example.ParseFromString(record)) {
      LOG(WARNING) << "Failed to parse example " << processed << ", skipping.";
      ++skipped;
      continue;
    }

    // Parse board features.
    const auto& board_feat =
        GetFeatureValues<std::string>("board", example).Get(0);
    const auto& last_moves_feat =
        GetFeatureValues<std::string>("last_moves", example).Get(0);
    const auto& color_feat =
        GetFeatureValues<std::string>("color", example).Get(0);
    const float komi = GetFeatureValues<float>("komi", example).Get(0);

    const auto board_pos =
        ParseSequence<Color, constants::kNumBoardLocs>(board_feat);
    const Color color_to_move = ParseScalar<Color>(color_feat);
    const auto last_move_encodings =
        ParseSequence<int16_t, constants::kNumLastMoves>(last_moves_feat);

    // Reconstruct board and game.
    // NNInterface reads only .loc from last_moves, so color is a placeholder.
    game::Board board = BuildBoard(board_pos, komi);
    absl::InlinedVector<game::Move, constants::kNumLastMoves> last_moves;
    for (int i = 0; i < constants::kNumLastMoves; ++i) {
      last_moves.push_back(
          game::Move{color_to_move, AsLoc(last_move_encodings[i])});
    }
    Game game(board, last_moves, /*init_mv_num=*/0);

    // Ground truth.
    SearchResult gt;
    if (use_gumbel) {
      // Fixed per-position seed: same K=16 actions across GT and N-visit runs.
      const uint64_t pos_seed =
          0xdeadbeef12345678ULL ^ static_cast<uint64_t>(processed);
      // alpha=0.85, lambda=0.45 for PUCT; alpha=0.8, lambda=0.3 for Gumbel.
      BiasCache bias_cache(use_gumbel ? 0.8f : 0.85f,
                           use_gumbel ? 0.3f : 0.45f);
      GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                &bias_cache);
      gt = RunGroundTruth(evaluator, game, color_to_move, pos_seed);
    } else {
      // alpha=0.85, lambda=0.45 for PUCT; alpha=0.8, lambda=0.3 for Gumbel.
      BiasCache bias_cache(use_gumbel ? 0.8f : 0.85f,
                           use_gumbel ? 0.3f : 0.45f);
      GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                &bias_cache);
      gt = RunPuct(evaluator, game, color_to_move, kGroundTruthVisits);
    }

    const float kld_prior = ComputeKLD(gt.pi_improved, gt.prior_probs);
    kld_prior_samples.push_back(kld_prior);

    if (verbose) {
      printf("\n=== Example %d (color: %s) ===\n", processed,
             color_to_move == BLACK ? "black" : "white");
      printf("%s\n", game::ToString(game.board().position()).c_str());

      auto prior_top5 = TopKMoves(gt.prior_probs, game, color_to_move, 5);
      printf("  Prior top-5:");
      for (auto& [loc, prob] : prior_top5) {
        printf("  %s(%.3f)", LocToString(loc).c_str(), prob);
      }
      printf("\n");

      auto gt_top5 = TopKMoves(gt.pi_improved, game, color_to_move, 5);
      printf("  GT top-5:   ");
      for (auto& [loc, prob] : gt_top5) {
        printf("  %s(%.3f)", LocToString(loc).c_str(), prob);
      }
      printf("  KLD(GT,prior): %.5f\n", kld_prior);
    }

    // Collect search results for all N values before printing.
    std::vector<SearchResult> n_results(kNumNValues);
    for (int ni = 0; ni < kNumNValues; ++ni) {
      if (use_gumbel) {
        // alpha=0.85, lambda=0.45 for PUCT; alpha=0.8, lambda=0.3 for Gumbel.
        BiasCache bias_cache(use_gumbel ? 0.8f : 0.85f,
                             use_gumbel ? 0.3f : 0.45f);
        GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                  &bias_cache);
        const uint64_t pos_seed =
            0xdeadbeef12345678ULL ^ static_cast<uint64_t>(processed);
        n_results[ni] =
            RunSeededGumbel(evaluator, game, color_to_move, kNValues[ni],
                            kGumbelK, seed_visits, pos_seed);
      } else {
        // alpha=0.85, lambda=0.45 for PUCT; alpha=0.8, lambda=0.3 for Gumbel.
        BiasCache bias_cache(use_gumbel ? 0.8f : 0.85f,
                             use_gumbel ? 0.3f : 0.45f);
        GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                  &bias_cache);
        n_results[ni] = RunPuct(evaluator, game, color_to_move, kNValues[ni]);
      }
      const float kld = ComputeKLD(gt.pi_improved, n_results[ni].pi_improved);
      const float kld_prior_dist =
          ComputeKLD(n_results[ni].pi_improved, gt.prior_probs);
      kld_samples[ni].push_back(kld);
      kld_prior_dist_samples[ni].push_back(kld_prior_dist);
    }

    // N=800 + early stopping.
    SearchResult sr_es;
    if (use_gumbel) {
      const uint64_t pos_seed =
          0xdeadbeef12345678ULL ^ static_cast<uint64_t>(processed);
      BiasCache bias_cache(0.8f, 0.3f);
      GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                &bias_cache);
      sr_es = RunSeededGumbel(evaluator, game, color_to_move, 800, kGumbelK,
                              seed_visits, pos_seed, /*early_stopping=*/true);
    } else {
      BiasCache bias_cache(0.85f, 0.45f);
      GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                &bias_cache);
      sr_es = RunPuct(evaluator, game, color_to_move, 800);
    }
    kld_es_samples.push_back(ComputeKLD(gt.pi_improved, sr_es.pi_improved));
    kld_prior_dist_es_samples.push_back(
        ComputeKLD(sr_es.pi_improved, gt.prior_probs));
    visits_es_samples.push_back(static_cast<float>(sr_es.visits));
    printf("  800es actual_visits: %u\n", sr_es.visits);

    if (verbose) {
      // nats/visit summary for all N values.
      for (int ni = 0; ni < kNumNValues; ++ni) {
        const float kld = ComputeKLD(gt.pi_improved, n_results[ni].pi_improved);
        const float kld_pd =
            ComputeKLD(n_results[ni].pi_improved, gt.prior_probs);
        printf(
            "  N=%-4d  best: %-5s  KLD(GT||N): %.5f  KLD(N||prior): %.5f  "
            "nats/visit: %.6f\n",
            kNValues[ni], LocToString(n_results[ni].best_move).c_str(), kld,
            kld_pd, (kld_prior - kld) / kNValues[ni]);
      }
      {
        const float kld = ComputeKLD(gt.pi_improved, sr_es.pi_improved);
        const float kld_pd = ComputeKLD(sr_es.pi_improved, gt.prior_probs);
        printf(
            "  N=%-4s  best: %-5s  KLD(GT||N): %.5f  KLD(N||prior): %.5f  "
            "nats/visit: %.6f\n",
            "800es", LocToString(sr_es.best_move).c_str(), kld, kld_pd,
            (kld_prior - kld) / 800);
      }
      // Top-5 moves for all N values.
      printf("\n");
      for (int ni = 0; ni < kNumNValues; ++ni) {
        auto top5 =
            TopKMoves(n_results[ni].pi_improved, game, color_to_move, 5);
        printf("  N=%-4d  top-5:", kNValues[ni]);
        for (auto& [loc, prob] : top5) {
          printf("  %s(%.3f)", LocToString(loc).c_str(), prob);
        }
        printf("\n");
      }
      {
        auto top5 = TopKMoves(sr_es.pi_improved, game, color_to_move, 5);
        printf("  N=%-4s  top-5:", "800es");
        for (auto& [loc, prob] : top5) {
          printf("  %s(%.3f)", LocToString(loc).c_str(), prob);
        }
        printf("\n");
      }
    }

    ++processed;
    if (processed % 50 == 0) {
      LOG(INFO) << "Processed " << processed << "/" << num_examples
                << " examples.";
    }
  }

  LOG(INFO) << "Done. Processed " << processed << " examples, skipped "
            << skipped << ".";

  Stats prior_stats = ComputeStats(kld_prior_samples);

  printf("\nVisit Count Study\n");
  if (use_gumbel) {
    printf("Mode:         Gumbel (GT: N=%d K=%d, seed: %d PUCT visits)\n",
           kGroundTruthVisits, kGumbelK, seed_visits);
  } else {
    printf("Mode:         PUCT (GT: N=%d fpu=0.1, empirical visit dist)\n",
           kGroundTruthVisits);
  }
  printf("Examples:     %d\n\n", processed);
  printf("KLD(GT, prior) -- mean: %.5f  p75: %.5f  p95: %.5f  max: %.5f\n\n",
         prior_stats.mean, prior_stats.p75, prior_stats.p95, prior_stats.max);
  printf("%-8s  %-12s  %-12s  %-12s  %-12s\n", "N", "KLD(GT||N)",
         "KLD(N||prior)", "nats/visit", "mean_visits");
  printf("%-8s  %-12s  %-12s  %-12s  %-12s\n", "--------", "------------",
         "------------", "------------", "------------");
  for (int ni = 0; ni < kNumNValues; ++ni) {
    Stats s = ComputeStats(kld_samples[ni]);
    Stats sp = ComputeStats(kld_prior_dist_samples[ni]);
    float nats_per_visit = (prior_stats.mean - s.mean) / kNValues[ni];
    printf("%-8d  %-12.5f  %-12.5f  %-12.6f  %-12d\n", kNValues[ni], s.mean,
           sp.mean, nats_per_visit, kNValues[ni]);
  }
  {
    Stats s = ComputeStats(kld_es_samples);
    Stats sp = ComputeStats(kld_prior_dist_es_samples);
    Stats sv = ComputeStats(visits_es_samples);
    float nats_per_visit = (prior_stats.mean - s.mean) / sv.mean;
    printf("%-8s  %-12.5f  %-12.5f  %-12.6f  %-12.1f\n", "800es", s.mean,
           sp.mean, nats_per_visit, sv.mean);
  }

  return 0;
}
