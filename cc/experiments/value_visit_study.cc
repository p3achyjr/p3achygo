// Experiment: Value visit count study.
//
// Loads positions from a "ground truth" tfrecord chunk (e.g.
// value_gt.tfrecord.zz). For each position, runs Gumbel(N, K=5) for N in {30,
// 40, 60, 90, 150} and measures three value-related metrics vs. stored GT
// targets:
//
//   TD-error:          |root.v - q_target| for q6, q16, q50
//   Value-dist error:  KLD(gt_mcts_dist || root.v_categorical)
//   W/L error:         CCE(z, [loss_prob, win_prob])
//
// The GT q6/q16/q50 and mcts_value_dist are read from the chunk.
// z (game outcome) is derived from score_margin > 0.
// win_prob/loss_prob are derived from root->v_outcome after N visits.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
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
#include "cc/mcts/constants.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"
#include "cc/proto/feature_util.h"
#include "example.pb.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(std::string, chunk_path, "",
          "Path to value_gt tfrecord chunk (.tfrecord.zz).");
ABSL_FLAG(int, num_examples, 500, "Number of examples to process.");
ABSL_FLAG(int, seed_visits, 16,
          "PUCT visits to seed the tree before Gumbel search.");
ABSL_FLAG(bool, verbose, false, "If true, print per-example metrics.");

namespace {

using ::data::RecordReaderOptions;
using ::data::SequentialRecordReader;
using ::game::AsLoc;
using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::game::MoveOk;
using ::mcts::BiasCache;
using ::mcts::GumbelEvaluator;
using ::mcts::GumbelSearchParams;
using ::mcts::MctsNodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::TreeNode;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

static constexpr int kGumbelK = 5;
static constexpr size_t kCacheSize = 32768;
static constexpr int64_t kTimeoutUs = 400;
static constexpr float kEps = 1e-10f;

static constexpr int kNValues[] = {30, 40, 60, 90, 150};
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
  game::Loc best_move;
  uint32_t visits = 0;
  float v = 0;          // root mean Q (utility value) in [-1, 1]
  float v_outcome = 0;  // root mean outcome: P(win) - P(loss) in [-1, 1]
  std::array<uint32_t, mcts::kNumVBuckets> v_categorical{};
};

SearchResult RunSeededGumbel(GumbelEvaluator& evaluator, Game& game,
                             Color color_to_move, int n, int seed_visits,
                             uint64_t pos_seed) {
  auto node_table = std::make_unique<MctsNodeTable>();
  TreeNode* root =
      node_table->GetOrCreate(game.board().hash(), color_to_move, false);

  if (seed_visits > 0) {
    core::Probability puct_prob;
    evaluator.SearchRootPuct(
        puct_prob, game, node_table.get(), root, color_to_move, seed_visits,
        PuctParams::Builder().set_kind(PuctRootSelectionPolicy::kLcb).build());
  }

  core::Probability gumbel_prob(pos_seed);
  auto params = GumbelSearchParams::Builder().set_n(n).set_k(kGumbelK).build();
  auto result = evaluator.SearchRoot(gumbel_prob, game, node_table.get(), root,
                                     color_to_move, params);

  SearchResult sr;
  sr.best_move = result.mcts_move;
  sr.visits = result.visits;
  sr.v = root->v;
  sr.v_outcome = root->v_outcome;
  sr.v_categorical = root->v_categorical;
  return sr;
}

// Evaluates the root with SearchRoot(N=1) to get the raw NN estimate.
// Returns init_outcome_est and v_categorical from the root.
SearchResult RunNNOnly(nn::NNInterface* nn_interface, Game& game,
                       Color color_to_move) {
  auto node_table = std::make_unique<MctsNodeTable>();
  TreeNode* root =
      node_table->GetOrCreate(game.board().hash(), color_to_move, false);

  GumbelEvaluator evaluator(nn_interface, /*thread_id=*/0);
  core::Probability prob;
  auto params = GumbelSearchParams::Builder().set_n(1).set_k(kGumbelK).build();
  evaluator.SearchRoot(prob, game, node_table.get(), root, color_to_move,
                       params);

  SearchResult sr;
  sr.best_move = game::kNoopLoc;
  sr.visits = 0;
  sr.v = root->init_outcome_est;
  sr.v_outcome = root->init_outcome_est;
  sr.v_categorical = root->v_categorical;
  return sr;
}

// Expected value of a uint32 bucket distribution over [-1, 1].
// Returns 0 if the distribution is empty.
float ComputeExpectedV(const std::array<uint32_t, mcts::kNumVBuckets>& counts) {
  float total = 0;
  for (int i = 0; i < mcts::kNumVBuckets; ++i) {
    total += static_cast<float>(counts[i]);
  }
  if (total == 0) return 0.f;
  float ev = 0;
  for (int i = 0; i < mcts::kNumVBuckets; ++i) {
    float center = (i + 0.5f) * mcts::kBucketRange - 1.0f;
    ev += (static_cast<float>(counts[i]) / total) * center;
  }
  return ev;
}

// KLD(p || q) where p and q are uint32 count arrays normalized to distributions
// internally. Returns 0 if either distribution is empty.
float ComputeVDistKLD(
    const std::array<uint32_t, mcts::kNumVBuckets>& p_counts,
    const std::array<uint32_t, mcts::kNumVBuckets>& q_counts) {
  float p_sum = 0, q_sum = 0;
  for (int i = 0; i < mcts::kNumVBuckets; ++i) {
    p_sum += static_cast<float>(p_counts[i]);
    q_sum += static_cast<float>(q_counts[i]);
  }
  if (p_sum == 0 || q_sum == 0) return 0.f;

  float kld = 0;
  for (int i = 0; i < mcts::kNumVBuckets; ++i) {
    float p = static_cast<float>(p_counts[i]) / p_sum;
    float q = static_cast<float>(q_counts[i]) / q_sum;
    if (p < kEps) continue;
    kld += p * std::log(p / (q + kEps));
  }
  return kld;
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

  std::vector<std::vector<float>> td_q6_samples(kNumNValues);
  std::vector<std::vector<float>> td_q16_samples(kNumNValues);
  std::vector<std::vector<float>> td_q50_samples(kNumNValues);
  std::vector<std::vector<float>> vdist_kld_samples(kNumNValues);
  std::vector<std::vector<float>> cce_samples(kNumNValues);
  std::vector<std::vector<float>> v_outcome_mse_samples(kNumNValues);

  // Raw NN baseline (no search).
  std::vector<float> nn_td_q6_samples;
  std::vector<float> nn_td_q16_samples;
  std::vector<float> nn_td_q50_samples;
  std::vector<float> nn_vdist_kld_samples;
  std::vector<float> nn_cce_samples;
  std::vector<float> nn_v_outcome_mse_samples;

  // Read all records, shuffle, then process num_examples.
  std::vector<std::string> all_records;
  {
    SequentialRecordReader reader(chunk_path, RecordReaderOptions::Zlib());
    CHECK(reader.Init().ok()) << "Failed to open: " << chunk_path;
    std::string record;
    while (true) {
      auto status = reader.ReadRecord(&record);
      if (absl::IsOutOfRange(status)) break;
      CHECK(status.ok()) << "Read error: " << status;
      all_records.push_back(record);
    }
  }
  LOG(INFO) << "Loaded " << all_records.size() << " records. Shuffling.";
  std::shuffle(all_records.begin(), all_records.end(), std::mt19937{42});

  int processed = 0;
  int skipped = 0;

  const int limit =
      std::min(static_cast<int>(all_records.size()), num_examples);
  for (int i = 0; i < limit; ++i) {
    const std::string& record = all_records[i];

    Example example;
    if (!example.ParseFromString(record)) {
      LOG(WARNING) << "Failed to parse example " << processed << ", skipping.";
      ++skipped;
      continue;
    }

    // Board features.
    const auto& board_feat =
        GetFeatureValues<std::string>("board", example).Get(0);
    const auto& last_moves_feat =
        GetFeatureValues<std::string>("last_moves", example).Get(0);
    const auto& color_feat =
        GetFeatureValues<std::string>("color", example).Get(0);
    const float komi = GetFeatureValues<float>("komi", example).Get(0);

    // GT value targets.
    const float q6 = GetFeatureValues<float>("q6", example).Get(0);
    const float q16 = GetFeatureValues<float>("q16", example).Get(0);
    const float q50 = GetFeatureValues<float>("q50", example).Get(0);
    const float score_margin =
        GetFeatureValues<float>("score_margin", example).Get(0);
    const auto& mcts_dist_feat =
        GetFeatureValues<std::string>("mcts_value_dist", example).Get(0);
    const auto gt_mcts_dist =
        ParseSequence<uint32_t, mcts::kNumVBuckets>(mcts_dist_feat);

    const auto board_pos =
        ParseSequence<Color, constants::kNumBoardLocs>(board_feat);
    const Color color_to_move = ParseScalar<Color>(color_feat);
    const auto last_move_encodings =
        ParseSequence<int16_t, constants::kNumLastMoves>(last_moves_feat);

    game::Board board = BuildBoard(board_pos, komi);
    absl::InlinedVector<game::Move, constants::kNumLastMoves> last_moves;
    for (int i = 0; i < constants::kNumLastMoves; ++i) {
      last_moves.push_back(
          game::Move{color_to_move, AsLoc(last_move_encodings[i])});
    }
    Game game(board, last_moves, /*init_mv_num=*/0);

    const float gt_v_outcome = ComputeExpectedV(gt_mcts_dist);
    const bool z_win = score_margin > 0;
    const uint64_t pos_seed =
        0xdeadbeef12345678ULL ^ static_cast<uint64_t>(processed);

    if (verbose) {
      printf(
          "\n=== Example %d (color: %s, q6=%.3f q16=%.3f q50=%.3f z=%s) "
          "===\n",
          processed, color_to_move == BLACK ? "black" : "white", q6, q16, q50,
          z_win ? "win" : "loss");
      printf("%s\n", game::ToString(game.board().position()).c_str());
    }

    {
      SearchResult sr = RunNNOnly(nn_interface.get(), game, color_to_move);

      nn_td_q6_samples.push_back(std::abs(sr.v - q6));
      nn_td_q16_samples.push_back(std::abs(sr.v - q16));
      nn_td_q50_samples.push_back(std::abs(sr.v - q50));
      nn_vdist_kld_samples.push_back(
          ComputeVDistKLD(gt_mcts_dist, sr.v_categorical));
      float win_prob = std::clamp((sr.v_outcome + 1.f) / 2.f, kEps, 1.f - kEps);
      float loss_prob = 1.f - win_prob;
      nn_cce_samples.push_back(z_win ? -std::log(win_prob)
                                     : -std::log(loss_prob));
      float v_outcome_err = sr.v_outcome - gt_v_outcome;
      nn_v_outcome_mse_samples.push_back(v_outcome_err * v_outcome_err);

      if (verbose) {
        printf(
            "  N=%-4s  v=%.4f  |v-q6|=%.4f  |v-q16|=%.4f  |v-q50|=%.4f"
            "  vdist_kld=%.5f  cce=%.4f\n",
            "nn", sr.v, std::abs(sr.v - q6), std::abs(sr.v - q16),
            std::abs(sr.v - q50), nn_vdist_kld_samples.back(),
            nn_cce_samples.back());
      }
    }

    for (int ni = 0; ni < kNumNValues; ++ni) {
      BiasCache bias_cache(0.8f, 0.3f);
      GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                &bias_cache);
      SearchResult sr = RunSeededGumbel(evaluator, game, color_to_move,
                                        kNValues[ni], seed_visits, pos_seed);

      td_q6_samples[ni].push_back(std::abs(sr.v - q6));
      td_q16_samples[ni].push_back(std::abs(sr.v - q16));
      td_q50_samples[ni].push_back(std::abs(sr.v - q50));
      vdist_kld_samples[ni].push_back(
          ComputeVDistKLD(gt_mcts_dist, sr.v_categorical));

      float win_prob = std::clamp((sr.v_outcome + 1.f) / 2.f, kEps, 1.f - kEps);
      float loss_prob = 1.f - win_prob;
      float cce = z_win ? -std::log(win_prob) : -std::log(loss_prob);
      cce_samples[ni].push_back(cce);

      float v_outcome_err = sr.v_outcome - gt_v_outcome;
      v_outcome_mse_samples[ni].push_back(v_outcome_err * v_outcome_err);

      if (verbose) {
        printf(
            "  N=%-4d  v=%.4f  |v-q6|=%.4f  |v-q16|=%.4f  |v-q50|=%.4f"
            "  vdist_kld=%.5f  cce=%.4f\n",
            kNValues[ni], sr.v, std::abs(sr.v - q6), std::abs(sr.v - q16),
            std::abs(sr.v - q50), vdist_kld_samples[ni].back(), cce);
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

  printf("\nValue Visit Study\n");
  printf("Mode: Gumbel K=%d, seed=%d PUCT visits\n", kGumbelK, seed_visits);
  printf("Examples: %d\n\n", processed);

  printf("%-8s  %-12s  %-12s  %-12s  %-12s  %-10s  %-12s\n", "N", "|v-q6|",
         "|v-q16|", "|v-q50|", "vdist_kld", "cce", "v_out_mse");
  printf("%-8s  %-12s  %-12s  %-12s  %-12s  %-10s  %-12s\n", "--------",
         "------------", "------------", "------------", "------------",
         "----------", "------------");
  {
    Stats s6 = ComputeStats(nn_td_q6_samples);
    Stats s16 = ComputeStats(nn_td_q16_samples);
    Stats s50 = ComputeStats(nn_td_q50_samples);
    Stats sv = ComputeStats(nn_vdist_kld_samples);
    Stats sc = ComputeStats(nn_cce_samples);
    Stats sm = ComputeStats(nn_v_outcome_mse_samples);
    printf("%-8s  %-12.5f  %-12.5f  %-12.5f  %-12.5f  %-10.5f  %-12.5f\n", "nn",
           s6.mean, s16.mean, s50.mean, sv.mean, sc.mean, sm.mean);
  }
  for (int ni = 0; ni < kNumNValues; ++ni) {
    Stats s6 = ComputeStats(td_q6_samples[ni]);
    Stats s16 = ComputeStats(td_q16_samples[ni]);
    Stats s50 = ComputeStats(td_q50_samples[ni]);
    Stats sv = ComputeStats(vdist_kld_samples[ni]);
    Stats sc = ComputeStats(cce_samples[ni]);
    Stats sm = ComputeStats(v_outcome_mse_samples[ni]);
    printf("%-8d  %-12.5f  %-12.5f  %-12.5f  %-12.5f  %-10.5f  %-12.5f\n",
           kNValues[ni], s6.mean, s16.mean, s50.mean, sv.mean, sc.mean,
           sm.mean);
  }
  return 0;
}
