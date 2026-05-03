// Experiment: Upside vs. variance bonus study.
//
// For each position loaded from a chunk, runs three PUCT searches:
//   (1) base:      4000 visits, root_fpu=0.1
//   (2) var_cpuct: 600 visits,  root_fpu=0.1, enable_var_scaling,
//   prior_visits=10 (3) v_cat_var: 600 visits,  root_fpu=0.1,
//   enable_v_cat_var_scaling, prior_visits=10
//
// Reports, per position: top-5 moves with visit counts/fractions and the
// LCB-selected move for each configuration.
//
// Summary stats: agreement rate of each 600-visit method vs. the 4k base.
//
// All three searches use the same per-position seed for reproducibility.

#include <algorithm>
#include <array>
#include <cmath>
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
ABSL_FLAG(int, num_examples, 100, "Number of examples to process.");
ABSL_FLAG(int, skip_first, 0, "Number of examples to skip before processing.");
ABSL_FLAG(bool, verbose, true, "If true, print per-example details.");

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
using ::mcts::MctsNodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::TreeNode;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

static constexpr int kBaseVisits = 4000;
static constexpr int kShortVisits = 600;
static constexpr int kShortBaseVisits = 600;
static constexpr size_t kCacheSize = 32768;
static constexpr int64_t kTimeoutUs = 400;
static constexpr int kTopK = 5;

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

std::string LocToString(game::Loc loc) {
  if (loc == game::kPassLoc) return "pass";
  if (loc == game::kNoopLoc) return "noop";
  static constexpr char kCols[] = "ABCDEFGHIJKLMNOPQRST";
  return std::string(1, kCols[loc.j]) + std::to_string(loc.i);
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

struct ChildInfo {
  int action;
  int visits;
  float q;      // Q from the root's perspective = -child->v
  float q_var;  // child->v_var
};

struct SearchResult {
  game::Loc lcb_move;
  // Sorted descending by visits.
  std::vector<ChildInfo> children;
  int total_visits = 0;
};

SearchResult RunPuct(GumbelEvaluator& evaluator, game::Game& game,
                     Color color_to_move, int n, const PuctParams& params,
                     uint64_t pos_seed) {
  auto node_table = std::make_unique<mcts::MctsNodeTable>();
  TreeNode* root =
      node_table->GetOrCreate(game.board().hash(), color_to_move, false);

  core::Probability prob(pos_seed);
  auto result = evaluator.SearchRootPuct(prob, game, node_table.get(), root,
                                         color_to_move, n, params);

  SearchResult sr;
  sr.lcb_move = result.mcts_move;

  int total = 0;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    int visits = root->child_visits[a];
    if (visits > 0) {
      const TreeNode* child = root->child(a);
      float q = child != nullptr ? -child->v : 0.f;
      float q_var = child != nullptr ? child->v_var : 0.f;
      sr.children.push_back({a, visits, q, q_var});
      total += visits;
    }
  }
  std::sort(sr.children.begin(), sr.children.end(),
            [](const auto& a, const auto& b) { return a.visits > b.visits; });
  sr.total_visits = total;
  return sr;
}

void PrintVisitDist(const SearchResult& sr, const game::Game& game,
                    Color color_to_move, int k) {
  int shown = std::min(k, static_cast<int>(sr.children.size()));
  for (int i = 0; i < shown; ++i) {
    const auto& c = sr.children[i];
    float frac = sr.total_visits > 0
                     ? static_cast<float>(c.visits) / sr.total_visits
                     : 0.f;
    printf("    [%d] %s  visits=%d (%.3f)  q=%.4f  var=%.4f\n", i + 1,
           LocToString(AsLoc(c.action)).c_str(), c.visits, frac, c.q, c.q_var);
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string chunk_path = absl::GetFlag(FLAGS_chunk_path);
  const int num_examples = absl::GetFlag(FLAGS_num_examples);
  const int skip_first = absl::GetFlag(FLAGS_skip_first);
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

  // Configuration 1: base, 4000 visits.
  const PuctParams kBaseParams = PuctParams::Builder()
                                     .set_root_fpu(0.1f)
                                     .set_kind(PuctRootSelectionPolicy::kLcb)
                                     .build();

  // Configuration 2: short base, 600 visits, no modifiers.
  const PuctParams kShortBaseParams =
      PuctParams::Builder()
          .set_root_fpu(0.1f)
          .set_kind(PuctRootSelectionPolicy::kLcb)
          .build();

  // Configuration 3: var_cpuct (variance-scaled c_puct), 600 visits.
  const PuctParams kVarCpuctParams =
      PuctParams::Builder()
          .set_root_fpu(0.1f)
          .set_kind(PuctRootSelectionPolicy::kLcb)
          .set_enable_var_scaling(true)
          .set_var_scale_prior_visits(10)
          .build();

  // Configuration 3: v_cat_var (categorical value variance scaling), 600
  // visits.
  const PuctParams kVCatVarParams = PuctParams::Builder()
                                        .set_root_fpu(0.1f)
                                        .set_kind(PuctRootSelectionPolicy::kLcb)
                                        .set_enable_v_cat_var_scaling(true)
                                        .set_v_cat_var_scale_prior_visits(10)
                                        .build();

  int base_short_base_agree = 0;
  int base_var_cpuct_agree = 0;
  int base_v_cat_var_agree = 0;
  int processed = 0;
  int skipped = 0;

  SequentialRecordReader reader(chunk_path, RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok()) << "Failed to open: " << chunk_path;

  std::string record;
  for (int i = 0; i < skip_first; ++i) {
    auto status = reader.ReadRecord(&record);
    if (absl::IsOutOfRange(status)) {
      LOG(ERROR) << "EOF while skipping at record " << i
                 << " (skip_first=" << skip_first << ").";
      return 1;
    }
    CHECK(status.ok()) << "Read error while skipping: " << status;
  }
  if (skip_first > 0) {
    LOG(INFO) << "Skipped " << skip_first << " examples.";
  }

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

    game::Board board = BuildBoard(board_pos, komi);
    absl::InlinedVector<game::Move, constants::kNumLastMoves> last_moves;
    for (int i = 0; i < constants::kNumLastMoves; ++i) {
      last_moves.push_back(
          game::Move{color_to_move, AsLoc(last_move_encodings[i])});
    }
    Game game(board, last_moves, /*init_mv_num=*/0);

    const uint64_t pos_seed =
        0xdeadbeef12345678ULL ^ static_cast<uint64_t>(processed);

    // Each search gets its own evaluator + node table to avoid state bleed.
    SearchResult base_sr, short_base_sr, var_cpuct_sr, v_cat_var_sr;
    {
      BiasCache bias_cache(0.8f, 0.3f);
      GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                &bias_cache);
      base_sr = RunPuct(evaluator, game, color_to_move, kBaseVisits,
                        kBaseParams, pos_seed);
    }
    {
      BiasCache bias_cache(0.8f, 0.3f);
      GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                &bias_cache);
      short_base_sr = RunPuct(evaluator, game, color_to_move, kShortBaseVisits,
                              kShortBaseParams, pos_seed);
    }
    {
      BiasCache bias_cache(0.8f, 0.3f);
      GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                &bias_cache);
      var_cpuct_sr = RunPuct(evaluator, game, color_to_move, kShortVisits,
                             kVarCpuctParams, pos_seed);
    }
    {
      BiasCache bias_cache(0.8f, 0.3f);
      GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                                &bias_cache);
      v_cat_var_sr = RunPuct(evaluator, game, color_to_move, kShortVisits,
                             kVCatVarParams, pos_seed);
    }

    if (short_base_sr.lcb_move == base_sr.lcb_move) ++base_short_base_agree;
    if (var_cpuct_sr.lcb_move == base_sr.lcb_move) ++base_var_cpuct_agree;
    if (v_cat_var_sr.lcb_move == base_sr.lcb_move) ++base_v_cat_var_agree;

    if (verbose) {
      printf("\n=== Example %d (color: %s) ===\n", processed,
             color_to_move == BLACK ? "black" : "white");
      printf("%s\n", game::ToString(game.board().position()).c_str());

      printf("  [base %d visits]  LCB: %s\n", kBaseVisits,
             LocToString(base_sr.lcb_move).c_str());
      PrintVisitDist(base_sr, game, color_to_move, kTopK);

      printf("  [short_base %d visits]  LCB: %s  agree=%s\n", kShortBaseVisits,
             LocToString(short_base_sr.lcb_move).c_str(),
             (short_base_sr.lcb_move == base_sr.lcb_move ? "YES" : "NO"));
      PrintVisitDist(short_base_sr, game, color_to_move, kTopK);

      printf("  [var_cpuct %d visits]  LCB: %s  agree=%s\n", kShortVisits,
             LocToString(var_cpuct_sr.lcb_move).c_str(),
             (var_cpuct_sr.lcb_move == base_sr.lcb_move ? "YES" : "NO"));
      PrintVisitDist(var_cpuct_sr, game, color_to_move, kTopK);

      printf("  [v_cat_var %d visits]  LCB: %s  agree=%s\n", kShortVisits,
             LocToString(v_cat_var_sr.lcb_move).c_str(),
             (v_cat_var_sr.lcb_move == base_sr.lcb_move ? "YES" : "NO"));
      PrintVisitDist(v_cat_var_sr, game, color_to_move, kTopK);
    }

    ++processed;
    if (processed % 20 == 0) {
      LOG(INFO) << "Processed " << processed << "/" << num_examples;
    }
  }

  LOG(INFO) << "Done. Processed " << processed << " examples, skipped "
            << skipped << ".";

  printf("\n=== Summary ===\n");
  printf("Examples:               %d\n", processed);
  printf("Base visits:            %d\n", kBaseVisits);
  printf("Short visits:           %d\n", kShortVisits);
  printf("base params:            root_fpu=0.1, LCB\n");
  printf("short_base params:      root_fpu=0.1, LCB\n");
  printf(
      "var_cpuct params:       root_fpu=0.1, enable_var_scaling, "
      "prior_visits=10, LCB\n");
  printf(
      "v_cat_var params:       root_fpu=0.1, enable_v_cat_var_scaling, "
      "prior_visits=10, LCB\n");
  printf("\n");
  printf("Agreement with base (%d visits):\n", kBaseVisits);
  printf("  short_base: %d / %d  (%.1f%%)\n", base_short_base_agree, processed,
         100.f * base_short_base_agree / processed);
  printf("  var_cpuct:  %d / %d  (%.1f%%)\n", base_var_cpuct_agree, processed,
         100.f * base_var_cpuct_agree / processed);
  printf("  v_cat_var:  %d / %d  (%.1f%%)\n", base_v_cat_var_agree, processed,
         100.f * base_v_cat_var_agree / processed);

  return 0;
}
