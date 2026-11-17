// Experiment: node variance dump.
//
// Runs a PUCT search (eval base config: LCB, p_opt_weight 1, root_fpu 0.1,
// bias cache 0.85/0.45, no child variance scaling) on positions from a
// tfrecord chunk, then walks every tree and records (N, v_var, depth) for
// every node with N >= 3. Also records the NN's predicted short-term outcome
// error err = sqrt(q6_err) of every evaluated node and the KataGo-style leaf
// weight c / (err + c / max_weight) it implies. Output:
//   --csv_path   optional per-node CSV: example,depth,n,v_var
//   stdout       per-log2(N)-bin stats of sqrt(v_var): count, mean, median,
//                mean of v_var, and the resulting s_ref table.
//   --s_ref_path optional file with kSRefBins whitespace-separated floats
//                (mean sqrt(v_var) per bin; empty bins filled with the nearest
//                populated lower bin, or --s_ref_default if none).
//
// This calibrates s_ref(N) for the N-aware parent variance scaling in
// search_policy.h (enable_parent_var_scaling).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
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
ABSL_FLAG(int, num_examples, 200, "Number of examples to process.");
ABSL_FLAG(int, skip_first, 0, "Number of examples to skip first.");
ABSL_FLAG(int, visits, 1600, "PUCT visits per position.");
ABSL_FLAG(std::string, csv_path, "", "Optional per-node CSV output.");
ABSL_FLAG(std::string, s_ref_path, "", "Optional s_ref table output.");
ABSL_FLAG(std::string, sib_csv_path, "",
          "Optional per-(parent, visited child) CSV: example,depth,parent_n,"
          "parent_v_var,a,edge_n,child_n,child_v_var,child_q.");
ABSL_FLAG(float, s_ref_default, 0.4f, "Fill value for bins with no data.");
ABSL_FLAG(float, uw_coeff, 0.25f, "Leaf weight coefficient c.");
ABSL_FLAG(float, uw_max_weight, 8.0f, "Leaf weight cap.");

namespace {

using ::data::RecordReaderOptions;
using ::data::SequentialRecordReader;
using ::game::AsLoc;
using ::game::Color;
using ::game::Game;
using ::mcts::GumbelEvaluator;
using ::mcts::MctsNodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::TreeNode;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

static constexpr size_t kCacheSize = 32768;
static constexpr int64_t kTimeoutUs = 400;
static constexpr int kBins = PuctParams::kSRefBins;

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
      CHECK(game::MoveOk(res)) << "Failed to place black stone at index " << i;
    }
  }
  for (int i = 0; i < constants::kNumBoardLocs; ++i) {
    if (pos[i] == WHITE) {
      auto res = board.PlayMove(AsLoc(i), WHITE);
      CHECK(game::MoveOk(res)) << "Failed to place white stone at index " << i;
    }
  }
  return board;
}

// Eval base config (leela.cfg): LCB, p_opt_weight 1, root_fpu 0.1.
static const PuctParams kLivePuctParams =
    PuctParams::Builder()
        .set_root_fpu(0.1f)
        .set_p_opt_weight(1.0)
        .set_kind(PuctRootSelectionPolicy::kLcb)
        .build();

struct NodeRecord {
  int depth;
  int n;
  float v_var;
  float err;
};

void WalkTree(const TreeNode* node, int depth,
              std::unordered_set<const TreeNode*>& seen,
              std::vector<NodeRecord>& out) {
  if (node == nullptr || !seen.insert(node).second) return;
  if (node->n >= 1) {
    out.push_back({depth, node->n, node->v_var, node->init_err_est});
  }
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    const TreeNode* child = node->child(a);
    if (child != nullptr) WalkTree(child, depth + 1, seen, out);
  }
}

// Every (parent, visited child) pair of the tree, for the child variance
// scaling study. child_q is from the parent's perspective.
void WalkSiblings(const TreeNode* node, int depth, int example,
                  std::unordered_set<const TreeNode*>& seen,
                  std::ofstream& out) {
  if (node == nullptr || !seen.insert(node).second) return;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    const TreeNode* child = node->child(a);
    if (child == nullptr) continue;
    if (node->child_visits[a] > 0) {
      out << example << ',' << depth << ',' << node->n << ',' << node->v_var
          << ',' << a << ',' << node->child_visits[a] << ',' << child->n
          << ',' << child->v_var << ',' << -child->v << '\n';
    }
    WalkSiblings(child, depth + 1, example, seen, out);
  }
}

int Bin(int n) {
  int b = 0;
  while ((n >> (b + 1)) > 0) ++b;
  return std::min(b, kBins - 1);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string chunk_path = absl::GetFlag(FLAGS_chunk_path);
  const int num_examples = absl::GetFlag(FLAGS_num_examples);
  const int skip_first = absl::GetFlag(FLAGS_skip_first);
  const int visits = absl::GetFlag(FLAGS_visits);
  CHECK(!model_path.empty()) << "--model_path required";
  CHECK(!chunk_path.empty()) << "--chunk_path required";

  auto engine = nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                                 /*batch_size=*/1,
                                 nn::GetVersionFromModelPath(model_path));
  auto nn_interface = std::make_unique<nn::NNInterface>(
      /*num_threads=*/1, kTimeoutUs, kCacheSize, std::move(engine));
  mcts::BiasCache bias_cache(0.85f, 0.45f);
  GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0,
                            mcts::ScoreUtilityParams{}, &bias_cache);
  const float uw_coeff = absl::GetFlag(FLAGS_uw_coeff);
  const float uw_max_weight = absl::GetFlag(FLAGS_uw_max_weight);
  std::vector<float> errs, root_errs;  // every evaluated node; roots only

  SequentialRecordReader reader(chunk_path, RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok()) << "Failed to open: " << chunk_path;

  std::ofstream csv;
  if (!absl::GetFlag(FLAGS_csv_path).empty()) {
    csv.open(absl::GetFlag(FLAGS_csv_path));
    CHECK(csv.is_open()) << "cannot open " << absl::GetFlag(FLAGS_csv_path);
    csv << "example,depth,n,v_var,err\n";
  }

  std::ofstream sib_csv;
  if (!absl::GetFlag(FLAGS_sib_csv_path).empty()) {
    sib_csv.open(absl::GetFlag(FLAGS_sib_csv_path));
    CHECK(sib_csv.is_open()) << "cannot open "
                             << absl::GetFlag(FLAGS_sib_csv_path);
    sib_csv << "example,depth,parent_n,parent_v_var,a,edge_n,child_n,"
               "child_v_var,child_q\n";
  }

  std::array<std::vector<float>, kBins> bin_s;  // sqrt(v_var) samples per bin
  std::array<double, kBins> bin_var_sum{};

  int processed = 0, seen_records = 0;
  std::string record;
  while (processed < num_examples) {
    auto status = reader.ReadRecord(&record);
    if (absl::IsOutOfRange(status)) break;
    CHECK(status.ok()) << status;
    if (seen_records++ < skip_first) continue;

    Example example;
    if (!example.ParseFromString(record)) continue;
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

    auto node_table = std::make_unique<MctsNodeTable>();
    TreeNode* root =
        node_table->GetOrCreate(game.board().hash(), color_to_move, false);
    core::Probability prob(0xdeadbeef12345678ULL ^
                           static_cast<uint64_t>(seen_records));
    evaluator.SearchRootPuct(prob, game, node_table.get(), root, color_to_move,
                             visits, kLivePuctParams);

    std::vector<NodeRecord> nodes;
    std::unordered_set<const TreeNode*> seen_nodes;
    WalkTree(root, 0, seen_nodes, nodes);
    root_errs.push_back(root->init_err_est);
    if (sib_csv.is_open()) {
      std::unordered_set<const TreeNode*> sib_seen;
      WalkSiblings(root, 0, processed, sib_seen, sib_csv);
    }
    for (const NodeRecord& r : nodes) {
      errs.push_back(r.err);
      if (csv.is_open()) {
        csv << processed << ',' << r.depth << ',' << r.n << ',' << r.v_var
            << ',' << r.err << '\n';
      }
      if (r.n < 3) continue;
      const int b = Bin(r.n);
      bin_s[b].push_back(std::sqrt(std::max(r.v_var, 0.0f)));
      bin_var_sum[b] += r.v_var;
    }
    ++processed;
    if (processed % 25 == 0) {
      LOG(INFO) << "Processed " << processed << "/" << num_examples
                << " nodes so far " << [&] {
                     size_t t = 0;
                     for (const auto& v : bin_s) t += v.size();
                     return t;
                   }();
    }
  }

  printf("\nNode variance dump  (visits %d, examples %d)\n", visits, processed);
  printf("%-4s %-14s %-8s %-10s %-10s %-10s\n", "bin", "N range", "count",
         "mean_s", "median_s", "mean_var");
  std::array<float, kBins> s_ref{};
  float last = absl::GetFlag(FLAGS_s_ref_default);
  for (int b = 0; b < kBins; ++b) {
    auto& v = bin_s[b];
    if (v.empty()) {
      s_ref[b] = last;
      continue;
    }
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (float x : v) sum += x;
    const float mean_s = static_cast<float>(sum / v.size());
    const float median_s = v[v.size() / 2];
    s_ref[b] = mean_s;
    last = mean_s;
    printf("%-4d [%6d,%6d) %-8zu %-10.4f %-10.4f %-10.4f\n", b, 1 << b,
           1 << (b + 1), v.size(), mean_s, median_s, bin_var_sum[b] / v.size());
  }
  printf("\ns_ref:");
  for (int b = 0; b < kBins; ++b) printf(" %.4f", s_ref[b]);
  printf("\n");

  // Error head and implied leaf weights. Each evaluated node is one playout's
  // leaf, so the mean over nodes is the mean weight per playout.
  const auto print_dist = [](const char* name, std::vector<float> v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (float x : v) sum += x;
    const auto q = [&](double p) { return v[std::min(v.size() - 1,
                                                     size_t(p * v.size()))]; };
    printf("%-12s n=%-8zu mean=%.4f p1=%.4f p5=%.4f p25=%.4f p50=%.4f "
           "p75=%.4f p95=%.4f p99=%.4f\n",
           name, v.size(), sum / v.size(), q(0.01), q(0.05), q(0.25), q(0.5),
           q(0.75), q(0.95), q(0.99));
  };
  std::vector<float> weights;
  for (float e : errs) weights.push_back(uw_coeff / (e + uw_coeff / uw_max_weight));
  std::vector<float> err2s;
  for (float e : errs) err2s.push_back(e * e);
  printf("\nError head (err = sqrt(q6_err)), leaf weight c=%.3f max=%.1f\n",
         uw_coeff, uw_max_weight);
  print_dist("err", errs);
  print_dist("err2", err2s);
  print_dist("root_err", root_errs);
  print_dist("weight", weights);

  if (!absl::GetFlag(FLAGS_s_ref_path).empty()) {
    std::ofstream out(absl::GetFlag(FLAGS_s_ref_path));
    CHECK(out.is_open()) << "cannot open " << absl::GetFlag(FLAGS_s_ref_path);
    for (int b = 0; b < kBins; ++b)
      out << s_ref[b] << (b + 1 < kBins ? ' ' : '\n');
  }
  return 0;
}
