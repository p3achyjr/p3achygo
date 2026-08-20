// Experiment: is retrying after a collision a bad idea?
//
// Hypothesis (p3achyjr): when a collision aborts, it means soft virtual loss +
// virtual visits were not enough to steer the second thread off that path --
// i.e. the network strongly prefers it. Forking elsewhere therefore spends the
// visit on a path the search did not want, which is noise at best.
//
// Three predictions, each measured directly at the fork points that
// SmartRetryCollisionPolicy chooses (it forks at the cheapest deviation point
// on the colliding path, so these are lower bounds on the cost of deviating):
//
//   P1  The abandoned move is strongly preferred by the NETWORK.
//       -> p_best (prior on the abandoned move) is high, p_chosen is low.
//   P2  The abandoned move is strongly preferred by the SEARCH.
//       -> the PUCT gap at the fork is large, even though it is the minimum
//          gap available anywhere on the path.
//   P3  The retry explores something the final tree agrees was worse.
//       -> after the search, the forked action's Q and visit count are below
//          the fork node's best child.
//
// Baseline for P1/P2: the same quantities at every node of the search tree,
// weighted by visits, i.e. what a typical descent decision looks like.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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
#include "cc/mcts/node_table.h"
#include "cc/mcts/search.h"
#include "cc/mcts/search_policy.h"
#include "cc/mcts/tree.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"
#include "cc/proto/feature_util.h"
#include "example.pb.h"

ABSL_FLAG(std::string, engine_path,
          "/home/p3achyjr/p3achygo-data/v4/model_cands/_onnx/model_0633.trt",
          "Path to TRT engine.");
ABSL_FLAG(std::string, chunk_path,
          "/home/p3achyjr/p3achygo-data/v4/goldens/chunk_0633.tfrecord.zz",
          "Path to tfrecord chunk (.tfrecord.zz).");
ABSL_FLAG(int, num_positions, 60, "Number of positions to sample.");
ABSL_FLAG(int, visits, 1600, "Visit budget.");
ABSL_FLAG(int, num_threads, 32, "Worker threads.");
ABSL_FLAG(float, vl_delta, -1.0f, "Virtual loss magnitude (negative).");
ABSL_FLAG(float, root_fpu, 0.1f, "Root FPU reduction.");
ABSL_FLAG(int, max_collision_retries, 4, "Retry budget.");
ABSL_FLAG(uint64_t, seed, 42, "Seed for position sampling.");
ABSL_FLAG(std::string, csv_path, "", "If non-empty, write per-fork rows here.");
ABSL_FLAG(std::string, path_csv_path, "",
          "If non-empty, write one row per (fork, path node) here: the PUCT "
          "gap profile along each colliding path.");

namespace {

using ::data::RecordReaderOptions;
using ::data::SequentialRecordReader;
using ::game::AsLoc;
using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::game::MoveOk;
using ::mcts::ForkEvent;
using ::mcts::ForkEventSink;
using ::mcts::MctsNodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::Search;
using ::mcts::TreeNode;
using ::nn::NNInterface;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

static constexpr int kNumActions = constants::kMaxMovesPerPosition;

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

std::vector<std::string> SampleRecords(const std::string& chunk_path, int n,
                                       uint64_t seed) {
  SequentialRecordReader reader(chunk_path, RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok()) << "Failed to open: " << chunk_path;
  std::mt19937_64 rng(seed);
  std::vector<std::string> reservoir;
  reservoir.reserve(n);
  std::string record;
  int64_t seen = 0;
  while (true) {
    auto status = reader.ReadRecord(&record);
    if (absl::IsOutOfRange(status)) break;
    CHECK(status.ok()) << "Read error: " << status;
    if (static_cast<int>(reservoir.size()) < n) {
      reservoir.push_back(record);
    } else {
      const int64_t j = std::uniform_int_distribution<int64_t>(0, seen)(rng);
      if (j < n) reservoir[j] = record;
    }
    ++seen;
  }
  LOG(INFO) << "Sampled " << reservoir.size() << " of " << seen << " records.";
  return reservoir;
}

// One fork, joined against the final tree.
struct ForkRow {
  int position;
  int fork_depth;
  int path_len;
  float p_best;    // prior on the move the search wanted.
  float p_chosen;  // prior on the move the retry took.
  float puct_gap;  // puct_best - puct_chosen at fork time.
  int n_best_at_fork;
  int n_chosen_at_fork;
  // Final-tree verdict at the fork node.
  bool final_valid;
  float final_q_best;    // Q of the abandoned action, end of search.
  float final_q_chosen;  // Q of the action the retry took.
  int final_n_best;
  int final_n_chosen;
  int final_n_total;      // fork node's total child visits.
  int final_rank_chosen;  // visit rank of the chosen action (0 = most visited).
};

// One node of one colliding path.
struct PathRow {
  int position;
  int fork_id;
  int depth;
  int path_len;
  float gap;
  int node_n;
  int n_children;
  bool is_fork;  // this node is the argmin that SmartRetry chose.
};

// Visit-weighted baseline: what an ordinary decision node looks like.
struct NodeBaseline {
  double w_p_top = 0;  // prior on the most-visited child.
  double w_p_second = 0;
  double w_gap_q = 0;  // Q gap between best and second-best child.
  double weight = 0;
  int64_t nodes = 0;
};

// Walks the final tree and accumulates visit-weighted stats over all expanded
// nodes with >= 2 visited children.
void AccumulateBaseline(const TreeNode* node, NodeBaseline* out, int depth) {
  if (node == nullptr || depth > 8) return;
  int a1 = -1, a2 = -1;
  for (int a = 0; a < kNumActions; ++a) {
    if (node->child_visits[a] == 0) continue;
    if (a1 < 0 || node->child_visits[a] > node->child_visits[a1]) {
      a2 = a1;
      a1 = a;
    } else if (a2 < 0 || node->child_visits[a] > node->child_visits[a2]) {
      a2 = a;
    }
  }
  if (a1 >= 0 && a2 >= 0) {
    const TreeNode* c1 = node->child(a1);
    const TreeNode* c2 = node->child(a2);
    if (c1 != nullptr && c2 != nullptr) {
      const double w = node->n;
      out->w_p_top += w * node->move_probs[a1];
      out->w_p_second += w * node->move_probs[a2];
      out->w_gap_q += w * ((-c1->v) - (-c2->v));
      out->weight += w;
      out->nodes += 1;
    }
  }
  for (int a = 0; a < kNumActions; ++a) {
    if (node->child_visits[a] > 0) {
      AccumulateBaseline(node->child(a), out, depth + 1);
    }
  }
}

double Mean(const std::vector<float>& v) {
  if (v.empty()) return 0;
  double s = 0;
  for (float x : v) s += x;
  return s / v.size();
}

double Median(std::vector<float> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double Quantile(std::vector<float> v, float q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min<size_t>(v.size() - 1, static_cast<size_t>(q * v.size()))];
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  const std::string engine_path = absl::GetFlag(FLAGS_engine_path);
  const int num_positions = absl::GetFlag(FLAGS_num_positions);
  const int visits = absl::GetFlag(FLAGS_visits);
  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  const float vl_delta = absl::GetFlag(FLAGS_vl_delta);
  const float root_fpu = absl::GetFlag(FLAGS_root_fpu);
  const int max_retries = absl::GetFlag(FLAGS_max_collision_retries);
  const std::string csv_path = absl::GetFlag(FLAGS_csv_path);
  const std::string path_csv_path = absl::GetFlag(FLAGS_path_csv_path);

  NNInterface nn(
      num_threads, /*timeout=*/0, /*cache_size=*/0,
      nn::CreateEngine(nn::KindFromEnginePath(engine_path), engine_path,
                       num_threads, nn::GetVersionFromModelPath(engine_path)),
      NNInterface::SignalKind::kExplicit,
      /*num_shared_search_tasks=*/1);

  ForkEventSink sink;
  Search::Params params{
      .num_threads = num_threads,
      .total_visit_budget = visits,
      .total_visit_time_ms = 0,
      .puct_params = PuctParams::Builder()
                         .set_kind(PuctRootSelectionPolicy::kVisitCount)
                         .set_root_fpu(root_fpu)
                         .build(),
      .q_fn_kind = mcts::QFnKind::kVirtualLossSoft,
      .n_fn_kind = mcts::NFnKind::kVirtualVisit,
      .descent_policy_kind = mcts::DescentPolicyKind::kDeterministic,
      .collision_policy_kind = mcts::CollisionPolicyKind::kSmartRetry,
      .collision_detector_kind = mcts::CollisionDetectorKind::kNoOp,
      .vl_delta = vl_delta,
      .max_collision_retries = max_retries,
      .mode = Search::Mode::kConcurrent,
      .fork_sink = &sink,
  };

  const std::vector<std::string> records =
      SampleRecords(absl::GetFlag(FLAGS_chunk_path), num_positions,
                    absl::GetFlag(FLAGS_seed));

  std::vector<ForkRow> rows;
  std::vector<PathRow> path_rows;
  NodeBaseline baseline;

  for (size_t idx = 0; idx < records.size(); ++idx) {
    Example example;
    if (!example.ParseFromString(records[idx])) continue;

    const auto board_pos = ParseSequence<Color, constants::kNumBoardLocs>(
        GetFeatureValues<std::string>("board", example).Get(0));
    const Color color_to_move = ParseScalar<Color>(
        GetFeatureValues<std::string>("color", example).Get(0));
    const auto last_move_encodings =
        ParseSequence<int16_t, constants::kNumLastMoves>(
            GetFeatureValues<std::string>("last_moves", example).Get(0));
    const float komi = GetFeatureValues<float>("komi", example).Get(0);

    game::Board board = BuildBoard(board_pos, komi);
    absl::InlinedVector<game::Move, constants::kNumLastMoves> last_moves;
    for (int i = 0; i < constants::kNumLastMoves; ++i) {
      last_moves.push_back(
          game::Move{color_to_move, AsLoc(last_move_encodings[i])});
    }
    Game game(board, last_moves, /*init_mv_num=*/0);

    (void)sink.Drain();  // discard anything left over.
    MctsNodeTable node_table;
    TreeNode* root = node_table.GetOrCreateGuarded(game.board().hash(),
                                                   color_to_move, false);
    core::Probability probability;
    Search search(nn.MakeSlot(0));
    search.Run(probability, game, &node_table, root, color_to_move, params);

    // Join fork events against the finished tree. Must happen before
    // node_table goes out of scope.
    for (const ForkEvent& e : sink.Drain()) {
      ForkRow r{};
      r.position = static_cast<int>(idx);
      r.fork_depth = e.fork_depth;
      r.path_len = e.path_len;
      r.p_best = e.p_best;
      r.p_chosen = e.p_chosen;
      r.puct_gap = e.puct_best - e.puct_chosen;
      r.n_best_at_fork = e.n_best;
      r.n_chosen_at_fork = e.n_chosen;

      const TreeNode* fn = e.node;
      const TreeNode* cb = fn->child(e.action_best);
      const TreeNode* cc = fn->child(e.action_chosen);
      if (cb != nullptr && cc != nullptr) {
        r.final_valid = true;
        r.final_q_best = -cb->v;
        r.final_q_chosen = -cc->v;
        r.final_n_best = fn->child_visits[e.action_best];
        r.final_n_chosen = fn->child_visits[e.action_chosen];
        int total = 0, rank = 0;
        for (int a = 0; a < kNumActions; ++a) {
          total += fn->child_visits[a];
          if (fn->child_visits[a] > r.final_n_chosen) ++rank;
        }
        r.final_n_total = total;
        r.final_rank_chosen = rank;
      }
      for (int d = 0; d < e.path_recorded; ++d) {
        if (e.path_gap[d] < 0) continue;
        path_rows.push_back(PathRow{
            .position = static_cast<int>(idx),
            .fork_id = static_cast<int>(rows.size()),
            .depth = d,
            .path_len = e.path_len,
            .gap = e.path_gap[d],
            .node_n = e.path_node_n[d],
            .n_children = e.path_n_children[d],
            .is_fork = (d == e.fork_depth),
        });
      }
      rows.push_back(r);
    }

    AccumulateBaseline(root, &baseline, 0);
    if ((idx + 1) % 10 == 0) {
      LOG(INFO) << "Processed " << (idx + 1) << "/" << records.size()
                << ", forks so far: " << rows.size();
    }
  }

  if (rows.empty()) {
    LOG(ERROR) << "No fork events recorded.";
    return 1;
  }

  std::vector<float> p_best, p_chosen, gap, q_delta, frac_chosen, depth_frac;
  std::vector<float> ranks;
  int valid = 0, chosen_worse = 0, chosen_unvisited_after = 0;
  for (const ForkRow& r : rows) {
    p_best.push_back(r.p_best);
    p_chosen.push_back(r.p_chosen);
    gap.push_back(r.puct_gap);
    depth_frac.push_back(r.path_len > 0 ? float(r.fork_depth) / r.path_len : 0);
    if (!r.final_valid) continue;
    ++valid;
    q_delta.push_back(r.final_q_chosen - r.final_q_best);
    if (r.final_q_chosen < r.final_q_best) ++chosen_worse;
    frac_chosen.push_back(
        r.final_n_total > 0 ? float(r.final_n_chosen) / r.final_n_total : 0.f);
    ranks.push_back(r.final_rank_chosen);
    if (r.final_n_chosen <= 1) ++chosen_unvisited_after;
  }

  const int n = static_cast<int>(rows.size());
  printf("\n=== Collision-retry hypothesis check ===\n");
  printf("Positions: %d   visits: %d   threads: %d   vl_delta: %.2f\n",
         num_positions, visits, num_threads, vl_delta);
  printf("Fork events recorded: %d  (%.1f per position)\n", n,
         float(n) / num_positions);

  printf("\n--- P1: is the abandoned move preferred by the NETWORK? ---\n");
  printf(
      "prior on abandoned move  : mean %.4f  median %.4f  p10 %.4f  p90 %.4f\n",
      Mean(p_best), Median(p_best), Quantile(p_best, 0.1f),
      Quantile(p_best, 0.9f));
  printf(
      "prior on retried move    : mean %.4f  median %.4f  p10 %.4f  p90 %.4f\n",
      Mean(p_chosen), Median(p_chosen), Quantile(p_chosen, 0.1f),
      Quantile(p_chosen, 0.9f));
  printf("ratio of means           : %.2fx\n",
         Mean(p_chosen) > 0 ? Mean(p_best) / Mean(p_chosen) : 0.0);
  printf("baseline, visit-weighted over %ld tree nodes:\n", baseline.nodes);
  printf("  prior on most-visited child  : %.4f\n",
         baseline.weight > 0 ? baseline.w_p_top / baseline.weight : 0.0);
  printf("  prior on 2nd-most-visited    : %.4f\n",
         baseline.weight > 0 ? baseline.w_p_second / baseline.weight : 0.0);

  printf("\n--- P2: is it preferred by the SEARCH? ---\n");
  printf("PUCT gap at fork (minimum available on the path):\n");
  printf("  mean %.4f  median %.4f  p10 %.4f  p90 %.4f\n", Mean(gap),
         Median(gap), Quantile(gap, 0.1f), Quantile(gap, 0.9f));
  printf("fork depth / path length : mean %.3f  median %.3f\n",
         Mean(depth_frac), Median(depth_frac));
  printf("baseline Q gap between best and 2nd-best child: %.4f\n",
         baseline.weight > 0 ? baseline.w_gap_q / baseline.weight : 0.0);

  printf("\n--- P3: was the retried move worse in the final tree? ---\n");
  printf("Forks joinable to final tree: %d / %d\n", valid, n);
  printf("Retried move's final Q minus abandoned move's final Q:\n");
  printf("  mean %+.4f  median %+.4f  p10 %+.4f  p90 %+.4f\n", Mean(q_delta),
         Median(q_delta), Quantile(q_delta, 0.1f), Quantile(q_delta, 0.9f));
  printf("Retried move ended up worse:      %d / %d  (%.1f%%)\n", chosen_worse,
         valid, valid ? 100.f * chosen_worse / valid : 0.f);
  printf("Retried move's final visit share: mean %.4f  median %.4f\n",
         Mean(frac_chosen), Median(frac_chosen));
  printf("Retried move's final visit rank:  mean %.2f  median %.0f\n",
         Mean(ranks), Median(ranks));
  printf("Retried move left with <=1 visit: %d / %d  (%.1f%%)\n",
         chosen_unvisited_after, valid,
         valid ? 100.f * chosen_unvisited_after / valid : 0.f);

  printf(
      "\n--- Why does the argmin land at the root? "
      "PUCT gap by depth over all colliding paths ---\n");
  printf("%6s %8s %10s %10s %10s %10s %10s\n", "depth", "nodes", "med_gap",
         "mean_gap", "med_node_n", "med_kids", "pct_argmin");
  for (int d = 0; d < mcts::kMaxForkPathRecord; ++d) {
    std::vector<float> g, nn, kids;
    int argmin = 0;
    for (const PathRow& pr : path_rows) {
      if (pr.depth != d) continue;
      g.push_back(pr.gap);
      nn.push_back(pr.node_n);
      kids.push_back(pr.n_children);
      argmin += pr.is_fork;
    }
    if (g.empty()) continue;
    printf("%6d %8zu %10.4f %10.4f %10.0f %10.0f %9.1f%%\n", d, g.size(),
           Median(g), Mean(g), Median(nn), Median(kids),
           100.f * argmin / g.size());
  }

  if (!path_csv_path.empty()) {
    FILE* f = fopen(path_csv_path.c_str(), "w");
    CHECK(f != nullptr) << "Could not open " << path_csv_path;
    fprintf(f,
            "position,fork_id,depth,path_len,gap,node_n,n_children,is_fork\n");
    for (const PathRow& pr : path_rows) {
      fprintf(f, "%d,%d,%d,%d,%.6f,%d,%d,%d\n", pr.position, pr.fork_id,
              pr.depth, pr.path_len, pr.gap, pr.node_n, pr.n_children,
              pr.is_fork ? 1 : 0);
    }
    fclose(f);
    LOG(INFO) << "Wrote " << path_csv_path;
  }

  if (!csv_path.empty()) {
    FILE* f = fopen(csv_path.c_str(), "w");
    CHECK(f != nullptr) << "Could not open " << csv_path;
    fprintf(f,
            "position,fork_depth,path_len,p_best,p_chosen,puct_gap,"
            "n_best_at_fork,n_chosen_at_fork,final_valid,final_q_best,"
            "final_q_chosen,final_n_best,final_n_chosen,final_n_total,"
            "final_rank_chosen\n");
    for (const ForkRow& r : rows) {
      fprintf(f, "%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%d,%.6f,%.6f,%d,%d,%d,%d\n",
              r.position, r.fork_depth, r.path_len, r.p_best, r.p_chosen,
              r.puct_gap, r.n_best_at_fork, r.n_chosen_at_fork,
              r.final_valid ? 1 : 0, r.final_q_best, r.final_q_chosen,
              r.final_n_best, r.final_n_chosen, r.final_n_total,
              r.final_rank_chosen);
    }
    fclose(f);
    LOG(INFO) << "Wrote " << csv_path;
  }
  return 0;
}
