// Experiment: parallel search (soft virtual loss + virtual visits) vs. serial.
//
// Loads positions from a training chunk and, for each position, runs several
// searches at the same visit budget through the same `mcts::Search::Run` code
// path. Arm 0 is the serial reference; every other arm is compared against it.
//
//   serial       num_threads=1,  q=identity,          n=identity
//   control      same as serial, different RNG seed
//   par_abort    num_threads=N,  q=virtual_loss_soft, n=virtual_visit,
//                collision_policy=abort
//   par_sretry   same as par_abort but collision_policy=smart_retry
//
// The control arm's divergence from serial is the noise floor: TV distance and
// disagreement below that level are not attributable to parallelism.
//
// Both thread counts load the same TRT engine file (the pipeline's engines are
// built with dynamic batch shapes), so there is no engine-numerics confound.
//
// Reports, per position: the top-5 root children of each arm (visits, visit
// fraction, winrate, score, utility), plus root winrate/score and the
// abort/collision counts. Summary: per-arm top-1 agreement, top-5 overlap,
// total-variation distance between root visit distributions, and root
// winrate/score deltas, all relative to the serial arm.

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
          "Path to TRT engine. Loaded once per thread count; dynamic batch "
          "shapes let the same file serve both the 1- and N-thread arms.");
ABSL_FLAG(std::string, chunk_path,
          "/home/p3achyjr/p3achygo-data/v4/goldens/chunk_0633.tfrecord.zz",
          "Path to tfrecord chunk (.tfrecord.zz).");
ABSL_FLAG(int, num_positions, 150, "Number of positions to sample.");
ABSL_FLAG(int, visits, 1600, "Visit budget for every arm.");
ABSL_FLAG(int, num_threads, 32, "Worker threads for the parallel arms.");
ABSL_FLAG(float, vl_delta, -1.0f, "Virtual loss magnitude (negative).");
ABSL_FLAG(float, root_fpu, 0.1f, "Root FPU reduction, all arms.");
ABSL_FLAG(int, max_collision_retries, 4,
          "Retry budget for the smart_retry arm.");
ABSL_FLAG(uint64_t, seed, 42, "Seed for position sampling and search RNG.");
ABSL_FLAG(bool, verbose, true, "If true, print per-position details.");
ABSL_FLAG(std::string, csv_path, "",
          "If non-empty, write per-(position, arm) scalars here.");
ABSL_FLAG(bool, run_control, true,
          "Run the noise-floor arm (serial config, different seed).");
ABSL_FLAG(bool, run_abort, true, "Run the abort-collision parallel arm.");
ABSL_FLAG(bool, run_smart_retry, true,
          "Run the smart_retry-collision parallel arm.");
ABSL_FLAG(bool, run_sretry_noroot, true,
          "Run the smart_retry-but-never-fork-at-root parallel arm.");
ABSL_FLAG(bool, run_sampled, true,
          "Run the preemptive-forking (sampled descent) parallel arm.");
ABSL_FLAG(float, descent_temperature, 0.05f,
          "Softmax temperature for the sampled-descent arm.");
ABSL_FLAG(bool, run_oracle, true,
          "Run a high-visit serial oracle and score every arm against it.");
ABSL_FLAG(int, oracle_visits, 10000, "Visit budget for the serial oracle.");

namespace {

using ::data::RecordReaderOptions;
using ::data::SequentialRecordReader;
using ::game::AsLoc;
using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::game::MoveOk;
using ::mcts::MctsNodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::Search;
using ::mcts::TreeNode;
using ::nn::NNInterface;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

static constexpr int kTopK = 5;
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

std::string LocToString(Loc loc) {
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
  float winrate;  // from the root player's perspective, in [0, 1].
  float score;    // from the root player's perspective, in points.
  float util;     // -child->v, the value PUCT actually maximizes.
};

struct SearchResult {
  Loc lcb_move = game::kNoopLoc;  // what Search::Run returns.
  Loc top_visit_move = game::kNoopLoc;
  std::vector<ChildInfo> children;  // sorted descending by visits.
  std::array<int, kNumActions> child_visits{};
  int total_child_visits = 0;
  int root_n = 0;
  float root_winrate = 0.f;
  float root_score = 0.f;
  size_t num_aborted = 0;
  size_t num_collisions = 0;
  size_t time_ms = 0;
};

// One search configuration. Arm 0 is the reference all others are scored
// against.
struct Arm {
  std::string name;
  NNInterface* nn;
  Search::Params params;
  // XORed into the per-position seed, so the control arm can share the serial
  // arm's params but draw different randomness.
  uint64_t seed_mask;
};

SearchResult RunSearch(const Arm& arm, Game& game, Color color_to_move,
                       uint64_t pos_seed) {
  MctsNodeTable node_table;
  TreeNode* root =
      node_table.GetOrCreateGuarded(game.board().hash(), color_to_move, false);
  core::Probability probability(pos_seed ^ arm.seed_mask);

  // A fresh Search per run: GlobalSearchState's visit/abort counters are not
  // cleared by Run(), only by the constructor's default init.
  Search search(arm.nn->MakeSlot(0));
  Search::Result result = search.Run(probability, game, &node_table, root,
                                     color_to_move, arm.params);

  SearchResult sr;
  sr.lcb_move = result.move;
  sr.num_aborted = result.num_aborted;
  sr.num_collisions = result.num_collisions;
  sr.time_ms = result.time_ms;
  sr.root_n = root->n;
  sr.root_winrate = (root->v_outcome + 1.f) / 2.f;
  sr.root_score = root->score;

  for (int a = 0; a < kNumActions; ++a) {
    const int visits = root->child_visits[a];
    sr.child_visits[a] = visits;
    sr.total_child_visits += visits;
    if (visits == 0) continue;
    const TreeNode* child = root->child(a);
    if (child == nullptr) continue;
    sr.children.push_back(
        {a, visits, (-child->v_outcome + 1.f) / 2.f, -child->score, -child->v});
  }
  std::sort(sr.children.begin(), sr.children.end(),
            [](const ChildInfo& a, const ChildInfo& b) {
              return a.visits > b.visits;
            });
  if (!sr.children.empty()) {
    sr.top_visit_move = AsLoc(sr.children[0].action);
  }
  return sr;
}

// 0.5 * sum_a |p_a - q_a| over the two root visit distributions. Equivalently,
// the fraction of visit mass one would relocate to turn a into b.
float TotalVariation(const SearchResult& a, const SearchResult& b) {
  if (a.total_child_visits == 0 || b.total_child_visits == 0) return 0.f;
  float tv = 0.f;
  for (int i = 0; i < kNumActions; ++i) {
    tv +=
        std::abs(static_cast<float>(a.child_visits[i]) / a.total_child_visits -
                 static_cast<float>(b.child_visits[i]) / b.total_child_visits);
  }
  return 0.5f * tv;
}

// The arm's stats for `action`, or nullptr if it never visited it.
const ChildInfo* ChildFor(const SearchResult& sr, int action) {
  for (const ChildInfo& c : sr.children) {
    if (c.action == action) return &c;
  }
  return nullptr;
}

// Number of actions present in both searches' top-k lists.
int TopKOverlap(const SearchResult& a, const SearchResult& b, int k) {
  int overlap = 0;
  const int na = std::min<int>(k, a.children.size());
  const int nb = std::min<int>(k, b.children.size());
  for (int i = 0; i < na; ++i) {
    for (int j = 0; j < nb; ++j) {
      if (a.children[i].action == b.children[j].action) {
        ++overlap;
        break;
      }
    }
  }
  return overlap;
}

void PrintTopK(const SearchResult& sr, int k) {
  const int shown = std::min<int>(k, sr.children.size());
  for (int i = 0; i < shown; ++i) {
    const ChildInfo& c = sr.children[i];
    const float frac =
        sr.total_child_visits > 0
            ? static_cast<float>(c.visits) / sr.total_child_visits
            : 0.f;
    printf(
        "    [%d] %-5s visits=%5d (%.3f)  wr=%.4f  score=%+7.2f  util=%+.4f\n",
        i + 1, LocToString(AsLoc(c.action)).c_str(), c.visits, frac, c.winrate,
        c.score, c.util);
  }
}

// One arm's divergence from the serial reference on one position.
struct Cmp {
  float tv;
  int top5_overlap;
  bool top1_agree;
  bool lcb_agree;
  float d_winrate;
  float d_score;
  // Absolute properties of the arm itself, not of the comparison.
  int root_n;
  size_t aborted;
  size_t collisions;
  size_t time_ms;
};

// How much value the oracle assigns to an arm's chosen move, relative to the
// oracle's own best move. Deltas are <= 0 by construction; 0 means the arm
// picked what the oracle picked.
struct Regret {
  bool valid;        // oracle visited both its own best and the arm's choice.
  float visit_frac;  // oracle visit share of the arm's chosen move.
  float d_util;
  float d_winrate;
  float d_score;
  int choice_action;
};

Regret RegretVsOracle(const SearchResult& oracle, Loc choice) {
  Regret r{};
  r.choice_action = game::AsIndex(choice, BOARD_LEN);
  if (oracle.children.empty()) return r;
  const ChildInfo& best = oracle.children[0];  // sorted by visits.
  const ChildInfo* got = ChildFor(oracle, r.choice_action);
  if (got == nullptr) return r;  // oracle never visited the arm's choice.
  r.valid = true;
  r.visit_frac =
      oracle.total_child_visits > 0
          ? static_cast<float>(got->visits) / oracle.total_child_visits
          : 0.f;
  r.d_util = got->util - best.util;
  r.d_winrate = got->winrate - best.winrate;
  r.d_score = got->score - best.score;
  return r;
}

Cmp Compare(const SearchResult& ref, const SearchResult& x) {
  return Cmp{
      .tv = TotalVariation(ref, x),
      .top5_overlap = TopKOverlap(ref, x, kTopK),
      .top1_agree = ref.top_visit_move == x.top_visit_move,
      .lcb_agree = ref.lcb_move == x.lcb_move,
      .d_winrate = x.root_winrate - ref.root_winrate,
      .d_score = x.root_score - ref.root_score,
      .root_n = x.root_n,
      .aborted = x.num_aborted,
      .collisions = x.num_collisions,
      .time_ms = x.time_ms,
  };
}

template <typename Proj>
float Mean(const std::vector<Cmp>& v, Proj proj) {
  if (v.empty()) return 0.f;
  double sum = 0;
  for (const auto& c : v) sum += proj(c);
  return static_cast<float>(sum / v.size());
}

template <typename Proj>
float Quantile(const std::vector<Cmp>& v, Proj proj, float q) {
  if (v.empty()) return 0.f;
  std::vector<float> xs;
  xs.reserve(v.size());
  for (const auto& c : v) xs.push_back(proj(c));
  std::sort(xs.begin(), xs.end());
  return xs[std::min<size_t>(xs.size() - 1,
                             static_cast<size_t>(q * xs.size()))];
}

template <typename Proj>
float MeanValid(const std::vector<Regret>& v, Proj proj) {
  double sum = 0;
  int k = 0;
  for (const auto& r : v) {
    if (!r.valid) continue;
    sum += proj(r);
    ++k;
  }
  return k ? static_cast<float>(sum / k) : 0.f;
}

void PrintArmSummary(const std::string& name, const std::vector<Cmp>& v) {
  const int n = static_cast<int>(v.size());
  if (n == 0) return;
  int top1 = 0, lcb = 0;
  for (const auto& c : v) {
    top1 += c.top1_agree;
    lcb += c.lcb_agree;
  }
  printf("\n--- %s vs. serial ---\n", name.c_str());
  printf("Top-1 (visits) agreement:  %d / %d  (%.1f%%)\n", top1, n,
         100.f * top1 / n);
  printf("LCB move agreement:        %d / %d  (%.1f%%)\n", lcb, n,
         100.f * lcb / n);
  printf("Top-5 overlap:             mean %.2f / 5   median %.0f / 5\n",
         Mean(v, [](const Cmp& c) { return c.top5_overlap; }),
         Quantile(v, [](const Cmp& c) { return c.top5_overlap; }, 0.5f));
  printf(
      "Visit-dist TV distance:    mean %.4f  median %.4f  p90 %.4f  max %.4f\n",
      Mean(v, [](const Cmp& c) { return c.tv; }),
      Quantile(
          v, [](const Cmp& c) { return c.tv; }, 0.5f),
      Quantile(
          v, [](const Cmp& c) { return c.tv; }, 0.9f),
      Quantile(v, [](const Cmp& c) { return c.tv; }, 1.0f));
  printf("Root winrate delta:        mean %+.4f  mean |.| %.4f\n",
         Mean(v, [](const Cmp& c) { return c.d_winrate; }),
         Mean(v, [](const Cmp& c) { return std::abs(c.d_winrate); }));
  printf("Root score delta:          mean %+.3f  mean |.| %.3f\n",
         Mean(v, [](const Cmp& c) { return c.d_score; }),
         Mean(v, [](const Cmp& c) { return std::abs(c.d_score); }));
  printf("Realized root visits:      mean %.1f\n",
         Mean(v, [](const Cmp& c) { return c.root_n; }));
  printf("Aborts / collisions:       mean %.1f / %.1f\n",
         Mean(v, [](const Cmp& c) { return c.aborted; }),
         Mean(v, [](const Cmp& c) { return c.collisions; }));
  printf("Wall time per position:    mean %.0fms\n",
         Mean(v, [](const Cmp& c) { return c.time_ms; }));
}

// Regret of an arm's chosen move under the oracle's evaluation.
void PrintRegretSummary(const std::string& name, const std::vector<Regret>& v,
                        const std::vector<Cmp>& vs_oracle) {
  const int n = static_cast<int>(v.size());
  if (n == 0) return;
  int valid = 0, exact = 0;
  for (int i = 0; i < n; ++i) {
    valid += v[i].valid;
    if (v[i].valid && v[i].d_util == 0.f) ++exact;
  }
  int top1 = 0;
  for (const auto& c : vs_oracle) top1 += c.top1_agree;
  printf("\n--- %s vs. ORACLE ---\n", name.c_str());
  printf("Picked oracle's top move:  %d / %d  (%.1f%%)\n", top1, n,
         100.f * top1 / n);
  printf("Top-5 overlap w/ oracle:   mean %.2f / 5\n",
         Mean(vs_oracle, [](const Cmp& c) { return c.top5_overlap; }));
  printf("Visit-dist TV vs oracle:   mean %.4f  median %.4f\n",
         Mean(vs_oracle, [](const Cmp& c) { return c.tv; }),
         Quantile(vs_oracle, [](const Cmp& c) { return c.tv; }, 0.5f));
  printf("Choice unvisited by oracle: %d / %d\n", n - valid, n);
  printf("Regret of chosen move (oracle's valuation, <= 0):\n");
  printf("  utility:  mean %+.4f\n",
         MeanValid(v, [](const Regret& r) { return r.d_util; }));
  printf("  winrate:  mean %+.4f\n",
         MeanValid(v, [](const Regret& r) { return r.d_winrate; }));
  printf("  score:    mean %+.3f pts\n",
         MeanValid(v, [](const Regret& r) { return r.d_score; }));
  printf("  oracle visit share of chosen move: mean %.3f\n",
         MeanValid(v, [](const Regret& r) { return r.visit_frac; }));
}

Search::Params MakeParams(int num_threads, int visits, float root_fpu,
                          mcts::QFnKind q_fn, mcts::NFnKind n_fn,
                          float vl_delta,
                          mcts::CollisionPolicyKind collision_policy,
                          int max_collision_retries,
                          mcts::DescentPolicyKind descent_policy =
                              mcts::DescentPolicyKind::kDeterministic,
                          float descent_temperature = 0.0f) {
  return Search::Params{
      .num_threads = num_threads,
      .total_visit_budget = visits,
      .total_visit_time_ms = 0,
      .puct_params = PuctParams::Builder()
                         .set_kind(PuctRootSelectionPolicy::kVisitCount)
                         .set_root_fpu(root_fpu)
                         .build(),
      .q_fn_kind = q_fn,
      .n_fn_kind = n_fn,
      .descent_policy_kind = descent_policy,
      .collision_policy_kind = collision_policy,
      .collision_detector_kind = mcts::CollisionDetectorKind::kNoOp,
      .vl_delta = vl_delta,
      .max_collision_retries = max_collision_retries,
      .descent_temperature = descent_temperature,
      .mode = Search::Mode::kConcurrent,
  };
}

std::unique_ptr<nn::Engine> MakeEngine(const std::string& path,
                                       int batch_size) {
  CHECK(!path.empty()) << "Empty engine path.";
  return nn::CreateEngine(nn::KindFromEnginePath(path), path, batch_size,
                          nn::GetVersionFromModelPath(path));
}

// Reservoir-samples `n` records from the chunk so the sample spans the whole
// file rather than just its prefix.
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
  LOG(INFO) << "Sampled " << reservoir.size() << " of " << seen
            << " records from " << chunk_path;
  return reservoir;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  const std::string engine_path = absl::GetFlag(FLAGS_engine_path);
  const std::string chunk_path = absl::GetFlag(FLAGS_chunk_path);
  const int num_positions = absl::GetFlag(FLAGS_num_positions);
  const int visits = absl::GetFlag(FLAGS_visits);
  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  const float vl_delta = absl::GetFlag(FLAGS_vl_delta);
  const float root_fpu = absl::GetFlag(FLAGS_root_fpu);
  const int max_retries = absl::GetFlag(FLAGS_max_collision_retries);
  const uint64_t seed = absl::GetFlag(FLAGS_seed);
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  const std::string csv_path = absl::GetFlag(FLAGS_csv_path);
  const bool run_oracle = absl::GetFlag(FLAGS_run_oracle);
  const int oracle_visits = absl::GetFlag(FLAGS_oracle_visits);

  NNInterface nn_serial(/*num_threads=*/1, /*timeout=*/0, /*cache_size=*/0,
                        MakeEngine(engine_path, 1),
                        NNInterface::SignalKind::kExplicit,
                        /*num_shared_search_tasks=*/1);
  NNInterface nn_parallel(num_threads, /*timeout=*/0, /*cache_size=*/0,
                          MakeEngine(engine_path, num_threads),
                          NNInterface::SignalKind::kExplicit,
                          /*num_shared_search_tasks=*/1);

  const Search::Params serial_params = MakeParams(
      1, visits, root_fpu, mcts::QFnKind::kIdentity, mcts::NFnKind::kIdentity,
      vl_delta, mcts::CollisionPolicyKind::kAbort, max_retries);
  const auto parallel_params = [&](mcts::CollisionPolicyKind cp,
                                   mcts::DescentPolicyKind dp =
                                       mcts::DescentPolicyKind::kDeterministic,
                                   float temp = 0.0f) {
    return MakeParams(
        num_threads, visits, root_fpu, mcts::QFnKind::kVirtualLossSoft,
        mcts::NFnKind::kVirtualVisit, vl_delta, cp, max_retries, dp, temp);
  };
  const float descent_temp = absl::GetFlag(FLAGS_descent_temperature);

  // Arm 0 is the reference.
  std::vector<Arm> arms;
  arms.push_back({"serial", &nn_serial, serial_params, 0});
  if (absl::GetFlag(FLAGS_run_control)) {
    arms.push_back({"control", &nn_serial, serial_params, ~0ULL});
  }
  if (absl::GetFlag(FLAGS_run_abort)) {
    arms.push_back({"par_abort", &nn_parallel,
                    parallel_params(mcts::CollisionPolicyKind::kAbort), 0});
  }
  if (absl::GetFlag(FLAGS_run_smart_retry)) {
    arms.push_back({"par_sretry", &nn_parallel,
                    parallel_params(mcts::CollisionPolicyKind::kSmartRetry),
                    0});
  }
  if (absl::GetFlag(FLAGS_run_sretry_noroot)) {
    arms.push_back(
        {"par_sr_noroot", &nn_parallel,
         parallel_params(mcts::CollisionPolicyKind::kSmartRetryNoRoot), 0});
  }
  if (absl::GetFlag(FLAGS_run_sampled)) {
    arms.push_back(
        {"par_sampled", &nn_parallel,
         parallel_params(mcts::CollisionPolicyKind::kAbort,
                         mcts::DescentPolicyKind::kSampled, descent_temp),
         0});
  }
  // Oracle goes last so arm indices of the others are unaffected.
  const int oracle_idx = run_oracle ? static_cast<int>(arms.size()) : -1;
  if (run_oracle) {
    arms.push_back(
        {"oracle", &nn_serial,
         MakeParams(1, oracle_visits, root_fpu, mcts::QFnKind::kIdentity,
                    mcts::NFnKind::kIdentity, vl_delta,
                    mcts::CollisionPolicyKind::kAbort, max_retries),
         0});
  }
  CHECK(arms.size() > 1) << "Nothing to compare against the serial arm.";

  const std::vector<std::string> records =
      SampleRecords(chunk_path, num_positions, seed);

  // cmps[arm][position], for arm >= 1.
  std::vector<std::vector<Cmp>> cmps(arms.size());
  // Same, but scored against the oracle instead of the serial arm.
  std::vector<std::vector<Cmp>> ocmps(arms.size());
  std::vector<std::vector<Regret>> regrets(arms.size());
  // Top move each arm chose, per position; used for the disagreement subset.
  std::vector<std::vector<Loc>> choices(arms.size());
  std::vector<int> position_index;
  std::vector<float> serial_ms;
  std::vector<int> serial_root_n;

  for (size_t idx = 0; idx < records.size(); ++idx) {
    Example example;
    if (!example.ParseFromString(records[idx])) {
      LOG(WARNING) << "Failed to parse example " << idx << ", skipping.";
      continue;
    }

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

    const uint64_t pos_seed = seed ^ (0x9e3779b97f4a7c15ULL * (idx + 1));

    std::vector<SearchResult> results;
    results.reserve(arms.size());
    for (const Arm& arm : arms) {
      results.push_back(RunSearch(arm, game, color_to_move, pos_seed));
    }
    for (size_t a = 1; a < arms.size(); ++a) {
      cmps[a].push_back(Compare(results[0], results[a]));
    }
    if (oracle_idx >= 0) {
      for (size_t a = 0; a < arms.size(); ++a) {
        if (static_cast<int>(a) == oracle_idx) continue;
        ocmps[a].push_back(Compare(results[oracle_idx], results[a]));
        regrets[a].push_back(
            RegretVsOracle(results[oracle_idx], results[a].top_visit_move));
      }
    }
    for (size_t a = 0; a < arms.size(); ++a) {
      choices[a].push_back(results[a].top_visit_move);
    }
    position_index.push_back(static_cast<int>(idx));
    serial_ms.push_back(results[0].time_ms);
    serial_root_n.push_back(results[0].root_n);

    if (verbose) {
      printf("\n=== Position %zu (to move: %s, komi %.1f) ===\n", idx,
             color_to_move == BLACK ? "black" : "white", komi);
      printf("%s\n", game::ToString(game.board().position()).c_str());
      for (size_t a = 0; a < arms.size(); ++a) {
        const SearchResult& r = results[a];
        printf(
            "  [%-10s n=%d root_wr=%.4f root_score=%+.2f lcb=%s %zums "
            "aborted=%zu collisions=%zu]\n",
            arms[a].name.c_str(), r.root_n, r.root_winrate, r.root_score,
            LocToString(r.lcb_move).c_str(), r.time_ms, r.num_aborted,
            r.num_collisions);
        PrintTopK(r, kTopK);
      }
      for (size_t a = 1; a < arms.size(); ++a) {
        const Cmp& c = cmps[a].back();
        printf(
            "  %-10s vs serial: top1_agree=%-3s top5_overlap=%d/5 tv=%.4f "
            "dwr=%+.4f dscore=%+.2f\n",
            arms[a].name.c_str(), c.top1_agree ? "YES" : "NO", c.top5_overlap,
            c.tv, c.d_winrate, c.d_score);
      }
    }

    if ((idx + 1) % 10 == 0) {
      LOG(INFO) << "Processed " << (idx + 1) << "/" << records.size();
    }
  }

  const int n = static_cast<int>(position_index.size());
  if (n == 0) {
    LOG(ERROR) << "No positions processed.";
    return 1;
  }

  double sms = 0, srn = 0;
  for (int i = 0; i < n; ++i) {
    sms += serial_ms[i];
    srn += serial_root_n[i];
  }

  printf("\n=== Summary ===\n");
  printf("Positions:            %d\n", n);
  printf("Visit budget:         %d\n", visits);
  printf("Engine:               %s\n", engine_path.c_str());
  printf("serial:               threads=1,  q=identity,          n=identity\n");
  printf("control:              serial config, different seed (noise floor)\n");
  printf(
      "par_abort:            threads=%d, q=virtual_loss_soft, n=virtual_visit,"
      " vl_delta=%.2f, collision=abort\n",
      num_threads, vl_delta);
  printf(
      "par_sretry:           as par_abort but collision=smart_retry,"
      " max_retries=%d\n",
      max_retries);
  printf(
      "par_sr_noroot:        as par_sretry but aborts when the cheapest fork"
      " point is the root\n");
  printf(
      "par_sampled:          as par_abort but descent=sampled softmax,"
      " temperature=%.3f\n",
      descent_temp);
  printf("All arms:             root_fpu=%.2f, noop collision detector\n",
         root_fpu);
  printf("serial arm:           realized root visits %.1f, %.0fms/position\n",
         srn / n, sms / n);

  for (size_t a = 1; a < arms.size(); ++a) {
    if (static_cast<int>(a) == oracle_idx) continue;
    PrintArmSummary(arms[a].name, cmps[a]);
  }

  if (oracle_idx >= 0) {
    printf("\n===== Scored against the %d-visit serial oracle =====\n",
           oracle_visits);
    for (size_t a = 0; a < arms.size(); ++a) {
      if (static_cast<int>(a) == oracle_idx) continue;
      PrintRegretSummary(arms[a].name, regrets[a], ocmps[a]);
    }

    // Where the two parallel arms actually differ: restrict to positions where
    // they picked different top moves, and compare their regret there.
    int ai = -1, si = -1;
    for (size_t a = 0; a < arms.size(); ++a) {
      if (arms[a].name == "par_abort") ai = static_cast<int>(a);
      if (arms[a].name == "par_sretry") si = static_cast<int>(a);
    }
    if (ai >= 0 && si >= 0) {
      std::vector<Regret> ra, rs;
      int abort_better = 0, sretry_better = 0, tied = 0;
      for (int i = 0; i < n; ++i) {
        if (choices[ai][i] == choices[si][i]) continue;
        if (!regrets[ai][i].valid || !regrets[si][i].valid) continue;
        ra.push_back(regrets[ai][i]);
        rs.push_back(regrets[si][i]);
        const float da = regrets[ai][i].d_util, ds = regrets[si][i].d_util;
        if (da > ds)
          ++abort_better;
        else if (ds > da)
          ++sretry_better;
        else
          ++tied;
      }
      printf(
          "\n--- Disagreement subset: par_abort vs par_sretry picked "
          "different moves ---\n");
      printf("Positions:                 %zu / %d\n", ra.size(), n);
      if (!ra.empty()) {
        printf("par_abort  regret: util %+.4f  winrate %+.4f  score %+.3f\n",
               MeanValid(ra, [](const Regret& r) { return r.d_util; }),
               MeanValid(ra, [](const Regret& r) { return r.d_winrate; }),
               MeanValid(ra, [](const Regret& r) { return r.d_score; }));
        printf("par_sretry regret: util %+.4f  winrate %+.4f  score %+.3f\n",
               MeanValid(rs, [](const Regret& r) { return r.d_util; }),
               MeanValid(rs, [](const Regret& r) { return r.d_winrate; }),
               MeanValid(rs, [](const Regret& r) { return r.d_score; }));
        printf(
            "Head-to-head on utility: par_abort better %d, par_sretry "
            "better %d, tied %d\n",
            abort_better, sretry_better, tied);
      }
    }
  }

  // Worst positions by TV, per parallel arm.
  for (size_t a = 1; a < arms.size(); ++a) {
    if (static_cast<int>(a) == oracle_idx) continue;
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int x, int y) { return cmps[a][x].tv > cmps[a][y].tv; });
    printf("\nWorst 10 positions by TV, %s:\n", arms[a].name.c_str());
    for (int i = 0; i < std::min(10, n); ++i) {
      const Cmp& c = cmps[a][order[i]];
      printf(
          "  pos=%-4d tv=%.4f  top5_overlap=%d/5  top1_agree=%s  dwr=%+.4f  "
          "dscore=%+.2f  aborts=%zu\n",
          position_index[order[i]], c.tv, c.top5_overlap,
          c.top1_agree ? "Y" : "N", c.d_winrate, c.d_score, c.aborted);
    }
  }

  if (!csv_path.empty()) {
    FILE* f = fopen(csv_path.c_str(), "w");
    CHECK(f != nullptr) << "Could not open " << csv_path;
    fprintf(f,
            "position,arm,tv,top5_overlap,top1_agree,lcb_agree,d_winrate,"
            "d_score,root_n,aborted,collisions,time_ms,serial_root_n,"
            "serial_ms,otv,otop5_overlap,otop1_agree,regret_valid,"
            "regret_util,regret_winrate,regret_score,oracle_visit_frac\n");
    for (size_t a = 1; a < arms.size(); ++a) {
      if (static_cast<int>(a) == oracle_idx) continue;
      for (int i = 0; i < n; ++i) {
        const Cmp& c = cmps[a][i];
        const bool has_o = oracle_idx >= 0;
        const Cmp oc = has_o ? ocmps[a][i] : Cmp{};
        const Regret r = has_o ? regrets[a][i] : Regret{};
        fprintf(f,
                "%d,%s,%.6f,%d,%d,%d,%.6f,%.6f,%d,%zu,%zu,%zu,%d,%.0f,"
                "%.6f,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                position_index[i], arms[a].name.c_str(), c.tv, c.top5_overlap,
                c.top1_agree ? 1 : 0, c.lcb_agree ? 1 : 0, c.d_winrate,
                c.d_score, c.root_n, c.aborted, c.collisions, c.time_ms,
                serial_root_n[i], serial_ms[i], oc.tv, oc.top5_overlap,
                oc.top1_agree ? 1 : 0, r.valid ? 1 : 0, r.d_util, r.d_winrate,
                r.d_score, r.visit_frac);
      }
    }
    fclose(f);
    LOG(INFO) << "Wrote " << csv_path;
  }

  return 0;
}
