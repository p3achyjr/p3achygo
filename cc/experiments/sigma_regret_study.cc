// Experiment: sigma regret study.
//
// Compares the production Gumbel sigma (Q transform) against the SE-scaled
// sigma, using a sigma-independent ground truth:
//
//   GT: Gumbel(N=10000, K=16) with the production sigma. q_GT(a) is the root
//       child Q for GT-visited children and the GT v_mix for the rest.
//       q* = max q_GT(a) over children with at least 1% of GT visits.
//
// For every arm and every N in kNValues, a fresh tree is seeded with
// --seed_visits PUCT visits (simulating tree reuse) and then searched with
// Gumbel(N, K=16) under the arm's sigma. The same per-position seed drives the
// Gumbel noise in GT and all arms, so the top-K action set is shared.
//
// Metrics per (arm, N):
//   regret_pi   = q* - sum_a pi_n(a) q_GT(a)      (policy-improvement
//   objective) regret_move = q* - q_GT(mcts_move_n)          (move actually
//   played) top1        = mcts_move_n == argmax q_GT       (eligible children
//   only) kl_prodgt   = KL(pi_GT_prod || pi_n), pi_GT_prod =
//   ComputeImprovedPolicy(gt, 0)
//
// Arms: prod | se_t_m0 | se_t_m10 | se_t_w | se_b_m0 | se_b_m10 | se_b_w
//   se_t*: SE sigma on the training target only.
//   se_b*: SE sigma on target and sequential halving.
//   *_m0 / *_m10: N_eff = min visits over top-k; c_visit floor 0 or 10.
//   *_w: N_eff = visits of the selected move (prod's N_win analogue).
//   entropy of pi_n is reported because regret_pi rewards sharpness whenever
//   the ranking is right; read regret together with entropy.

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
ABSL_FLAG(int, num_examples, 300, "Number of examples to process.");
ABSL_FLAG(int, skip_first, 0, "Number of examples to skip first.");
ABSL_FLAG(int, seed_visits, 30,
          "PUCT visits used to seed each tree before Gumbel search.");
ABSL_FLAG(float, se_c, 1.2f,
          "c in sigma = (c_visit + c*sqrt(N_eff)/sigma_q)*q");
ABSL_FLAG(float, se_sigma_q, 0.4f,
          "sigma_q constant. <= 0 derives it from root children.");
ABSL_FLAG(bool, verbose, false, "Print per-example lines.");

namespace {

using ::data::RecordReaderOptions;
using ::data::SequentialRecordReader;
using ::game::AsLoc;
using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::game::MoveOk;
using ::mcts::ComputeImprovedPolicy;
using ::mcts::ComputeKLD;
using ::mcts::GumbelEvaluator;
using ::mcts::GumbelSearchParams;
using ::mcts::MctsNodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::TreeNode;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

static constexpr int kGroundTruthVisits = 10000;
static constexpr int kGumbelK = 16;
static constexpr size_t kCacheSize = 32768;
static constexpr int64_t kTimeoutUs = 400;
static constexpr float kEligibleFrac = 0.01f;

static constexpr int kNValues[] = {64, 100, 150, 200, 300, 400, 800};
static constexpr int kNumNValues =
    static_cast<int>(sizeof(kNValues) / sizeof(kNValues[0]));

struct Arm {
  const char* name;
  GumbelSearchParams::SigmaMode mode;
  bool halving;
  float c_visit;
  GumbelSearchParams::SigmaSeNEff n_eff;
  float shrink_k;  // completed-Q shrinkage toward v_mix (0 = off)
};

// prod = production sigma; _sK = visited-child Q shrunk toward v_mix with K
// pseudo-visits before sigma (target only); se_t_w = SE sigma, target only,
// N_eff = winner visits. Selection is identical across all target-only arms.
static const Arm kArms[] = {
    {"prod", GumbelSearchParams::SigmaMode::kProd, false, 0.0f,
     GumbelSearchParams::SigmaSeNEff::kMinTopK, 0.0f},
    {"prod_s1", GumbelSearchParams::SigmaMode::kProd, false, 0.0f,
     GumbelSearchParams::SigmaSeNEff::kMinTopK, 1.0f},
    {"prod_s2", GumbelSearchParams::SigmaMode::kProd, false, 0.0f,
     GumbelSearchParams::SigmaSeNEff::kMinTopK, 2.0f},
    {"prod_s4", GumbelSearchParams::SigmaMode::kProd, false, 0.0f,
     GumbelSearchParams::SigmaSeNEff::kMinTopK, 4.0f},
    {"se_t_w", GumbelSearchParams::SigmaMode::kSe, false, 0.0f,
     GumbelSearchParams::SigmaSeNEff::kWinner, 0.0f},
    {"se_t_w_s2", GumbelSearchParams::SigmaMode::kSe, false, 0.0f,
     GumbelSearchParams::SigmaSeNEff::kWinner, 2.0f},
};
static constexpr int kNumArms = static_cast<int>(sizeof(kArms) / sizeof(Arm));

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

// Mirrors VMixed() in gumbel.cc (anonymous namespace there).
float VMixed(const TreeNode* node) {
  const float sum_children_n = node->n - 1;
  if (sum_children_n <= 0) return node->init_util_est;
  double weighted_q = 0, visited_prob = 0;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (node->child_visits[a] > 0) {
      weighted_q += node->move_probs[a] * mcts::Q(node, a);
      visited_prob += node->move_probs[a];
    }
  }
  if (visited_prob <= 0) return node->init_util_est;
  return static_cast<float>(
      (weighted_q * sum_children_n / visited_prob + node->init_util_est) /
      (1 + sum_children_n));
}

struct GroundTruth {
  std::array<float, constants::kMaxMovesPerPosition> q_gt{};
  std::array<float, constants::kMaxMovesPerPosition> pi_prod{};
  float q_star = 0;
  Loc best = game::kNoopLoc;
  int visits = 0;
};

GroundTruth RunGroundTruth(GumbelEvaluator& evaluator, Game& game,
                           Color color_to_move, uint64_t pos_seed) {
  auto node_table = std::make_unique<MctsNodeTable>();
  TreeNode* root =
      node_table->GetOrCreate(game.board().hash(), color_to_move, false);
  core::Probability prob(pos_seed);
  auto result =
      evaluator.SearchRoot(prob, game, node_table.get(), root, color_to_move,
                           GumbelSearchParams::Builder()
                               .set_n(kGroundTruthVisits)
                               .set_k(kGumbelK)
                               .build());

  GroundTruth gt;
  gt.visits = result.visits;
  const float v_mix = VMixed(root);
  int total_child_n = 0;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    total_child_n += root->child_visits[a];
  }
  const int min_n =
      std::max(1, static_cast<int>(kEligibleFrac * total_child_n));
  gt.q_star = -1e9f;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    const int n_a = root->child_visits[a];
    gt.q_gt[a] = n_a > 0 ? mcts::Q(root, a) : v_mix;
    if (n_a >= min_n && game.IsValidMove(a, color_to_move) &&
        gt.q_gt[a] > gt.q_star) {
      gt.q_star = gt.q_gt[a];
      gt.best = AsLoc(a);
    }
  }
  gt.pi_prod = ComputeImprovedPolicy(root, 0);
  return gt;
}

struct ArmResult {
  std::array<float, constants::kMaxMovesPerPosition> pi{};
  Loc move = game::kNoopLoc;
  uint32_t visits = 0;
};

ArmResult RunArm(GumbelEvaluator& evaluator, Game& game, Color color_to_move,
                 const Arm& arm, int n, int seed_visits, uint64_t pos_seed) {
  auto node_table = std::make_unique<MctsNodeTable>();
  TreeNode* root =
      node_table->GetOrCreate(game.board().hash(), color_to_move, false);
  if (seed_visits > 0) {
    core::Probability puct_prob(pos_seed ^ 0x9e3779b97f4a7c15ULL);
    evaluator.SearchRootPuct(
        puct_prob, game, node_table.get(), root, color_to_move, seed_visits,
        PuctParams::Builder().set_kind(PuctRootSelectionPolicy::kLcb).build());
  }
  core::Probability gumbel_prob(pos_seed);
  auto params = GumbelSearchParams::Builder()
                    .set_n(n)
                    .set_k(kGumbelK)
                    .set_sigma_mode(arm.mode)
                    .set_sigma_se_halving(arm.halving)
                    .set_sigma_se_c(absl::GetFlag(FLAGS_se_c))
                    .set_sigma_se_sigma_q(absl::GetFlag(FLAGS_se_sigma_q))
                    .set_sigma_se_c_visit(arm.c_visit)
                    .set_sigma_se_n_eff(arm.n_eff)
                    .set_completed_q_shrink_k(arm.shrink_k)
                    .build();
  auto result = evaluator.SearchRoot(gumbel_prob, game, node_table.get(), root,
                                     color_to_move, params);
  return {result.pi_improved, result.mcts_move, result.visits};
}

struct Acc {
  double regret_pi = 0, regret_move = 0, kl = 0, entropy = 0;
  int top1 = 0, count = 0;
};

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string chunk_path = absl::GetFlag(FLAGS_chunk_path);
  const int num_examples = absl::GetFlag(FLAGS_num_examples);
  const int skip_first = absl::GetFlag(FLAGS_skip_first);
  const int seed_visits = absl::GetFlag(FLAGS_seed_visits);
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  CHECK(!model_path.empty()) << "--model_path required";
  CHECK(!chunk_path.empty()) << "--chunk_path required";

  auto engine = nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                                 /*batch_size=*/1,
                                 nn::GetVersionFromModelPath(model_path));
  auto nn_interface = std::make_unique<nn::NNInterface>(
      /*num_threads=*/1, kTimeoutUs, kCacheSize, std::move(engine));
  GumbelEvaluator evaluator(nn_interface.get(), /*thread_id=*/0);

  SequentialRecordReader reader(chunk_path, RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok()) << "Failed to open: " << chunk_path;

  std::array<std::array<Acc, kNumNValues>, kNumArms> acc{};
  std::vector<float> gt_gap_samples;  // q* - E_prior[q_GT]

  printf("arm,N,example,regret_pi,regret_move,top1,kl_prodgt,entropy,visits\n");
  int processed = 0, seen = 0;
  std::string record;
  while (processed < num_examples) {
    auto status = reader.ReadRecord(&record);
    if (absl::IsOutOfRange(status)) break;
    CHECK(status.ok()) << status;
    if (seen++ < skip_first) continue;

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

    const uint64_t pos_seed =
        0xdeadbeef12345678ULL ^ static_cast<uint64_t>(seen);
    GroundTruth gt = RunGroundTruth(evaluator, game, color_to_move, pos_seed);

    for (int ai = 0; ai < kNumArms; ++ai) {
      for (int ni = 0; ni < kNumNValues; ++ni) {
        ArmResult r = RunArm(evaluator, game, color_to_move, kArms[ai],
                             kNValues[ni], seed_visits, pos_seed);
        double e_q = 0, entropy = 0;
        for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
          e_q += r.pi[a] * gt.q_gt[a];
          if (r.pi[a] > 0) entropy -= r.pi[a] * std::log(r.pi[a]);
        }
        const float regret_pi = gt.q_star - static_cast<float>(e_q);
        const float regret_move =
            gt.q_star - gt.q_gt[game::AsIndex(r.move, BOARD_LEN)];
        const bool top1 = r.move == gt.best;
        const float kl = ComputeKLD(gt.pi_prod, r.pi);
        Acc& x = acc[ai][ni];
        x.regret_pi += regret_pi;
        x.regret_move += regret_move;
        x.kl += kl;
        x.entropy += entropy;
        x.top1 += top1;
        x.count += 1;
        if (verbose) {
          printf("%s,%d,%d,%.5f,%.5f,%d,%.5f,%.4f,%u\n", kArms[ai].name,
                 kNValues[ni], processed, regret_pi, regret_move, top1 ? 1 : 0,
                 kl, entropy, r.visits);
        }
      }
    }
    ++processed;
    if (processed % 25 == 0) {
      LOG(INFO) << "Processed " << processed << "/" << num_examples;
    }
  }

  printf(
      "\nSigma Regret Study  (GT: Gumbel N=%d K=%d; seed %d PUCT visits; "
      "examples %d; se_c %.2f se_sigma_q %.2f)\n",
      kGroundTruthVisits, kGumbelK, seed_visits, processed,
      absl::GetFlag(FLAGS_se_c), absl::GetFlag(FLAGS_se_sigma_q));
  printf("%-9s %-5s %-12s %-12s %-8s %-10s %-8s\n", "arm", "N", "regret_pi",
         "regret_move", "top1%", "kl_prodgt", "entropy");
  for (int ai = 0; ai < kNumArms; ++ai) {
    for (int ni = 0; ni < kNumNValues; ++ni) {
      const Acc& x = acc[ai][ni];
      if (x.count == 0) continue;
      printf("%-9s %-5d %-12.5f %-12.5f %-8.1f %-10.5f %-8.4f\n",
             kArms[ai].name, kNValues[ni], x.regret_pi / x.count,
             x.regret_move / x.count, 100.0 * x.top1 / x.count, x.kl / x.count,
             x.entropy / x.count);
    }
  }
  return 0;
}
