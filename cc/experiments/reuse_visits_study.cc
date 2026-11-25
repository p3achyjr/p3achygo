// Reuse-size visit allocation study.
//
// For each golden position i:
//   s_i ~ U{0..seed_max}: size (PUCT visits) of a pre-seeded tree, standing in
//   for a reused subtree.
//   Oracle:  fresh tree, Gumbel(oracle_n, k), its own noise seed.
//   VAR arm: seed tree (s_i PUCT visits), then Gumbel(var_total - s_i, k).
//   FIX arm: the same seed tree, then Gumbel(fixed_n, k).
// The two arms rebuild the seed tree from the same RNG seed and use the same
// Gumbel noise seed; the oracle's seeds are independent of both. The NN cache
// is off, so a search depends only on its own RNG (NN symmetries are drawn
// from it on every eval). Every search gets its own BiasCache(0.8, 0.3).
//
// Outputs (in --out_dir):
//   rows_<shard>.tsv: one row per position (see kHeader).
//   pols_<shard>.f32: per row, 8 x 362 float32:
//     prior, oracle_t, var_t, fix_t, oracle_c0, var_c0, fix_c0, seed_c0.
//     *_t = GumbelResult::pi_improved, the self-play policy training target
//     (ComputeRootImprovedPolicy, visit-dependent Q weight);
//     *_c0 = ComputeImprovedPolicy(root, 0) (as in gumbel_visits_study);
//     seed_c0 = the seed tree before the arm's Gumbel search.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
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

ABSL_FLAG(std::string, model_path, "", "TRT engine.");
ABSL_FLAG(std::string, chunk_path, "", "Golden chunk (.tfrecord.zz).");
ABSL_FLAG(std::string, out_dir, "", "Output directory.");
ABSL_FLAG(int, start, 0, "First record index to use.");
ABSL_FLAG(int, num_examples, 1000, "Number of positions.");
ABSL_FLAG(int, num_threads, 16, "Worker threads (= NN batch).");
ABSL_FLAG(int, seed_max, 400, "Seed tree size s ~ U{0..seed_max}.");
ABSL_FLAG(int, var_total, 600, "VAR arm runs Gumbel(var_total - s).");
ABSL_FLAG(int, fixed_n, 400, "FIX arm runs Gumbel(fixed_n).");
ABSL_FLAG(int, oracle_n, 3200, "Oracle Gumbel visits.");
ABSL_FLAG(int, k, 16, "Gumbel k for every search.");
ABSL_FLAG(float, noise_scaling, 1.0f, "Gumbel noise scaling, every search.");
ABSL_FLAG(uint64_t, run_seed, 20261125, "Base RNG seed.");

namespace {

using ::game::AsLoc;
using ::game::Color;
using ::game::Game;
using ::game::Loc;
using ::mcts::BiasCache;
using ::mcts::ComputeImprovedPolicy;
using ::mcts::GumbelEvaluator;
using ::mcts::GumbelSearchParams;
using ::mcts::MctsNodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::TreeNode;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;
using Policy = std::array<float, constants::kMaxMovesPerPosition>;

constexpr int64_t kTimeoutUs = 400;
constexpr char kHeader[] =
    "idx\tcolor\tstones\tkomi\ts\tseed_n\tseed_fp_match\tvar_req\tvar_spent\t"
    "fix_spent\toracle_spent\tvar_move\tfix_move\toracle_move\n";

struct Position {
  int idx;
  std::array<Color, constants::kNumBoardLocs> board;
  std::array<int16_t, constants::kNumLastMoves> last_moves;
  Color color;
  float komi;
};

uint64_t Mix(uint64_t x) {  // splitmix64
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

uint64_t SeedFor(uint64_t run_seed, int idx, int stream) {
  return Mix(Mix(run_seed ^ Mix(static_cast<uint64_t>(idx))) + stream);
}

Game MakeGame(const Position& p) {
  game::Board board(p.komi);
  for (Color c : {BLACK, WHITE}) {
    for (int i = 0; i < constants::kNumBoardLocs; ++i) {
      if (p.board[i] == c) {
        CHECK(game::MoveOk(board.PlayMove(AsLoc(i), c)));
      }
    }
  }
  absl::InlinedVector<game::Move, constants::kNumLastMoves> last_moves;
  for (int16_t m : p.last_moves) last_moves.push_back({p.color, AsLoc(m)});
  return Game(board, last_moves, /*init_mv_num=*/0);
}

// Root visits and a hash of the root's child visit counts.
std::pair<int, uint64_t> Fingerprint(const TreeNode* root) {
  uint64_t h = Mix(root->n);
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    h = Mix(h ^ (static_cast<uint64_t>(root->child_visits[a]) << 9 | a));
  }
  return {root->n, h};
}

GumbelSearchParams Params(int n) {
  return GumbelSearchParams::Builder()
      .set_n(n)
      .set_k(absl::GetFlag(FLAGS_k))
      .set_noise_scaling(absl::GetFlag(FLAGS_noise_scaling))
      .set_nonroot_var_scale_prior_visits(-1)  // prod self-play: off
      .build();
}

struct ArmResult {
  Policy pi_t, pi, seed_pi;
  int seed_n = 0;
  uint64_t seed_fp = 0;
  int spent = 0;
  int move = -1;
};

// PUCT seed of `s` visits (RNG `seed_seed`), then Gumbel(n) (RNG `arm_seed`).
ArmResult RunArm(nn::NNInterface* nn, int tid, const Position& p, int s, int n,
                 uint64_t seed_seed, uint64_t arm_seed) {
  Game game = MakeGame(p);
  auto table = std::make_unique<MctsNodeTable>();
  TreeNode* root = table->GetOrCreate(game.board().hash(), p.color, false);
  BiasCache bias_cache(0.8f, 0.3f);
  GumbelEvaluator evaluator(nn, tid, &bias_cache);
  if (s > 0) {
    core::Probability seed_prob(seed_seed);
    evaluator.SearchRootPuct(
        seed_prob, game, table.get(), root, p.color, s,
        PuctParams::Builder().set_kind(PuctRootSelectionPolicy::kLcb).build());
  }
  ArmResult r;
  std::tie(r.seed_n, r.seed_fp) = Fingerprint(root);
  r.seed_pi = ComputeImprovedPolicy(root, 0);
  core::Probability arm_prob(arm_seed);
  const auto res = evaluator.SearchRoot(arm_prob, game, table.get(), root,
                                        p.color, Params(n));
  r.spent = root->n - r.seed_n;
  r.pi_t = res.pi_improved;
  r.pi = ComputeImprovedPolicy(root, 0);
  r.move = game::AsIndex(res.mcts_move, BOARD_LEN);
  return r;
}

void Worker(nn::NNInterface* nn, int tid, int num_threads,
            const std::vector<Position>* positions, std::FILE* tsv,
            std::FILE* pols, absl::Mutex* out_mu) {
  nn->RegisterThread(tid);
  const uint64_t run_seed = absl::GetFlag(FLAGS_run_seed);
  const int seed_max = absl::GetFlag(FLAGS_seed_max);
  for (size_t j = tid; j < positions->size(); j += num_threads) {
    const Position& p = (*positions)[j];
    core::Probability size_prob(SeedFor(run_seed, p.idx, 0));
    const int s = std::min(
        seed_max, static_cast<int>(size_prob.Uniform() * (seed_max + 1)));

    // Oracle: fresh tree, own noise.
    Game game = MakeGame(p);
    auto table = std::make_unique<MctsNodeTable>();
    TreeNode* root = table->GetOrCreate(game.board().hash(), p.color, false);
    BiasCache oracle_cache(0.8f, 0.3f);
    GumbelEvaluator oracle_eval(nn, tid, &oracle_cache);
    core::Probability oracle_prob(SeedFor(run_seed, p.idx, 1));
    const auto ores =
        oracle_eval.SearchRoot(oracle_prob, game, table.get(), root, p.color,
                               Params(absl::GetFlag(FLAGS_oracle_n)));
    const Policy prior = root->move_probs;
    const Policy oracle_pi = ComputeImprovedPolicy(root, 0);
    const Policy oracle_t = ores.pi_improved;
    const int oracle_spent = root->n;
    const int oracle_move = game::AsIndex(ores.mcts_move, BOARD_LEN);
    table.reset();

    const uint64_t seed_seed = SeedFor(run_seed, p.idx, 2);
    const uint64_t arm_seed = SeedFor(run_seed, p.idx, 3);
    const int var_req = absl::GetFlag(FLAGS_var_total) - s;
    const ArmResult var = RunArm(nn, tid, p, s, var_req, seed_seed, arm_seed);
    const ArmResult fix = RunArm(nn, tid, p, s, absl::GetFlag(FLAGS_fixed_n),
                                 seed_seed, arm_seed);

    int stones = 0;
    for (Color c : p.board) stones += c != EMPTY;
    absl::MutexLock l(out_mu);
    std::fprintf(
        tsv, "%d\t%d\t%d\t%.1f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
        p.idx, static_cast<int>(p.color), stones, p.komi, s, var.seed_n,
        var.seed_n == fix.seed_n && var.seed_fp == fix.seed_fp, var_req,
        var.spent, fix.spent, oracle_spent, var.move, fix.move, oracle_move);
    for (const Policy* pol : {&prior, &oracle_t, &var.pi_t, &fix.pi_t,
                              &oracle_pi, &var.pi, &fix.pi, &var.seed_pi}) {
      std::fwrite(pol->data(), sizeof(float), pol->size(), pols);
    }
    std::fflush(tsv);
    std::fflush(pols);
  }
  nn->UnregisterThread(tid);
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  const std::string out_dir = absl::GetFlag(FLAGS_out_dir);
  const int start = absl::GetFlag(FLAGS_start);
  const int num_examples = absl::GetFlag(FLAGS_num_examples);
  const int num_threads = absl::GetFlag(FLAGS_num_threads);
  CHECK(!out_dir.empty());

  std::vector<Position> positions;
  data::SequentialRecordReader reader(absl::GetFlag(FLAGS_chunk_path),
                                      data::RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok());
  std::string record;
  for (int idx = 0; static_cast<int>(positions.size()) < num_examples; ++idx) {
    CHECK(reader.ReadRecord(&record).ok()) << "chunk ended at record " << idx;
    if (idx < start) continue;
    Example ex;
    CHECK(ex.ParseFromString(record));
    Position p;
    p.idx = idx;
    const std::string& b = GetFeatureValues<std::string>("board", ex).Get(0);
    const std::string& lm =
        GetFeatureValues<std::string>("last_moves", ex).Get(0);
    const std::string& c = GetFeatureValues<std::string>("color", ex).Get(0);
    CHECK_EQ(b.size(), sizeof(p.board));
    CHECK_EQ(lm.size(), sizeof(p.last_moves));
    CHECK_EQ(c.size(), sizeof(Color));
    std::memcpy(p.board.data(), b.data(), b.size());
    std::memcpy(p.last_moves.data(), lm.data(), lm.size());
    std::memcpy(&p.color, c.data(), sizeof(Color));
    p.komi = GetFeatureValues<float>("komi", ex).Get(0);
    positions.push_back(p);
  }

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  auto engine =
      nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                       num_threads, nn::GetVersionFromModelPath(model_path));
  nn::NNInterface nn_interface(num_threads, kTimeoutUs, /*cache_size=*/0,
                               std::move(engine));

  const std::string tag = std::to_string(start);
  std::FILE* tsv = std::fopen((out_dir + "/rows_" + tag + ".tsv").c_str(), "w");
  std::FILE* pols =
      std::fopen((out_dir + "/pols_" + tag + ".f32").c_str(), "wb");
  CHECK(tsv && pols);
  std::fputs(kHeader, tsv);
  absl::Mutex out_mu;
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(Worker, &nn_interface, t, num_threads, &positions, tsv,
                         pols, &out_mu);
  }
  for (auto& t : threads) t.join();
  std::fclose(tsv);
  std::fclose(pols);
  LOG(INFO) << "done: " << positions.size() << " positions";
  return 0;
}
