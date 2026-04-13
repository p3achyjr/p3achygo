#include "cc/recorder/tf_recorder.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cc/constants/constants.h"
#include "cc/core/doctest_include.h"
#include "cc/game/board.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/recorder/move_search_stats.h"

namespace recorder {
namespace {

using game::Board;
using game::Game;

// Build a minimal 2-pass game (Black passes, White passes => result by resign).
Game MakePassGame() {
  Game game;
  game.Pass(BLACK);
  game.Pass(WHITE);
  game.SetWinner(BLACK);
  return game;
}

// Build a MoveSearchStats for a non-sampled move with the given stddev and
// visit_count_pre.
MoveSearchStats MakeStat(float v_outcome_stddev, float visit_count_pre,
                         float sel_mult_modifier = 1.0f, float kld = 0.1f) {
  return MoveSearchStats::Builder{}
      .sampled_raw_policy(false)
      .nn_q(0.1f)
      .mcts_q(0.1f)
      .nn_mcts_diff(0.0f)
      .v_outcome_stddev(v_outcome_stddev)
      .prior_entropy(1.0f)
      .nn_uncertainty(0.05f)
      .kld(kld)
      .pre_kld(kld)
      .sel_mult_modifier(sel_mult_modifier)
      .sel_mult_modifier_weight(1.0f)
      .visit_count(128.0f)
      .visit_count_pre(visit_count_pre)
      .build();
}

// Record one game into the recorder and flush, returning the .stats file path
// (written to /tmp). Caller must remove the files.
std::string RecordAndFlush(TfRecorder* recorder,
                           const std::vector<MoveSearchStats>& stats) {
  Game game = MakePassGame();
  Board init_board;
  const int num_moves = game.num_moves();

  TfRecorder::ImprovedPolicies pis(num_moves);
  for (auto& pi : pis) pi.fill(1.0f / constants::kMaxMovesPerPosition);

  std::vector<uint8_t> trainable(num_moves, 1);
  std::vector<float> root_qs(num_moves, 0.1f);
  std::vector<float> root_scores(num_moves, 0.0f);
  std::vector<float> klds(num_moves, 0.1f);

  recorder->RecordGame(0, init_board, game, pis, trainable, root_qs,
                       root_scores, klds, stats);
  recorder->Flush();

  // Find the .stats file written to /tmp/test_chunks/
  // The filename format includes gen/batch/etc; glob for it.
  for (const auto* ext : {".stats"}) {
    for (int i = 0; i < 1000; ++i) {
      // We use a known-prefix path; just scan /tmp/test_chunks/
      (void)ext;
    }
  }
  // Return empty — caller uses glob.
  return "";
}

std::string FindStats(const std::string& dir) {
  // Use popen to find the stats file.
  std::string cmd = "ls " + dir + "/*.stats 2>/dev/null | head -1";
  FILE* f = popen(cmd.c_str(), "r");
  if (!f) return "";
  char buf[512];
  std::string result;
  if (fgets(buf, sizeof(buf), f)) {
    result = buf;
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
      result.pop_back();
  }
  pclose(f);
  return result;
}

bool StatsFileHasField(const std::string& path, const std::string& field) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind(field, 0) == 0) return true;
  }
  return false;
}

std::string GetScalarLine(const std::string& path, const std::string& key) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind(key + "=", 0) == 0) return line;
  }
  return "";
}

TEST_CASE("stats file has expected_std.n* and v_outcome_stddev_adj rows") {
  const std::string dir = "/tmp/tf_recorder_test";
  std::system(("mkdir -p " + dir).c_str());

  auto recorder = TfRecorder::Create(dir, /*num_threads=*/1, /*gen=*/0,
                                     /*worker_id=*/"test");

  // Provide 2 moves (one per pass) with different visit_count_pre bins.
  std::vector<MoveSearchStats> stats = {
      MakeStat(/*v_outcome_stddev=*/0.15f, /*visit_count_pre=*/30.0f),
      MakeStat(/*v_outcome_stddev=*/0.20f, /*visit_count_pre=*/80.0f),
  };

  Game game = MakePassGame();
  Board init_board;
  const int num_moves = game.num_moves();
  TfRecorder::ImprovedPolicies pis(num_moves);
  for (auto& pi : pis) pi.fill(1.0f / constants::kMaxMovesPerPosition);
  std::vector<uint8_t> trainable(num_moves, 1);
  std::vector<float> root_qs(num_moves, 0.1f);
  std::vector<float> root_scores(num_moves, 0.0f);
  std::vector<float> klds(num_moves, 0.1f);

  recorder->RecordGame(0, init_board, game, pis, trainable, root_qs,
                       root_scores, klds, stats);
  recorder->Flush();

  std::string stats_path = FindStats(dir);
  REQUIRE(!stats_path.empty());

  CHECK(StatsFileHasField(stats_path, "v_outcome_stddev "));
  CHECK(StatsFileHasField(stats_path, "v_outcome_stddev_adj "));
  CHECK(StatsFileHasField(stats_path, "freq_weight "));
  CHECK(StatsFileHasField(stats_path, "sel_mult_modifier "));

  // expected_std bins for n=30 and n=80.
  CHECK(StatsFileHasField(stats_path, "expected_std.n30="));
  CHECK(StatsFileHasField(stats_path, "expected_std.n80="));

  // expected_std.n30 should be ~0.15 (only one sample in that bin).
  std::string n30 = GetScalarLine(stats_path, "expected_std.n30");
  REQUIRE(!n30.empty());
  float val = std::stof(n30.substr(n30.find('=') + 1));
  CHECK(val == doctest::Approx(0.15f).epsilon(0.001f));

  std::system(("rm -rf " + dir).c_str());
}

TEST_CASE("v_outcome_stddev_adj row appears before expected_std scalars") {
  const std::string dir = "/tmp/tf_recorder_test3";
  std::system(("mkdir -p " + dir).c_str());

  auto recorder = TfRecorder::Create(dir, 1, 0, "test");
  std::vector<MoveSearchStats> stats = {
      MakeStat(0.15f, 30.0f),
      MakeStat(0.20f, 80.0f),
  };
  Game game = MakePassGame();
  Board init_board;
  const int num_moves = game.num_moves();
  TfRecorder::ImprovedPolicies pis(num_moves);
  for (auto& pi : pis) pi.fill(1.0f / constants::kMaxMovesPerPosition);
  std::vector<uint8_t> trainable(num_moves, 1);
  std::vector<float> root_qs(num_moves, 0.1f);
  std::vector<float> root_scores(num_moves, 0.0f);
  std::vector<float> klds(num_moves, 0.1f);
  recorder->RecordGame(0, init_board, game, pis, trainable, root_qs,
                       root_scores, klds, stats);
  recorder->Flush();

  std::string stats_path = FindStats(dir);
  REQUIRE(!stats_path.empty());

  // Find line numbers of v_outcome_stddev_adj row and first expected_std
  // scalar.
  std::ifstream f(stats_path);
  std::string line;
  int adj_line = -1, expected_std_line = -1, lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    if (adj_line < 0 && line.rfind("v_outcome_stddev_adj ", 0) == 0)
      adj_line = lineno;
    if (expected_std_line < 0 && line.rfind("expected_std.n", 0) == 0)
      expected_std_line = lineno;
  }
  REQUIRE(adj_line > 0);
  REQUIRE(expected_std_line > 0);
  CHECK(adj_line < expected_std_line);

  std::system(("rm -rf " + dir).c_str());
}

TEST_CASE("ComputePercentiles: correct p01/p50/p99 on known distribution") {
  const std::string dir = "/tmp/tf_recorder_test4";
  std::system(("mkdir -p " + dir).c_str());

  // Feed 100 moves with kld values 0.01, 0.02, ..., 1.00.
  // p01 ~ 0.01, p50 ~ 0.50, p99 ~ 0.99.
  auto recorder = TfRecorder::Create(dir, 1, 0, "test");

  // Need enough moves for a valid game — use 100 moves with varied kld.
  // We'll record 50 2-pass games to get 100 stat entries.
  for (int g = 0; g < 50; ++g) {
    Game game = MakePassGame();
    Board init_board;
    const int num_moves = game.num_moves();
    TfRecorder::ImprovedPolicies pis(num_moves);
    for (auto& pi : pis) pi.fill(1.0f / constants::kMaxMovesPerPosition);
    std::vector<uint8_t> trainable(num_moves, 1);
    std::vector<float> root_qs(num_moves, 0.1f);
    std::vector<float> root_scores(num_moves, 0.0f);
    // kld values: 2*g+1 and 2*g+2 out of 100, scaled to [0.01, 1.00].
    std::vector<float> klds = {(2 * g + 1) / 100.0f, (2 * g + 2) / 100.0f};
    std::vector<MoveSearchStats> stats = {
        MakeStat(0.15f, 30.0f, 1.0f, klds[0]),
        MakeStat(0.15f, 30.0f, 1.0f, klds[1]),
    };
    recorder->RecordGame(0, init_board, game, pis, trainable, root_qs,
                         root_scores, klds, stats);
  }
  recorder->Flush();

  std::string stats_path = FindStats(dir);
  REQUIRE(!stats_path.empty());

  // Parse the pre_kld row: "pre_kld  p01 p05 ... p99"
  std::ifstream f(stats_path);
  std::string line;
  std::vector<float> pcts;
  while (std::getline(f, line)) {
    if (line.rfind("pre_kld ", 0) != 0) continue;
    std::istringstream ss(line.substr(7));
    float v;
    while (ss >> v) pcts.push_back(v);
    break;
  }
  REQUIRE(pcts.size() == 21);
  CHECK(pcts[0] == doctest::Approx(0.01f).epsilon(0.02f));   // p01
  CHECK(pcts[10] == doctest::Approx(0.50f).epsilon(0.05f));  // p50
  CHECK(pcts[20] == doctest::Approx(0.99f).epsilon(0.02f));  // p99

  std::system(("rm -rf " + dir).c_str());
}

TEST_CASE("freq_weight percentiles written and sensible") {
  const std::string dir = "/tmp/tf_recorder_test5";
  std::system(("mkdir -p " + dir).c_str());

  auto recorder = TfRecorder::Create(dir, 1, 0, "test");
  // Two moves with equal kld => avg_kld = kld => freq_weight = 0.5 + 0.5 = 1.0
  // for both.
  std::vector<MoveSearchStats> stats = {
      MakeStat(0.15f, 30.0f, 1.0f, 0.2f),
      MakeStat(0.15f, 30.0f, 1.0f, 0.2f),
  };
  Game game = MakePassGame();
  Board init_board;
  const int num_moves = game.num_moves();
  TfRecorder::ImprovedPolicies pis(num_moves);
  for (auto& pi : pis) pi.fill(1.0f / constants::kMaxMovesPerPosition);
  std::vector<uint8_t> trainable(num_moves, 1);
  std::vector<float> root_qs(num_moves, 0.1f);
  std::vector<float> root_scores(num_moves, 0.0f);
  std::vector<float> klds(num_moves, 0.2f);
  recorder->RecordGame(0, init_board, game, pis, trainable, root_qs,
                       root_scores, klds, stats);
  recorder->Flush();

  std::string stats_path = FindStats(dir);
  REQUIRE(!stats_path.empty());
  CHECK(StatsFileHasField(stats_path, "freq_weight "));

  // With equal klds, all freq_weights = 1.0, so all percentiles = 1.0.
  std::ifstream f(stats_path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("freq_weight ", 0) != 0) continue;
    std::istringstream ss(line.substr(11));
    float v;
    while (ss >> v) CHECK(v == doctest::Approx(1.0f).epsilon(0.001f));
    break;
  }

  std::system(("rm -rf " + dir).c_str());
}

TEST_CASE("sel_mult_mean scalar is written") {
  const std::string dir = "/tmp/tf_recorder_test2";
  std::system(("mkdir -p " + dir).c_str());

  auto recorder = TfRecorder::Create(dir, /*num_threads=*/1, /*gen=*/0,
                                     /*worker_id=*/"test");

  std::vector<MoveSearchStats> stats = {
      MakeStat(0.15f, 30.0f, /*sel_mult_modifier=*/1.5f),
      MakeStat(0.20f, 80.0f, /*sel_mult_modifier=*/0.5f),
  };

  Game game = MakePassGame();
  Board init_board;
  const int num_moves = game.num_moves();
  TfRecorder::ImprovedPolicies pis(num_moves);
  for (auto& pi : pis) pi.fill(1.0f / constants::kMaxMovesPerPosition);
  std::vector<uint8_t> trainable(num_moves, 1);
  std::vector<float> root_qs(num_moves, 0.1f);
  std::vector<float> root_scores(num_moves, 0.0f);
  std::vector<float> klds(num_moves, 0.1f);

  recorder->RecordGame(0, init_board, game, pis, trainable, root_qs,
                       root_scores, klds, stats);
  recorder->Flush();

  std::string stats_path = FindStats(dir);
  REQUIRE(!stats_path.empty());

  // sel_mult_mean = (1.5 + 0.5) / 2 = 1.0
  std::string mean_line = GetScalarLine(stats_path, "sel_mult_mean");
  REQUIRE(!mean_line.empty());
  float mean = std::stof(mean_line.substr(mean_line.find('=') + 1));
  CHECK(mean == doctest::Approx(1.0f).epsilon(0.001f));

  std::system(("rm -rf " + dir).c_str());
}

}  // namespace
}  // namespace recorder
