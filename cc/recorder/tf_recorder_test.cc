#include "cc/recorder/tf_recorder.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "cc/constants/constants.h"
#include "cc/core/doctest_include.h"
#include "cc/data/tfrecord/compression_options.h"
#include "cc/data/tfrecord/record_reader.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/recorder/move_search_stats.h"
#include "example.pb.h"

namespace recorder {
namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Reference implementation of the TD target formula (mirrors tf_recorder.cc).
// ---------------------------------------------------------------------------
float ExpWeightedQ(const std::vector<float>& root_qs, int move_num,
                   float lambda) {
  int horizon = static_cast<int>(root_qs.size()) - move_num - 1;
  float N = 0.0f;
  float q = 0.0f;
  for (int i = 0; i <= horizon; ++i) {
    float w = std::pow(lambda, static_cast<float>(i));
    float v_mult = (i % 2 == 0) ? 1.0f : -1.0f;
    N += w;
    q += v_mult * w * root_qs[move_num + i];
  }
  return q / N;
}

// ---------------------------------------------------------------------------
// Proto helpers.
// ---------------------------------------------------------------------------
float GetFloat(const tensorflow::Example& ex, const std::string& key) {
  return ex.features().feature().at(key).float_list().value(0);
}

std::array<float, constants::kMaxMovesPerPosition> DecodePi(
    const tensorflow::Example& ex) {
  const std::string& bytes =
      ex.features().feature().at("pi").bytes_list().value(0);
  std::array<float, constants::kMaxMovesPerPosition> pi{};
  std::memcpy(pi.data(), bytes.data(),
              sizeof(float) * constants::kMaxMovesPerPosition);
  return pi;
}

// ---------------------------------------------------------------------------
// Test fixture helpers.
// ---------------------------------------------------------------------------
PolicyArray MakeUniformPi() {
  PolicyArray pi{};
  pi.fill(1.0f / pi.size());
  return pi;
}

// One-hot pi — lets us verify which move's data appears in an example.
PolicyArray MakePiForMove(int move_num) {
  PolicyArray pi{};
  pi.fill(0.0f);
  pi[move_num] = 1.0f;
  return pi;
}

MoveSearchRecord SimpleRecord(PolicyArray pi, bool trainable, float kld = 0.0f,
                              float root_q = 0.5f, float root_score = 0.0f) {
  return MoveSearchRecord::Builder()
      .mcts_pi(pi)
      .move_trainable(trainable)
      .root_q_outcome(root_q)
      .root_score(root_score)
      .kld(kld)
      .root(nullptr)
      .move_stats(MoveSearchStats::Builder().build())
      .build();
}

// N-pass game (BLACK, WHITE, BLACK, …), result written.
game::Game MakePassGame(int n) {
  game::Game g;
  for (int i = 0; i < n; ++i) {
    g.PlayMove(game::kPassLoc, i % 2 == 0 ? BLACK : WHITE);
  }
  g.WriteResult();
  return g;
}

// Read all TFExamples from the first .tfrecord.zz found in dir.
// Returns empty vector if no file was written (e.g. no trainable moves).
std::vector<tensorflow::Example> ReadExamples(const std::string& dir) {
  std::string path;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.path().string().find(".tfrecord.zz") != std::string::npos) {
      path = e.path().string();
      break;
    }
  }
  if (path.empty()) return {};

  data::SequentialRecordReader reader(path, data::RecordReaderOptions::Zlib());
  REQUIRE(reader.Init().ok());

  std::vector<tensorflow::Example> out;
  std::string record;
  while (reader.ReadRecord(&record).ok()) {
    tensorflow::Example ex;
    REQUIRE(ex.ParseFromString(record));
    out.push_back(std::move(ex));
  }
  return out;
}

// Record one game and flush; return all examples.
std::vector<tensorflow::Example> RunAndRead(
    const std::string& dir, game::Game g,
    const std::vector<MoveSearchRecord>& move_infos) {
  auto rec = TfRecorder::Create(dir, /*num_threads=*/1, /*gen=*/0,
                                /*worker_id=*/"test");
  rec->RecordGame(/*thread_id=*/0, game::Board{}, g, move_infos);
  rec->Flush();
  return ReadExamples(dir);
}

// Convenience: 3-pass game.
std::vector<tensorflow::Example> RunAndRead(
    const std::string& dir, const std::vector<MoveSearchRecord>& move_infos) {
  return RunAndRead(dir, MakePassGame(3), move_infos);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("TfRecorder: TD targets written correctly") {
  char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
  REQUIRE(mkdtemp(tmpdir) != nullptr);
  std::string dir = std::string(tmpdir) + "/";

  const std::vector<float> root_qs = {0.5f, 0.3f, 0.7f};
  const std::vector<float> root_scores = {5.0f, -3.0f, 7.0f};

  // kld=0 everywhere → avg_kld=0 → freq_weight=1 → exactly 3 examples.
  std::vector<MoveSearchRecord> move_infos;
  for (int i = 0; i < 3; ++i) {
    move_infos.push_back(SimpleRecord(MakeUniformPi(), /*trainable=*/true,
                                      /*kld=*/0.0f, root_qs[i],
                                      root_scores[i]));
  }

  auto examples = RunAndRead(dir, move_infos);
  REQUIRE(examples.size() == 3);

  for (int m = 0; m < 3; ++m) {
    const auto& ex = examples[m];
    CHECK(
        GetFloat(ex, "q6") ==
        doctest::Approx(ExpWeightedQ(root_qs, m, 5.0f / 6.0f)).epsilon(1e-4f));
    CHECK(GetFloat(ex, "q16") ==
          doctest::Approx(ExpWeightedQ(root_qs, m, 15.0f / 16.0f))
              .epsilon(1e-4f));
    CHECK(GetFloat(ex, "q50") ==
          doctest::Approx(ExpWeightedQ(root_qs, m, 49.0f / 50.0f))
              .epsilon(1e-4f));
    CHECK(GetFloat(ex, "q6_score") ==
          doctest::Approx(ExpWeightedQ(root_scores, m, 5.0f / 6.0f))
              .epsilon(1e-4f));
    CHECK(GetFloat(ex, "q16_score") ==
          doctest::Approx(ExpWeightedQ(root_scores, m, 15.0f / 16.0f))
              .epsilon(1e-4f));
    CHECK(GetFloat(ex, "q50_score") ==
          doctest::Approx(ExpWeightedQ(root_scores, m, 49.0f / 50.0f))
              .epsilon(1e-4f));
  }

  fs::remove_all(tmpdir);
}

TEST_CASE("TfRecorder: horizon=0 at last move collapses to raw root_q") {
  // At the last move, horizon = game.num_moves() - move_num - 1 = 0, so the
  // TD sum has only i=0: N=1, q_short_term = +1 * root_q. All three lambdas
  // yield the same value.
  char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
  REQUIRE(mkdtemp(tmpdir) != nullptr);
  std::string dir = std::string(tmpdir) + "/";

  const float last_q = 0.42f;
  const float last_score = 7.5f;

  // 2-move game; only the last move (move 1) is trainable.
  std::vector<MoveSearchRecord> move_infos = {
      SimpleRecord(MakeUniformPi(), /*trainable=*/false),
      SimpleRecord(MakeUniformPi(), /*trainable=*/true, /*kld=*/0.0f, last_q,
                   last_score),
  };

  auto examples = RunAndRead(dir, MakePassGame(2), move_infos);
  REQUIRE(examples.size() == 1);

  CHECK(GetFloat(examples[0], "q6") == doctest::Approx(last_q).epsilon(1e-5f));
  CHECK(GetFloat(examples[0], "q16") == doctest::Approx(last_q).epsilon(1e-5f));
  CHECK(GetFloat(examples[0], "q50") == doctest::Approx(last_q).epsilon(1e-5f));
  CHECK(GetFloat(examples[0], "q6_score") ==
        doctest::Approx(last_score).epsilon(1e-5f));

  fs::remove_all(tmpdir);
}

TEST_CASE(
    "TfRecorder: high-KLD move duplicated deterministically (freq_weight=2)") {
  // With 3 trainable moves and kld=[k, 0, 0]:
  //   avg_kld = k/3
  //   freq_weight(move 0) = 0.5 + 0.5*(k / (k/3)) = 0.5 + 1.5 = 2.0
  //   → floor(2.0) = 2 guaranteed copies, probabilistic extra = 0.0 (never)
  //   freq_weight(moves 1,2) = 0.5 → floor=0, 50% probabilistic each
  //
  // We can therefore assert that examples[0] and examples[1] are both copies
  // of move 0's data (pi[0]=1), regardless of what the probabilistic copies do.
  char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
  REQUIRE(mkdtemp(tmpdir) != nullptr);
  std::string dir = std::string(tmpdir) + "/";

  std::vector<MoveSearchRecord> move_infos = {
      SimpleRecord(MakePiForMove(0), /*trainable=*/true, /*kld=*/9.0f),
      SimpleRecord(MakePiForMove(1), /*trainable=*/true, /*kld=*/0.0f),
      SimpleRecord(MakePiForMove(2), /*trainable=*/true, /*kld=*/0.0f),
  };

  auto examples = RunAndRead(dir, move_infos);

  // At least 2 (the two guaranteed copies of move 0).
  REQUIRE(examples.size() >= 2);
  CHECK(DecodePi(examples[0])[0] == doctest::Approx(1.0f));
  CHECK(DecodePi(examples[1])[0] == doctest::Approx(1.0f));

  fs::remove_all(tmpdir);
}

TEST_CASE("TfRecorder: two games accumulated before flush") {
  // RecordGame can be called multiple times before Flush. Both games'
  // trainable moves should appear in the single output file.
  char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
  REQUIRE(mkdtemp(tmpdir) != nullptr);
  std::string dir = std::string(tmpdir) + "/";

  auto rec = TfRecorder::Create(dir, /*num_threads=*/1, /*gen=*/0,
                                /*worker_id=*/"test");

  // Game 1: 2-move, move 0 trainable (pi[0]=1), move 1 not.
  std::vector<MoveSearchRecord> infos1 = {
      SimpleRecord(MakePiForMove(0), /*trainable=*/true),
      SimpleRecord(MakeUniformPi(), /*trainable=*/false),
  };
  rec->RecordGame(/*thread_id=*/0, game::Board{}, MakePassGame(2), infos1);

  // Game 2: 2-move, move 0 not trainable, move 1 trainable (pi[1]=1).
  std::vector<MoveSearchRecord> infos2 = {
      SimpleRecord(MakeUniformPi(), /*trainable=*/false),
      SimpleRecord(MakePiForMove(1), /*trainable=*/true),
  };
  rec->RecordGame(/*thread_id=*/0, game::Board{}, MakePassGame(2), infos2);

  rec->Flush();
  auto examples = ReadExamples(dir);

  REQUIRE(examples.size() == 2);
  // First example is from game 1 move 0.
  CHECK(DecodePi(examples[0])[0] == doctest::Approx(1.0f));
  // Second example is from game 2 move 1.
  CHECK(DecodePi(examples[1])[1] == doctest::Approx(1.0f));

  fs::remove_all(tmpdir);
}

TEST_CASE("TfRecorder: score_margin is player-relative") {
  // score_margin = bscore - wscore for BLACK's move, wscore - bscore for
  // WHITE's. The signs should be opposite for the two colors.
  char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
  REQUIRE(mkdtemp(tmpdir) != nullptr);
  std::string dir = std::string(tmpdir) + "/";

  // 2-move game: move 0 = BLACK, move 1 = WHITE, both trainable.
  std::vector<MoveSearchRecord> move_infos = {
      SimpleRecord(MakeUniformPi(), /*trainable=*/true),
      SimpleRecord(MakeUniformPi(), /*trainable=*/true),
  };

  auto examples = RunAndRead(dir, MakePassGame(2), move_infos);
  REQUIRE(examples.size() == 2);

  float margin_black = GetFloat(examples[0], "score_margin");
  float margin_white = GetFloat(examples[1], "score_margin");

  // One player's margin is the negation of the other's.
  CHECK(margin_black == doctest::Approx(-margin_white).epsilon(1e-5f));

  fs::remove_all(tmpdir);
}

TEST_CASE("TfRecorder: pi_aux encodes next move loc as int16") {
  // pi_aux = next_move.loc cast to int16 = row*BOARD_LEN + col.
  // For the last move the sentinel next_move is kPassLoc → 19*19 = 361.
  char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
  REQUIRE(mkdtemp(tmpdir) != nullptr);
  std::string dir = std::string(tmpdir) + "/";

  // 2-move game: BLACK at (0,5), WHITE at (0,7). Both trainable.
  //   Move 0 pi_aux = WHITE's loc = (0,7) → 0*19+7 = 7
  //   Move 1 pi_aux = sentinel kPassLoc     → 19*19+0 = 361
  game::Game g;
  g.PlayMove(game::Loc{0, 5}, BLACK);
  g.PlayMove(game::Loc{0, 7}, WHITE);
  g.WriteResult();

  std::vector<MoveSearchRecord> move_infos = {
      SimpleRecord(MakeUniformPi(), /*trainable=*/true),
      SimpleRecord(MakeUniformPi(), /*trainable=*/true),
  };

  auto examples = RunAndRead(dir, g, move_infos);
  REQUIRE(examples.size() == 2);

  auto decode_pi_aux = [](const tensorflow::Example& ex) -> int16_t {
    const std::string& bytes =
        ex.features().feature().at("pi_aux").bytes_list().value(0);
    int16_t v = 0;
    std::memcpy(&v, bytes.data(), sizeof(int16_t));
    return v;
  };

  CHECK(decode_pi_aux(examples[0]) == 0 * BOARD_LEN + 7);   // next = (0,7)
  CHECK(decode_pi_aux(examples[1]) == 19 * BOARD_LEN + 0);  // sentinel kPassLoc

  fs::remove_all(tmpdir);
}

TEST_CASE("TfRecorder: multi-thread accumulation flushed together") {
  // RecordGame calls from different thread_ids go into separate per-thread
  // record buffers; Flush drains all of them into one output file.
  char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
  REQUIRE(mkdtemp(tmpdir) != nullptr);
  std::string dir = std::string(tmpdir) + "/";

  auto rec = TfRecorder::Create(dir, /*num_threads=*/2, /*gen=*/0,
                                /*worker_id=*/"test");

  // Thread 0: 2-move game, move 0 trainable (pi[0]=1).
  std::vector<MoveSearchRecord> infos0 = {
      SimpleRecord(MakePiForMove(0), /*trainable=*/true),
      SimpleRecord(MakeUniformPi(), /*trainable=*/false),
  };
  rec->RecordGame(/*thread_id=*/0, game::Board{}, MakePassGame(2), infos0);

  // Thread 1: 2-move game, move 1 trainable (pi[1]=1).
  std::vector<MoveSearchRecord> infos1 = {
      SimpleRecord(MakeUniformPi(), /*trainable=*/false),
      SimpleRecord(MakePiForMove(1), /*trainable=*/true),
  };
  rec->RecordGame(/*thread_id=*/1, game::Board{}, MakePassGame(2), infos1);

  rec->Flush();
  auto examples = ReadExamples(dir);

  REQUIRE(examples.size() == 2);
  CHECK(DecodePi(examples[0])[0] == doctest::Approx(1.0f));
  CHECK(DecodePi(examples[1])[1] == doctest::Approx(1.0f));

  fs::remove_all(tmpdir);
}

TEST_CASE("TfRecorder: color field alternates with move color") {
  // The "color" feature stores the moving player's color as a raw byte.
  char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
  REQUIRE(mkdtemp(tmpdir) != nullptr);
  std::string dir = std::string(tmpdir) + "/";

  // 2-move game: move 0 = BLACK, move 1 = WHITE.
  std::vector<MoveSearchRecord> move_infos = {
      SimpleRecord(MakeUniformPi(), /*trainable=*/true),
      SimpleRecord(MakeUniformPi(), /*trainable=*/true),
  };
  auto examples = RunAndRead(dir, MakePassGame(2), move_infos);
  REQUIRE(examples.size() == 2);

  auto decode_color = [](const tensorflow::Example& ex) -> game::Color {
    const std::string& bytes =
        ex.features().feature().at("color").bytes_list().value(0);
    game::Color c = 0;
    std::memcpy(&c, bytes.data(), sizeof(game::Color));
    return c;
  };

  CHECK(decode_color(examples[0]) == BLACK);
  CHECK(decode_color(examples[1]) == WHITE);

  fs::remove_all(tmpdir);
}

TEST_CASE("TfRecorder: only trainable moves produce examples") {
  SUBCASE("only middle move trainable") {
    char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
    REQUIRE(mkdtemp(tmpdir) != nullptr);

    std::vector<MoveSearchRecord> move_infos;
    for (int i = 0; i < 3; ++i) {
      move_infos.push_back(
          SimpleRecord(MakePiForMove(i), /*trainable=*/i == 1));
    }

    auto examples = RunAndRead(std::string(tmpdir) + "/", move_infos);
    REQUIRE(examples.size() == 1);
    auto pi = DecodePi(examples[0]);
    CHECK(pi[0] == doctest::Approx(0.0f));
    CHECK(pi[1] == doctest::Approx(1.0f));
    CHECK(pi[2] == doctest::Approx(0.0f));

    fs::remove_all(tmpdir);
  }

  SUBCASE("only first move trainable") {
    char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
    REQUIRE(mkdtemp(tmpdir) != nullptr);

    std::vector<MoveSearchRecord> move_infos;
    for (int i = 0; i < 3; ++i) {
      move_infos.push_back(
          SimpleRecord(MakePiForMove(i), /*trainable=*/i == 0));
    }

    auto examples = RunAndRead(std::string(tmpdir) + "/", move_infos);
    REQUIRE(examples.size() == 1);
    auto pi = DecodePi(examples[0]);
    CHECK(pi[0] == doctest::Approx(1.0f));
    CHECK(pi[1] == doctest::Approx(0.0f));

    fs::remove_all(tmpdir);
  }

  SUBCASE("only last move trainable") {
    char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
    REQUIRE(mkdtemp(tmpdir) != nullptr);

    std::vector<MoveSearchRecord> move_infos;
    for (int i = 0; i < 3; ++i) {
      move_infos.push_back(
          SimpleRecord(MakePiForMove(i), /*trainable=*/i == 2));
    }

    auto examples = RunAndRead(std::string(tmpdir) + "/", move_infos);
    REQUIRE(examples.size() == 1);
    auto pi = DecodePi(examples[0]);
    CHECK(pi[1] == doctest::Approx(0.0f));
    CHECK(pi[2] == doctest::Approx(1.0f));

    fs::remove_all(tmpdir);
  }

  SUBCASE("no moves trainable — no file written") {
    char tmpdir[] = "/tmp/tf_recorder_test_XXXXXX";
    REQUIRE(mkdtemp(tmpdir) != nullptr);

    std::vector<MoveSearchRecord> move_infos;
    for (int i = 0; i < 3; ++i) {
      move_infos.push_back(SimpleRecord(MakeUniformPi(), /*trainable=*/false));
    }

    auto examples = RunAndRead(std::string(tmpdir) + "/", move_infos);
    CHECK(examples.empty());

    fs::remove_all(tmpdir);
  }
}

}  // namespace
}  // namespace recorder
