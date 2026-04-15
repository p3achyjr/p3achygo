#include "cc/sgf/parse_sgf.h"

#include "absl/status/statusor.h"
#include "cc/core/doctest_include.h"
#include "cc/game/move.h"
#include "cc/sgf/parse_sgf_test_data.h"
#include "cc/sgf/sgf_serializer.h"

namespace sgf {
namespace {
using namespace ::game;

bool operator==(const GameInfo& lhs, const GameInfo& rhs) {
  Game::Result lresult = lhs.result;
  Game::Result rresult = rhs.result;
  bool bscore_eq = lresult.bscore == doctest::Approx(rresult.bscore);
  bool wscore_eq = lresult.wscore == doctest::Approx(rresult.wscore);
  bool res_eq = lresult.winner == rresult.winner && bscore_eq && wscore_eq &&
                lresult.by_resign == rresult.by_resign;
  return lhs.board_size == rhs.board_size &&
         lhs.b_player_name == rhs.b_player_name && res_eq &&
         lhs.handicap == rhs.handicap &&
         lhs.main_variation == rhs.main_variation;
}

}  // namespace

static constexpr char kSmallSgf[] =
    "(;FF[4]GM[1]KM[7.5]RE[W+3.5]PB[player1]PW[player2];B[aa];W["
    "rs];B[tt];W[])";

static constexpr char kSmallSgfWhitespace[] =
    "(;FF[4]\nGM[1]\nKM[7.5]  RE[W+3.5]\rPB[player1]PW[player2];  \nB[aa]  \r "
    "\n;W["
    "rs];B[tt];W[])";

TEST_CASE("SgfSmall") {
  absl::StatusOr<std::unique_ptr<SgfNode>> tree = ParseSgf(kSmallSgf);
  CHECK(tree.ok());

  GameInfo info = ExtractGameInfo(tree->get());
  CHECK(info.komi == doctest::Approx(7.5));
  CHECK(info.result.winner == WHITE);
  CHECK(info.result.wscore - info.result.bscore == doctest::Approx(3.5));
  CHECK(info.b_player_name == "player1");
  CHECK(info.w_player_name == "player2");

  const auto& main_line = info.main_variation;
  CHECK_EQ(main_line[0], Move{BLACK, Loc{0, 0}});
  CHECK_EQ(main_line[1], Move{WHITE, Loc{17, 18}});
  CHECK_EQ(main_line[2], Move{BLACK, kPassLoc});
  CHECK_EQ(main_line[3], Move{WHITE, kPassLoc});
}

TEST_CASE("SgfSmallWhitespace") {
  absl::StatusOr<std::unique_ptr<SgfNode>> tree = ParseSgf(kSmallSgfWhitespace);
  CHECK(tree.ok());

  GameInfo info = ExtractGameInfo(tree->get());
  CHECK(info.komi == doctest::Approx(7.5));
  CHECK(info.result.winner == WHITE);
  CHECK(info.result.wscore - info.result.bscore == doctest::Approx(3.5));
  CHECK(info.b_player_name == "player1");
  CHECK(info.w_player_name == "player2");

  const auto& main_line = info.main_variation;
  CHECK_EQ(main_line[0], Move{BLACK, Loc{0, 0}});
  CHECK_EQ(main_line[1], Move{WHITE, Loc{17, 18}});
  CHECK_EQ(main_line[2], Move{BLACK, kPassLoc});
  CHECK_EQ(main_line[3], Move{WHITE, kPassLoc});
}

TEST_CASE("SgfFF4") {
  // Test that after serializing and parsing, we get identical games.
  absl::StatusOr<std::unique_ptr<SgfNode>> tree = ParseSgf(kSgfFF4);
  CHECK(tree.ok());

  GameInfo info_parse = ExtractGameInfo(tree->get());
  SgfSerializer serializer;
  std::string sgf_reserialized = serializer.Serialize(tree->get());
  absl::StatusOr<std::unique_ptr<SgfNode>> tree_reparsed =
      ParseSgf(sgf_reserialized);
  CHECK(tree_reparsed.ok());

  GameInfo info_reparsed = ExtractGameInfo(tree_reparsed->get());

  CHECK(info_parse == info_reparsed);
}

TEST_CASE("SgfFF3") {
  // Test that after serializing and parsing, we get identical games.
  absl::StatusOr<std::unique_ptr<SgfNode>> tree = ParseSgf(kSgfFF3);
  CHECK(tree.ok());

  GameInfo info_parse = ExtractGameInfo(tree->get());
  SgfSerializer serializer;
  std::string sgf_reserialized = serializer.Serialize(tree->get());
  absl::StatusOr<std::unique_ptr<SgfNode>> tree_reparsed =
      ParseSgf(sgf_reserialized);
  CHECK(tree_reparsed.ok());

  GameInfo info_reparsed = ExtractGameInfo(tree_reparsed->get());

  CHECK(info_parse == info_reparsed);
}

TEST_CASE("SgfHandicap") {
  // Test that after serializing and parsing, we get identical games.
  absl::StatusOr<std::unique_ptr<SgfNode>> tree = ParseSgf(kSgfHandicap);
  CHECK(tree.ok());

  GameInfo info_parse = ExtractGameInfo(tree->get());
  CHECK(info_parse.handicap == 2);
}

TEST_CASE("SgfNonStandardSize") {
  // Test that after serializing and parsing, we get identical games.
  absl::StatusOr<std::unique_ptr<SgfNode>> tree = ParseSgf(kSgfNonStandardSize);
  CHECK(tree.ok());

  GameInfo info_parse = ExtractGameInfo(tree->get());
  CHECK(info_parse.board_size == 13);
}

}  // namespace sgf
