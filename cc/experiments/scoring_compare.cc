// Compare default (Benson dead-stone removal) scoring against pure
// Tromp-Taylor from a real game position.
//
// CGOS scores Tromp-Taylor. The engine scores with Benson removal: a stone in
// the opponent's pass-alive region is treated as dead and not counted, and such
// stones inside your own region are counted as your points. Those disagree
// exactly when a region is pass-alive but the opponent keeps playing in it --
// which is how a game the engine read as +5 finished W+17.5.
//
// For each scoring mode this loads an SGF, truncates to --move_num, runs a PUCT
// search, and reports the move chosen plus the root's evaluation. It also
// prints what each scoring method says the position is worth statically, so the
// disagreement is visible on its own.

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cc/game/board.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
#include "cc/mcts/tree.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"
#include "cc/sgf/parse_sgf.h"

ABSL_FLAG(std::string, sgf_path, "", "SGF to replay.");
ABSL_FLAG(std::string, model_path, "", "TRT engine path.");
ABSL_FLAG(int, move_num, 400, "Replay this many moves, then search.");
ABSL_FLAG(int, visits, 1600, "Search visits per mode.");
ABSL_FLAG(float, komi, 7.5f, "Komi.");

namespace {

using ::game::Board;
using ::game::Color;
using ::game::Game;
using ::mcts::GumbelEvaluator;
using ::mcts::MctsNodeTable;
using ::mcts::PuctParams;
using ::mcts::PuctRootSelectionPolicy;
using ::mcts::TreeNode;
using ::nn::NNInterface;

std::string LocStr(game::Loc loc) {
  if (loc == game::kPassLoc) return "pass";
  if (loc == game::kNoopLoc) return "noop";
  static constexpr char kCols[] = "ABCDEFGHJKLMNOPQRST";
  return std::string(1, kCols[loc.j]) + std::to_string(BOARD_LEN - loc.i);
}

struct Outcome {
  game::Loc move;
  float winrate;
  float root_score;
  float bscore, wscore;
  std::array<Color, BOARD_LEN * BOARD_LEN> ownership;
  std::string board_str;
};

template <typename MoveSeq>
Outcome RunOne(NNInterface* nn, const MoveSeq& moves, int move_num,
               Board::Scoring scoring, float komi, int visits) {
  // prohibit_pass_alive=false matches the GTP service, which is what plays on
  // CGOS. Moves inside pass-alive regions are already legal there.
  Game g(/*prohibit_pass_alive=*/false, scoring);
  g.SetKomi(komi);
  for (int i = 0; i < move_num && i < (int)moves.size(); ++i) {
    g.PlayMove(moves[i].loc, moves[i].color);
  }
  const Color to_move = move_num % 2 == 0 ? BLACK : WHITE;

  MctsNodeTable table;
  TreeNode* root = table.GetOrCreate(g.board().hash(), to_move, false);
  core::Probability prob(1234);
  GumbelEvaluator ev(nn, 0);
  auto res =
      ev.SearchRootPuct(prob, g, &table, root, to_move, visits,
                        PuctParams::Builder()
                            .set_kind(PuctRootSelectionPolicy::kVisitCount)
                            .set_root_fpu(0.1f)
                            .build());

  game::Scores sc = g.GetScores();
  return Outcome{res.mcts_move,
                 (mcts::VOutcome(root) + 1.0f) / 2.0f,
                 mcts::Score(root),
                 sc.black_score,
                 sc.white_score,
                 sc.ownership,
                 game::ToString(g.board().position())};
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  const std::string sgf_path = absl::GetFlag(FLAGS_sgf_path);
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const int move_num = absl::GetFlag(FLAGS_move_num);
  const int visits = absl::GetFlag(FLAGS_visits);
  const float komi = absl::GetFlag(FLAGS_komi);
  CHECK(!sgf_path.empty()) << "--sgf_path required";
  CHECK(!model_path.empty()) << "--model_path required";

  auto sgf = sgf::ParseSgfFile(sgf_path);
  CHECK(sgf.ok()) << sgf.status();
  sgf::GameInfo info = sgf::ExtractGameInfo(sgf->get());
  const auto& moves = info.main_variation;
  LOG(INFO) << "loaded " << moves.size() << " moves from " << sgf_path;

  auto engine = nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                                 1, nn::GetVersionFromModelPath(model_path));
  auto nn = std::make_unique<NNInterface>(1, 0, 16384, std::move(engine));

  printf("\nposition: %s @ move %d   (%s to play)   komi %.1f   visits %d\n",
         sgf_path.c_str(), move_num, move_num % 2 == 0 ? "B" : "W", komi,
         visits);
  printf("%-14s %8s %10s %11s %12s %12s\n", "scoring", "move", "winrate",
         "root_score", "static_B", "static_W");
  for (auto [name, mode] :
       {std::pair{"default", Board::Scoring::kDefault},
        std::pair{"tromp_taylor", Board::Scoring::kTrompTaylor}}) {
    Outcome o = RunOne(nn.get(), moves, move_num, mode, komi, visits);
    printf("%-14s %8s %10.4f %11.2f %12.1f %12.1f\n", name,
           LocStr(o.move).c_str(), o.winrate, o.root_score, o.bscore, o.wscore);
  }
  // Board, and where the two methods disagree about who owns what.
  Outcome d =
      RunOne(nn.get(), moves, move_num, Board::Scoring::kDefault, komi, visits);
  Outcome t = RunOne(nn.get(), moves, move_num, Board::Scoring::kTrompTaylor,
                     komi, visits);
  // Print with GTP coordinates so the board and the move labels agree.
  // game::ToString() uses 0-indexed rows and columns A-S *including* I, which
  // does not match LocStr()'s GTP convention.
  {
    Game g(/*prohibit_pass_alive=*/false, Board::Scoring::kDefault);
    g.SetKomi(komi);
    for (int i = 0; i < move_num && i < (int)moves.size(); ++i) {
      g.PlayMove(moves[i].loc, moves[i].color);
    }
    static constexpr char kCols[] = "ABCDEFGHJKLMNOPQRST";
    printf("\nBoard at move %d (%s to play), GTP coordinates:\n", move_num,
           move_num % 2 == 0 ? "B" : "W");
    printf("    ");
    for (int j = 0; j < BOARD_LEN; ++j) printf("%c ", kCols[j]);
    printf("\n");
    for (int i = 0; i < BOARD_LEN; ++i) {
      printf(" %2d ", BOARD_LEN - i);
      for (int j = 0; j < BOARD_LEN; ++j) {
        const int c = g.board().at(i, j);
        printf("%s ", c == BLACK ? "X" : c == WHITE ? "O" : ".");
      }
      printf("%-2d\n", BOARD_LEN - i);
    }
    printf("    ");
    for (int j = 0; j < BOARD_LEN; ++j) printf("%c ", kCols[j]);
    printf("\n     (X = black, O = white)\n");
  }
  printf("default plays      : %s\n", LocStr(d.move).c_str());
  printf("tromp_taylor plays : %s\n\n", LocStr(t.move).c_str());

  // Ownership disagreement map. '.' = agree, 'b'/'w' = default claims it for
  // that color while Tromp-Taylor does not.
  int only_default_b = 0, only_default_w = 0;
  printf(
      "Ownership disagreement (default claims, Tromp-Taylor does not),\n"
      "same GTP coordinates:\n");
  {
    static constexpr char kCols[] = "ABCDEFGHJKLMNOPQRST";
    printf("    ");
    for (int j = 0; j < BOARD_LEN; ++j) printf("%c ", kCols[j]);
    printf("\n");
  }
  for (int i = 0; i < BOARD_LEN; ++i) {
    printf(" %2d ", BOARD_LEN - i);
    for (int j = 0; j < BOARD_LEN; ++j) {
      const int k = i * BOARD_LEN + j;
      char c = '.';
      if (d.ownership[k] != t.ownership[k]) {
        if (d.ownership[k] == BLACK) {
          c = 'b';
          ++only_default_b;
        } else if (d.ownership[k] == WHITE) {
          c = 'w';
          ++only_default_w;
        } else
          c = '-';
      }
      printf("%c ", c);
    }
    printf("\n");
  }
  printf("\n  points default gives B that TT does not: %d\n", only_default_b);
  printf("  points default gives W that TT does not: %d\n", only_default_w);
  printf(
      "\nstatic_B/static_W are what each method scores the *current* board;\n"
      "B - W - komi is the margin that method believes.\n");
  return 0;
}
