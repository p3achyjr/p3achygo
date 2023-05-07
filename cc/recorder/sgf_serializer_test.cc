#include "cc/recorder/sgf_serializer.h"

#include "cc/core/doctest_include.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"

namespace recorder {
using ::game::Game;
using ::game::Loc;
using ::game::Move;

static constexpr char kTestPB[] = "testB";
static constexpr char kTestPW[] = "testW";

game::Game GameFromMoves(const std::vector<Move>& moves) {
  game::Game game;
  for (const auto& move : moves) {
    game.PlayMove(move.loc, move.color);
  }
  return game;
}

std::unique_ptr<SgfNode> ToSgfNode(const Game& game) {
  std::unique_ptr<SgfNode> root_node = std::make_unique<SgfNode>();
  root_node->AddProperty(std::make_unique<SgfKomiProp>(game.komi()));
  root_node->AddProperty(std::make_unique<SgfResultProp>(game.result()));
  root_node->AddProperty(std::make_unique<SgfBPlayerProp>(kTestPB));
  root_node->AddProperty(std::make_unique<SgfWPlayerProp>(kTestPW));

  int num_moves = game.move_num();
  SgfNode* current_node = root_node.get();
  for (int i = 0; i < num_moves; ++i) {
    const Move& move = game.move(i);
    std::unique_ptr<SgfNode> child = std::make_unique<SgfNode>();
    if (move.color == BLACK) {
      child->AddProperty(std::make_unique<SgfBMoveProp>(move.loc));
    } else if (move.color == WHITE) {
      child->AddProperty(std::make_unique<SgfWMoveProp>(move.loc));
    }

    SgfNode* tmp = child.get();
    current_node->AddChild(std::move(child));
    current_node = tmp;
  }

  return root_node;
}

TEST_CASE("SgfSerializer") {
  SUBCASE("Empty") {
    SgfSerializer serializer;
    game::Game game = GameFromMoves({});

    CHECK_EQ(serializer.Serialize(ToSgfNode(game).get()),
             "(;FF[4]GM[1]KM[7.5]RE[?]PB[testB]PW[testW])\n");
  }

  SUBCASE("EmptyResult") {
    SgfSerializer serializer;
    game::Game game = GameFromMoves({});
    game.WriteResult();

    CHECK_EQ(serializer.Serialize(ToSgfNode(game).get()),
             "(;FF[4]GM[1]KM[7.5]RE[W+7.5]PB[testB]PW[testW])\n");
  }

  SUBCASE("Moves") {
    SgfSerializer serializer;
    game::Game game =
        GameFromMoves({Move{BLACK, Loc{0, 0}}, Move{WHITE, Loc{1, 0}},
                       Move{BLACK, Loc{0, 1}}, Move{WHITE, Loc{1, 1}}});

    CHECK_EQ(serializer.Serialize(ToSgfNode(game).get()),
             "(;FF[4]GM[1]KM[7.5]RE[?]PB[testB]PW[testW];B[aa];W[ba];B[ab]"
             ";W[bb])\n");
  }

  // . . o x . x
  // . . o x . x
  // o o o . x .
  SUBCASE("WithResult") {
    SgfSerializer serializer;
    game::Game game = GameFromMoves(
        {Move{BLACK, Loc{0, 2}}, Move{WHITE, Loc{0, 3}}, Move{BLACK, Loc{1, 2}},
         Move{WHITE, Loc{1, 3}}, Move{BLACK, Loc{2, 2}}, Move{WHITE, Loc{2, 4}},
         Move{BLACK, Loc{2, 0}}, Move{WHITE, Loc{1, 5}}, Move{BLACK, Loc{2, 1}},
         Move{WHITE, Loc{0, 5}}});
    game.WriteResult();

    CHECK_EQ(serializer.Serialize(ToSgfNode(game).get()),
             "(;FF[4]GM[1]KM[7.5]RE[W+5.5]PB[testB]PW[testW];B[ac];W[ad];B[bc]"
             ";W[bd];B[cc];W[ce];B[ca];W[bf];B[cb];W[af])\n");
  }

  SUBCASE("Branch") {
    SgfSerializer serializer;
    std::unique_ptr<SgfNode> root = std::make_unique<SgfNode>();
    std::unique_ptr<SgfNode> nodeA = std::make_unique<SgfNode>();
    std::unique_ptr<SgfNode> nodeB = std::make_unique<SgfNode>();
    std::unique_ptr<SgfNode> nodeC = std::make_unique<SgfNode>();
    std::unique_ptr<SgfNode> nodeD = std::make_unique<SgfNode>();

    nodeA->AddProperty(std::make_unique<SgfBMoveProp>(Loc{0, 0}));
    nodeB->AddProperty(std::make_unique<SgfBMoveProp>(Loc{0, 1}));
    nodeC->AddProperty(std::make_unique<SgfWMoveProp>(Loc{0, 2}));
    nodeD->AddProperty(std::make_unique<SgfWMoveProp>(Loc{0, 3}));

    nodeB->AddChild(std::move(nodeC));
    nodeB->AddChild(std::move(nodeD));
    root->AddChild(std::move(nodeA));
    root->AddChild(std::move(nodeB));

    CHECK_EQ(serializer.Serialize(root.get()),
             "(;FF[4]GM[1](;B[aa])(;B[ab](;W[ac])(;W[ad])))\n");
  }
}
}  // namespace recorder
