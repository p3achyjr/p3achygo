#include "cc/recorder/sgf_serializer.h"

#include "absl/strings/str_format.h"
#include "cc/game/game.h"

namespace recorder {
namespace {
using ::game::Game;
using ::game::Loc;

static constexpr char kCoords[] = "abcdefghijklmnopqrst";

std::string SgfString(const Game::Result& result) {
  switch (result.winner) {
    case BLACK:
      return absl::StrFormat("B+%g", result.bscore - result.wscore);
    case WHITE:
      return absl::StrFormat("W+%g", result.wscore - result.bscore);
    default:
      return "?";
  }
}

std::string SgfString(const Loc& loc) {
  if (loc == game::kPassLoc) {
    return "";
  }
  return absl::StrFormat("%c%c", kCoords[loc.i], kCoords[loc.j]);
}
}  // namespace

std::string SgfSerializer::Serialize(const SgfNode* node) {
  sgf_ = "(";
  VisitNode(node, true);
  sgf_ += ")\n";

  return sgf_;
}

void SgfSerializer::Visit(const SgfKomiProp* prop) {
  sgf_ += prop->tag() + "[" + absl::StrFormat("%g", prop->komi()) + "]";
}

void SgfSerializer::Visit(const SgfResultProp* prop) {
  sgf_ += prop->tag() + "[" + SgfString(prop->result()) + "]";
}

void SgfSerializer::Visit(const SgfBPlayerProp* prop) {
  sgf_ += prop->tag() + "[" + prop->player() + "]";
}

void SgfSerializer::Visit(const SgfWPlayerProp* prop) {
  sgf_ += prop->tag() + "[" + prop->player() + "]";
}

void SgfSerializer::Visit(const SgfBMoveProp* prop) {
  sgf_ += prop->tag() + "[" + SgfString(prop->move()) + "]";
}

void SgfSerializer::Visit(const SgfWMoveProp* prop) {
  sgf_ += prop->tag() + "[" + SgfString(prop->move()) + "]";
}

void SgfSerializer::Visit(const SgfNode* node) { VisitNode(node, false); }

void SgfSerializer::VisitNode(const SgfNode* node, bool is_root) {
  sgf_ += is_root ? ";FF[4]GM[1]" : ";";
  for (const auto& prop : node->properties()) {
    prop->Accept(this);
  }

  if (node->children().size() == 1) {
    node->child(0)->Accept(this);
  } else if (node->children().size() > 1) {
    for (const std::unique_ptr<SgfNode>& child : node->children()) {
      sgf_ += "(";
      child->Accept(this);
      sgf_ += ")";
    }
  }
}
}  // namespace recorder
