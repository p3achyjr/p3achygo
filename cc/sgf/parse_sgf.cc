#include "cc/sgf/parse_sgf.h"

#include <iostream>

#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "cc/game/color.h"
#include "cc/game/loc.h"
#include "cc/sgf/sgf_tree.h"

#define RETURN_IF_NOT(check, msg)            \
  do {                                       \
    if (__builtin_expect(!(check), false)) { \
      return absl::InternalError(msg);       \
    }                                        \
  } while (false);

#define ASSIGN_OR_RETURN(var, statusor_expr)         \
  do {                                               \
    auto statusor = statusor_expr;                   \
    if (__builtin_expect(!(statusor.ok()), false)) { \
      return absl::Status(statusor.status());        \
    }                                                \
    var = statusor.value();                          \
  } while (false);

namespace sgf {
namespace {

using namespace ::game;

static constexpr char kCoords[] = "abcdefghijklmnopqrst";

class GameInfoExtractor : public SgfVisitor {
 public:
  GameInfoExtractor() = default;
  ~GameInfoExtractor() = default;

  // Disable Copy and Move.
  GameInfoExtractor(GameInfoExtractor const&) = delete;
  GameInfoExtractor& operator=(GameInfoExtractor const&) = delete;
  GameInfoExtractor(GameInfoExtractor&&) = delete;
  GameInfoExtractor& operator=(GameInfoExtractor&&) = delete;

  // sgf::SgfVisitor Impl.
  void Visit(const sgf::SgfSizeProp* property) override {
    game_info_.board_size = property->size();
  }

  void Visit(const sgf::SgfKomiProp* property) override {
    game_info_.komi = property->komi();
  }

  void Visit(const sgf::SgfResultProp* property) override {
    game_info_.result = property->result();
  }

  void Visit(const sgf::SgfHandicapProp* property) override {
    game_info_.handicap = property->handicap();
  }

  void Visit(const sgf::SgfCommentProp* property) override {}

  void Visit(const sgf::SgfBPlayerProp* property) override {
    game_info_.b_player_name = property->player();
  }

  void Visit(const sgf::SgfWPlayerProp* property) override {
    game_info_.w_player_name = property->player();
  }

  void Visit(const sgf::SgfBMoveProp* property) override {
    // Nodes off main variation are filtered in Visit(SgfNode*).
    game_info_.main_variation.emplace_back(Move{BLACK, property->move()});
  }

  void Visit(const sgf::SgfWMoveProp* property) override {
    game_info_.main_variation.emplace_back(Move{WHITE, property->move()});
  }

  void Visit(const sgf::SgfNode* node) override {
    for (const auto& prop : node->properties()) {
      prop->Accept(this);
    }

    // Only visit first child to get main sequence.
    if (node->children().size() == 0) return;
    node->child(0)->Accept(this);
  }

  GameInfo Extract(const sgf::SgfNode* node) {
    Visit(node);
    return game_info_;
  }

 private:
  GameInfo game_info_;
};

absl::StatusOr<int> ParseNodeSequence(int offset, const std::string& contents,
                                      SgfNode* node);
absl::StatusOr<int> ParseTree(int offset, const std::string& contents,
                              SgfNode* node);

int CoordToIndex(char coord) { return coord - 'a'; }

int SeekToNonWhitespace(int offset, const std::string& contents) {
  while (std::isspace(contents[offset])) {
    ++offset;
  }
  return offset;
}

absl::StatusOr<Game::Result> ParseResult(std::string res_str) {
  char color = std::toupper(res_str[0]);
  RETURN_IF_NOT((color == 'B' || color == 'W') && (res_str[1] == '+'),
                absl::StrFormat("Result string malformed: (%s)", res_str));

  Color winner = color == 'B' ? BLACK : WHITE;
  if (res_str[2] == 'R') {
    // Resigned game.
    return Game::Result{winner, 0, 0, true /* by_resign */, {}};
  }

  // Attempt to parse score.
  std::string score_str;
  int decimal_count = 0;
  int cursor = 2;
  while ((cursor < res_str.size()) &&
         (res_str[cursor] == '.' || std::isdigit(res_str[cursor]))) {
    char c = res_str[cursor];
    if (c == '.') ++decimal_count;

    score_str += c;
    ++cursor;
  }

  float margin;
  RETURN_IF_NOT(
      (decimal_count <= 1) && absl::SimpleAtof(score_str, &margin),
      absl::StrFormat("Result score margin malformed: (%s)", res_str));

  return winner == BLACK
             ? Game::Result{winner, margin, 0, false /* by_resign */, {}}
             : Game::Result{winner, 0, margin, false /* by_resign */, {}};
}

absl::StatusOr<Loc> ParseLoc(std::string loc_str) {
  if (loc_str.size() == 0) return kPassLoc;

  RETURN_IF_NOT(loc_str.size() == 2,
                absl::StrFormat("Loc string malformed: (%s)", loc_str));
  char ci = std::tolower(loc_str[0]);
  char cj = std::tolower(loc_str[1]);

  int i = CoordToIndex(ci);
  int j = CoordToIndex(cj);
  if (i == BOARD_LEN && j == BOARD_LEN) return kPassLoc;

  RETURN_IF_NOT(i >= 0 && i < BOARD_LEN,
                absl::StrFormat("Loc i outside bounds: (%s)", loc_str));
  RETURN_IF_NOT(j >= 0 && j < BOARD_LEN,
                absl::StrFormat("Loc j outside bounds: (%s)", loc_str));

  return Loc{i, j};
}

// Parse a property and modify `node` in place.
absl::StatusOr<int> ParseProp(int offset, const std::string& contents,
                              SgfNode* node) {
  std::string prop_name;
  offset = SeekToNonWhitespace(offset, contents);
  while (std::isupper(contents[offset])) {
    RETURN_IF_NOT(offset < contents.size(),
                  absl::StrFormat("Prop Name not Uppercase. (%d)", offset));
    prop_name += contents[offset];
    ++offset;
  }
  offset = SeekToNonWhitespace(offset, contents);

  RETURN_IF_NOT(
      contents[offset] == '[',
      absl::StrFormat("Prop Name not followed by value. (%d)", offset));
  std::string prop_value;
  ++offset;

  offset = SeekToNonWhitespace(offset, contents);
  while (contents[offset] != ']') {
    RETURN_IF_NOT(offset < contents.size(),
                  absl::StrFormat("Missing closing brace. (%d)", offset));
    prop_value += contents[offset];
    ++offset;
  }
  offset = SeekToNonWhitespace(offset, contents);

  RETURN_IF_NOT(contents[offset] == ']',
                absl::StrFormat("Missing closing brace. (%d)", offset));
  ++offset;
  // Consume the rest of the values, but do not add to value.
  while (true) {
    offset = SeekToNonWhitespace(offset, contents);
    if (contents[offset] != '[') break;
    while (contents[offset] != ']') {
      RETURN_IF_NOT(offset < contents.size(),
                    absl::StrFormat("Offset past end of string. (%d)", offset));
      ++offset;
    }
    ++offset;
    offset = SeekToNonWhitespace(offset, contents);
  }

  if (prop_name == SgfSizeProp::kTag) {
    int board_size;
    RETURN_IF_NOT(absl::SimpleAtoi(prop_value, &board_size),
                  absl::StrFormat("Invalid Board Size. (%d)", offset));
    node->AddProperty(std::make_unique<SgfSizeProp>(board_size));
  } else if (prop_name == SgfKomiProp::kTag) {
    float komi;
    RETURN_IF_NOT(absl::SimpleAtof(prop_value, &komi),
                  absl::StrFormat("Invalid Komi. (%d)", offset));
    node->AddProperty(std::make_unique<SgfKomiProp>(komi));
  } else if (prop_name == SgfResultProp::kTag) {
    Game::Result result;
    ASSIGN_OR_RETURN(result, ParseResult(prop_value));
    node->AddProperty(std::make_unique<SgfResultProp>(result));
  } else if (prop_name == SgfHandicapProp::kTag) {
    int handicap;
    RETURN_IF_NOT(absl::SimpleAtoi(prop_value, &handicap),
                  absl::StrFormat("Invalid Handicap. (%d)", offset));
    node->AddProperty(std::make_unique<SgfHandicapProp>(handicap));
  } else if (prop_name == SgfBPlayerProp::kTag) {
    node->AddProperty(std::make_unique<SgfBPlayerProp>(prop_value));
  } else if (prop_name == SgfWPlayerProp::kTag) {
    node->AddProperty(std::make_unique<SgfWPlayerProp>(prop_value));
  } else if (prop_name == SgfBMoveProp::kTag) {
    Loc loc;
    ASSIGN_OR_RETURN(loc, ParseLoc(prop_value));
    node->AddProperty(std::make_unique<SgfBMoveProp>(loc));
  } else if (prop_name == SgfWMoveProp::kTag) {
    Loc loc;
    ASSIGN_OR_RETURN(loc, ParseLoc(prop_value));
    node->AddProperty(std::make_unique<SgfWMoveProp>(loc));
  }

  return offset;
}

// Populates current sequence of nodes. We are guaranteed at least one node
// according to the SGF grammar.
absl::StatusOr<int> ParseNodeSequence(int offset, const std::string& contents,
                                      SgfNode* node) {
  RETURN_IF_NOT(
      contents[offset] == ';',
      absl::StrFormat("Bad Entry Offset in ParseNodeSequence: (%d), ( %c )",
                      offset, contents[offset]));
  ++offset;
  while (true) {
    RETURN_IF_NOT(
        offset < contents.size(),
        absl::StrFormat("Offset past end of SGF string: (%d)", offset));
    char c = contents[offset];

    if (std::isspace(c)) {
      ++offset;
      continue;
    } else if (std::isupper(c)) {
      ASSIGN_OR_RETURN(offset, ParseProp(offset, contents, node));
    } else if (c == ';') {
      std::unique_ptr<SgfNode> child = std::make_unique<SgfNode>();
      SgfNode* child_raw = child.get();
      node->AddChild(std::move(child));
      ASSIGN_OR_RETURN(offset, ParseNodeSequence(offset, contents, child_raw));
    } else if (c == '(') {
      std::unique_ptr<SgfNode> child = std::make_unique<SgfNode>();
      SgfNode* child_raw = child.get();
      node->AddChild(std::move(child));
      ASSIGN_OR_RETURN(offset, ParseTree(offset, contents, child_raw));

      // This node sequence is finished. We should return here.
      return offset;
    } else if (c == ')') {
      // This is possible if there are no subtrees beneath this sequence.
      return offset;
    } else {
      return absl::InternalError(absl::StrFormat(
          "Invalid Char in ParseNodeSequence context: (%d), ( %c )", offset,
          c));
    }
  }
}

// Populates subtree and returns offset into SGF contents.
absl::StatusOr<int> ParseTree(int offset, const std::string& contents,
                              SgfNode* node) {
  offset = SeekToNonWhitespace(offset, contents);
  RETURN_IF_NOT(contents[offset] == '(',
                absl::StrFormat("Bad Entry Offset in ParseTree: (%d), ( %c )",
                                offset, contents[offset]));
  ++offset;
  while (true) {
    RETURN_IF_NOT(
        offset < contents.size(),
        absl::StrFormat("Offset past end of SGF string: (%d)", offset));
    char c = contents[offset];

    if (c == ')') {
      return offset + 1;
    } else if (std::isspace(c)) {
      ++offset;
      continue;
    } else if (c == ';') {
      ASSIGN_OR_RETURN(offset, ParseNodeSequence(offset, contents, node));
    } else if (c == '(') {
      std::unique_ptr<SgfNode> child = std::make_unique<SgfNode>();
      SgfNode* child_raw = child.get();
      node->AddChild(std::move(child));
      ASSIGN_OR_RETURN(offset, ParseTree(offset, contents, child_raw));
    } else {
      return absl::InternalError(absl::StrFormat(
          "Invalid Char in ParseTree context: (%d), ( %c )", offset, c));
    }
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<sgf::SgfNode>> ParseSgf(std::string contents) {
  std::unique_ptr<sgf::SgfNode> sgf_root = std::make_unique<SgfNode>();
  absl::StatusOr<int> parse_status = ParseTree(0, contents, sgf_root.get());
  if (!parse_status.ok()) {
    return parse_status.status();
  }

  return std::move(sgf_root);
}

absl::StatusOr<std::unique_ptr<sgf::SgfNode>> ParseSgfFile(
    std::string sgf_filename) {
  FILE* const fp = fopen(sgf_filename.c_str(), "r");
  if (fp == nullptr) {
    return absl::InternalError("Could not open file " + sgf_filename);
  }
  std::string contents;
  while (!feof(fp)) {
    char buf[4096];
    size_t num_read = fread(buf, 1, 4096, fp);
    contents.append(std::string(buf, num_read));
  }
  fclose(fp);

  return ParseSgf(contents);
}

GameInfo ExtractGameInfo(sgf::SgfNode* root) {
  GameInfoExtractor extractor;
  return extractor.Extract(root);
}

}  // namespace sgf
