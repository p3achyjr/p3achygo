#include "cc/gtp/analyze.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "absl/strings/numbers.h"
#include "cc/constants/constants.h"
#include "cc/game/loc.h"
#include "cc/gtp/command.h"
#include "cc/gtp/parse.h"

namespace gtp {
namespace {

using ::mcts::TreeNode;

constexpr int kMaxPvDepth = 10;

// Kata output flags we accept and do not produce.
bool IsIgnoredKataFlag(const std::string& key) {
  return key == "ownershipStdev" || key == "movesOwnership" ||
         key == "movesOwnershipStdev" || key == "pvVisits" ||
         key == "pvEdgeVisits" || key == "rootInfo" || key == "noResultValue";
}

int MostVisitedChild(const TreeNode* node) {
  int best = -1;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (node->child_visits[a] > 0 &&
        (best == -1 || node->child_visits[a] > node->child_visits[best])) {
      best = a;
    }
  }
  return best;
}

int Scaled(float x) { return static_cast<int>(std::lround(x * 10000.0f)); }

}  // namespace

absl::StatusOr<AnalyzeArgs> ParseAnalyzeArgs(
    bool kata, const std::vector<std::string>& tokens) {
  AnalyzeArgs args;
  args.kata = kata;
  size_t i = 0;
  game::Color color;
  if (i < tokens.size() && ParseColor(tokens[i], &color)) {
    args.color = color;
    ++i;
  }
  if (i < tokens.size() && absl::SimpleAtoi(tokens[i], &args.interval_cs)) {
    ++i;
  }
  for (; i < tokens.size(); i += 2) {
    const std::string& key = tokens[i];
    if (key == "avoid" || key == "allow" || key == "focus") {
      return absl::InvalidArgumentError(key + " is not supported");
    }
    if (i + 1 >= tokens.size()) {
      return absl::InvalidArgumentError("missing value for " + key);
    }
    const std::string& val = tokens[i + 1];
    int n;
    if (key == "interval" && absl::SimpleAtoi(val, &n)) {
      args.interval_cs = n;
    } else if (key == "maxmoves" && absl::SimpleAtoi(val, &n)) {
      args.max_moves = n;
    } else if (key == "minmoves" && absl::SimpleAtoi(val, &n)) {
      // Ignored: only visited moves are reported.
    } else if (kata && key == "ownership" &&
               (val == "true" || val == "false")) {
      args.ownership = val == "true";
    } else if (kata && IsIgnoredKataFlag(key) &&
               (val == "true" || val == "false")) {
    } else {
      return absl::InvalidArgumentError("bad argument: " + key + " " + val);
    }
  }
  if (args.interval_cs <= 0) {
    return absl::InvalidArgumentError("interval must be positive");
  }
  return args;
}

std::string AnalyzeLine(const TreeNode* root, const AnalyzeArgs& args,
                        const std::vector<float>& ownership) {
  std::vector<int> actions;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (root->child_visits[a] > 0 && root->children[a] != nullptr) {
      actions.push_back(a);
    }
  }
  if (actions.empty()) return "";
  std::stable_sort(actions.begin(), actions.end(), [root](int x, int y) {
    return root->child_visits[x] > root->child_visits[y];
  });
  if (args.max_moves > 0 && actions.size() > args.max_moves) {
    actions.resize(args.max_moves);
  }

  std::stringstream ss;
  ss << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < actions.size(); ++i) {
    const int a = actions[i];
    const TreeNode* child = root->children[a];
    const float winrate = (mcts::QOutcome(root, a) + 1.0f) / 2.0f;
    const float lcb = mcts::OutcomeWinrateLcb(root, a);
    const float prior = root->move_probs[a];
    ss << (i ? " " : "") << "info move " << GtpValueString(game::AsLoc(a))
       << " visits " << root->child_visits[a];
    if (args.kata) {
      const float score = -mcts::Score(child);
      ss << " winrate " << winrate << " scoreMean " << score << " scoreLead "
         << score << " prior " << prior << " lcb " << lcb;
    } else {
      ss << " winrate " << Scaled(winrate) << " prior " << Scaled(prior)
         << " lcb " << Scaled(lcb);
    }
    ss << " order " << i << " pv " << GtpValueString(game::AsLoc(a));
    const TreeNode* node = child;
    for (int d = 1; d < kMaxPvDepth && node != nullptr; ++d) {
      const int b = MostVisitedChild(node);
      if (b == -1) break;
      ss << " " << GtpValueString(game::AsLoc(b));
      node = node->children[b];
    }
  }
  if (args.kata && args.ownership && !ownership.empty()) {
    ss << " ownership";
    for (const float o : ownership) ss << " " << o;
  }
  return ss.str();
}

}  // namespace gtp
