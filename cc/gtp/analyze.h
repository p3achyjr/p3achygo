#ifndef GTP_ANALYZE_H_
#define GTP_ANALYZE_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "cc/game/color.h"
#include "cc/mcts/tree.h"

namespace gtp {

// `lz-analyze` / `kata-analyze` arguments (KataGo GTP extensions):
//   [player] [interval] [interval N] [maxmoves N] [minmoves N]
//   [ownership true|false] [<kata output flag> true|false ...]
// The interval is in centiseconds.
struct AnalyzeArgs {
  bool kata = false;  // kata-analyze output format, else lz-analyze.
  std::optional<game::Color> color;
  int interval_cs = 100;
  int max_moves = 0;  // 0 = every visited move.
  bool ownership = false;
};

// Parses the argument tokens after the command name. Rejects `avoid`, `allow`
// and `focus`, which the engine does not support. Accepts and ignores
// `minmoves` and kata output flags it does not produce (e.g. `pvVisits`).
absl::StatusOr<AnalyzeArgs> ParseAnalyzeArgs(
    bool kata, const std::vector<std::string>& tokens);

// One analysis line: an `info move ...` entry per visited root child, most
// visited first, all on one line with no trailing newline; with
// `args.kata && args.ownership`, followed by ` ownership` and `ownership`'s
// values. Values are from the side to move's view. lz-analyze scales
// winrate / prior / lcb to integers in [0, 10000]; kata-analyze prints them in
// [0, 1] and adds scoreMean / scoreLead (searched mean score of the move).
// Empty if no child is visited.
std::string AnalyzeLine(const mcts::TreeNode* root, const AnalyzeArgs& args,
                        const std::vector<float>& ownership);

}  // namespace gtp

#endif
