#include "cc/gtp/service.h"

#include <algorithm>
#include <array>
#include <sstream>

#include "cc/core/probability.h"
#include "cc/game/game.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/tree.h"
#include "cc/nn/nn_interface.h"

namespace gtp {
namespace {
using namespace ::core;
using namespace ::game;
using namespace ::nn;
using namespace ::mcts;

std::string GtpBoardString(const Board& board) {
  std::stringstream ss;
  auto is_star_point = [](int i, int j) {
    return (i == 3 || i == 9 || i == 15) && (j == 3 || j == 9 || j == 15);
  };

  for (auto i = 0; i < BOARD_LEN; i++) {
    int gtp_row = BOARD_LEN - i;
    if (gtp_row < 10)
      ss << gtp_row << "  ";
    else
      ss << gtp_row << " ";
    for (auto j = 0; j < BOARD_LEN; j++) {
      if (board.at(i, j) == EMPTY && is_star_point(i, j)) {
        ss << "+ ";
      } else if (board.at(i, j) == EMPTY) {
        ss << "⋅ ";
      } else if (board.at(i, j) == BLACK) {
        ss << "○ ";
      } else if (board.at(i, j) == WHITE) {
        ss << "● ";
      }
    }

    ss << "\n";
  }

  ss << "   "
     << "A B C D E F G H J K L M N O P Q R S T";

  return ss.str();
}

class ServiceImpl final : public Service {
 public:
  ServiceImpl(std::unique_ptr<NNInterface> nn_interface,
              std::unique_ptr<GumbelEvaluator> gumbel_evaluator);
  ~ServiceImpl() = default;

  Response<int> GtpProtocolVersion(std::optional<int> id) override;
  Response<std::string> GtpName(std::optional<int> id) override;
  Response<std::string> GtpVersion(std::optional<int> id) override;
  Response<bool> GtpKnownCommand(std::optional<int> id,
                                 std::string cmd_name) override;
  Response<std::string> GtpListCommands(std::optional<int> id) override;
  Response<> GtpBoardSize(std::optional<int> id, int board_size) override;
  Response<> GtpClearBoard(std::optional<int> id) override;
  Response<> GtpKomi(std::optional<int> id, float komi) override;
  Response<> GtpPlay(std::optional<int> id, Move move) override;
  Response<Loc> GtpGenMove(std::optional<int> id, Color color) override;
  Response<std::string> GtpPrintBoard(std::optional<int> id) override;

  // Private Commands.
  Response<std::string> GtpPlayDbg(std::optional<int> id,
                                   game::Move move) override;
  Response<std::string> GtpGenMoveDbg(std::optional<int> id,
                                      game::Color color) override;

 private:
  GumbelResult GenMoveCommon(game::Color color);
  std::vector<std::pair<game::Loc, float>> GetTopPolicyMoves(
      const std::array<float, constants::kMaxNumMoves>& pi);
  void MakeMove(Color color, Loc loc);

  Probability probability_;
  std::unique_ptr<NNInterface> nn_interface_;
  std::unique_ptr<GumbelEvaluator> gumbel_evaluator_;
  std::unique_ptr<Game> game_;
  std::unique_ptr<TreeNode> current_root_;
  Color current_color_;
};

ServiceImpl::ServiceImpl(std::unique_ptr<NNInterface> nn_interface,
                         std::unique_ptr<GumbelEvaluator> gumbel_evaluator)
    : nn_interface_(std::move(nn_interface)),
      gumbel_evaluator_(std::move(gumbel_evaluator)),
      game_(std::make_unique<Game>()),
      current_root_(std::make_unique<TreeNode>()) {}

Response<int> ServiceImpl::GtpProtocolVersion(std::optional<int> id) {
  return MakeResponse(id, 2);
}

Response<std::string> ServiceImpl::GtpName(std::optional<int> id) {
  return MakeResponse(id, std::string("p3achygo"));
}

Response<std::string> ServiceImpl::GtpVersion(std::optional<int> id) {
  return MakeResponse(id, std::string(""));
}

Response<bool> ServiceImpl::GtpKnownCommand(std::optional<int> id,
                                            std::string cmd_name) {
  if (StringToGTPCode(cmd_name) != GTPCode::kUnknownCommand) {
    return MakeResponse(id, true);
  }

  return MakeResponse(id, false);
}

Response<std::string> ServiceImpl::GtpListCommands(std::optional<int> id) {
  std::string ls_str;
  for (const auto& code : kSupportedCommands) {
    ls_str += (GTPCodeToString(code) + "\n");
  }

  return MakeResponse(id, ls_str);
}

Response<> ServiceImpl::GtpBoardSize(std::optional<int> id, int board_size) {
  if (board_size != BOARD_LEN) {
    return MakeErrorResponse(id, "unacceptable size");
  }

  return MakeResponse(id);
}

Response<> ServiceImpl::GtpClearBoard(std::optional<int> id) {
  game_ = std::make_unique<Game>();
  current_root_ = std::make_unique<TreeNode>();

  return MakeResponse(id);
}

Response<> ServiceImpl::GtpKomi(std::optional<int> id, float komi) {
  return MakeResponse(id);
}

Response<> ServiceImpl::GtpPlay(std::optional<int> id, Move move) {
  if (!game_->IsValidMove(move.loc, move.color)) {
    return MakeErrorResponse(id, "illegal move");
  }

  MakeMove(move.color, move.loc);
  return MakeResponse(id);
}

Response<Loc> ServiceImpl::GtpGenMove(std::optional<int> id, Color color) {
  GumbelResult search_result = GenMoveCommon(color);
  MakeMove(color, search_result.mcts_move);
  return MakeResponse(id, search_result.mcts_move);
}

Response<std::string> ServiceImpl::GtpPrintBoard(std::optional<int> id) {
  std::stringstream ss;
  ss << "\n" << GtpBoardString(game_->board());

  return MakeResponse(id, ss.str());
}

Response<std::string> ServiceImpl::GtpPlayDbg(std::optional<int> id,
                                              game::Move move) {
  if (!game_->IsValidMove(move.loc, move.color)) {
    return MakeErrorResponse<std::string>(id, "illegal move");
  }

  std::string dbg_str;
  dbg_str += "---- Move Num " + std::to_string(game_->num_moves()) + " ----\n";
  dbg_str += "\nTree Stats:\n  N: " +
             std::to_string(static_cast<int>(N(current_root_.get()))) +
             "\n  V: " + std::to_string(V(current_root_.get())) +
             "\n  Score: " + std::to_string(current_root_->score_est) + "\n";

  MakeMove(move.color, move.loc);
  dbg_str += "\n" + GtpBoardString(game_->board());

  return MakeResponse(id, dbg_str);
}

Response<std::string> ServiceImpl::GtpGenMoveDbg(std::optional<int> id,
                                                 game::Color color) {
  GumbelResult search_result = GenMoveCommon(color);
  std::vector<std::pair<game::Loc, float>> top_policy_moves =
      current_root_ ? GetTopPolicyMoves(current_root_->move_probs)
                    : std::vector<std::pair<game::Loc, float>>();
  std::vector<std::pair<game::Loc, float>> top_policy_moves_improved =
      GetTopPolicyMoves(search_result.pi_improved);
  MakeMove(color, search_result.mcts_move);

  // Run this to populate the policy logits at the new root.
  gumbel_evaluator_->SearchRoot(probability_, *game_, current_root_.get(),
                                game::OppositeColor(color), 1, 1);
  std::vector<std::pair<game::Loc, float>> top_policy_moves_next =
      current_root_ ? GetTopPolicyMoves(current_root_->move_probs)
                    : std::vector<std::pair<game::Loc, float>>();

  std::string dbg_str;
  dbg_str += "---- Move Num " + std::to_string(game_->num_moves()) + " ----\n";
  dbg_str +=
      "\nTree Stats:\n  N: " +
      std::to_string(static_cast<int>(N(current_root_.get()))) +
      "\n  V: " + std::to_string(V(current_root_.get())) + "\n  Score: " +
      std::to_string(current_root_ ? current_root_->score_est : 0.0f) + "\n";

  dbg_str += "Top Policy Moves:\n";
  for (const auto& move : top_policy_moves) {
    dbg_str +=
        GtpValueString(move.first) + ": " + std::to_string(move.second) + "\n";
  }

  dbg_str += "Completed Q:\n";
  for (const auto& move : top_policy_moves_improved) {
    dbg_str +=
        GtpValueString(move.first) + ": " + std::to_string(move.second) + "\n";
  }

  dbg_str += "Gumbel Selected Moves:\n";
  for (const auto& selected_child : search_result.child_stats) {
    dbg_str += GtpValueString(selected_child.move) +
               ", n: " + std::to_string(selected_child.n) +
               ", p: " + std::to_string(selected_child.prob) +
               ", q: " + std::to_string(selected_child.q) +
               ", qz: " + std::to_string(selected_child.qz) + "\n";
  }

  dbg_str += "Top Policy Moves Next:\n";
  for (const auto& move : top_policy_moves_next) {
    dbg_str +=
        GtpValueString(move.first) + ": " + std::to_string(move.second) + "\n";
  }

  dbg_str += "\n" + GtpBoardString(game_->board());
  return MakeResponse(id, dbg_str);
}

GumbelResult ServiceImpl::GenMoveCommon(Color color) {
  static constexpr int kDefaultN = 192;
  static constexpr int kDefaultK = 8;
  if (!(color == current_color_)) {
    // We are moving twice consecutively. Fake a pass first.
    current_color_ = color;
    current_root_ = std::move(current_root_->children[kPassLoc]);
    if (!current_root_) {
      current_root_ = std::make_unique<TreeNode>();
    }
  }

  GumbelResult search_result = gumbel_evaluator_->SearchRoot(
      probability_, *game_, current_root_.get(), color, kDefaultN, kDefaultK);
  return search_result;
}

std::vector<std::pair<game::Loc, float>> ServiceImpl::GetTopPolicyMoves(
    const std::array<float, constants::kMaxNumMoves>& pi) {
  static constexpr int kNumTopMoves = 8;
  std::vector<std::pair<game::Loc, float>> top_policy_moves;
  for (int i = 0; i < constants::kMaxNumMoves; ++i) {
    top_policy_moves.emplace_back(std::make_pair(AsLoc(i), pi[i]));
  }
  std::sort(
      top_policy_moves.begin(), top_policy_moves.end(),
      [](const std::pair<game::Loc, float>& x,
         const std::pair<game::Loc, float>& y) { return x.second > y.second; });

  return std::vector(top_policy_moves.begin(),
                     top_policy_moves.begin() + kNumTopMoves);
}

void ServiceImpl::MakeMove(Color color, Loc loc) {
  game_->PlayMove(loc, color);
  current_color_ = game::OppositeColor(color);
  current_root_ = std::move(current_root_->children[loc]);
  if (!current_root_) {
    current_root_ = std::make_unique<TreeNode>();
  }
}
}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<Service>> Service::CreateService(
    std::string model_path) {
  std::unique_ptr<NNInterface> nn_interface =
      std::make_unique<NNInterface>(1 /* num_threads */);
  if (!nn_interface->Initialize(std::move(model_path)).ok()) {
    return absl::InternalError("Could not initialize neural network.");
  }
  std::unique_ptr<GumbelEvaluator> gumbel_evaluator =
      std::make_unique<GumbelEvaluator>(nn_interface.get(), 0);

  return std::make_unique<ServiceImpl>(std::move(nn_interface),
                                       std::move(gumbel_evaluator));
}

}  // namespace gtp