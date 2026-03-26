#include "cc/gtp/service.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "cc/analysis/analysis.h"
#include "cc/core/probability.h"
#include "cc/eval/player_config.h"
#include "cc/game/game.h"
#include "cc/mcts/bias_cache.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search_policy.h"
#include "cc/mcts/tree.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"
#include "cc/recorder/sgf_recorder.h"
#include "cc/sgf/parse_sgf.h"
#include "cc/sgf/sgf_tree.h"

namespace gtp {
namespace {
using namespace ::core;
using namespace ::game;
using namespace ::nn;
using namespace ::mcts;
using namespace ::recorder;
using namespace ::sgf;

std::string GtpBoardString(const Board& board) {
  static constexpr char kColIndices[] =
      "   A B C D E F G H J K L M N O P Q R S T";
  std::stringstream ss;
  auto is_star_point = [](int i, int j) {
    return (i == 3 || i == 9 || i == 15) && (j == 3 || j == 9 || j == 15);
  };

  ss << kColIndices << "\n";
  for (auto i = 0; i < BOARD_LEN; i++) {
    int gtp_row = BOARD_LEN - i;
    if (gtp_row < 10) {
      ss << gtp_row << "  ";
    } else {
      ss << gtp_row << " ";
    }

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

    if (gtp_row < 10) {
      ss << " " << gtp_row;
    } else {
      ss << gtp_row;
    }

    ss << "\n";
  }

  ss << kColIndices;

  return ss.str();
}

mcts::ScoreUtilityParams MakeScoreUtilityParams(
    const eval::PlayerSearchConfig& cfg) {
  mcts::ScoreUtilityMode mode = cfg.score_utility_mode == "integral"
                                    ? mcts::ScoreUtilityMode::kIntegral
                                    : mcts::ScoreUtilityMode::kDirect;
  return mcts::ScoreUtilityParams{.score_weight = cfg.score_weight,
                                  .mode = mode};
}

mcts::PuctParams MakePuctParams(const eval::PlayerSearchConfig& cfg) {
  mcts::PuctRootSelectionPolicy policy;
  if (cfg.puct_root_policy == "lcb") {
    policy = mcts::PuctRootSelectionPolicy::kLcb;
  } else if (cfg.puct_root_policy == "visit_count_sample") {
    policy = mcts::PuctRootSelectionPolicy::kVisitCountSample;
  } else if (cfg.puct_root_policy == "visit_count") {
    policy = mcts::PuctRootSelectionPolicy::kVisitCount;
  } else {
    policy = cfg.use_lcb ? mcts::PuctRootSelectionPolicy::kLcb
                         : mcts::PuctRootSelectionPolicy::kVisitCount;
  }
  return mcts::PuctParams::Builder()
      .set_kind(policy)
      .set_c_puct(cfg.c_puct)
      .set_c_puct_visit_scaling(cfg.c_puct_visit_scaling)
      .set_c_puct_v_2(cfg.c_puct_v_2)
      .set_use_puct_v(cfg.use_puct_v)
      .set_enable_var_scaling(cfg.var_scale_cpuct)
      .set_var_scale_prior_visits(cfg.var_scale_prior_visits)
      .set_tau(cfg.tau)
      .set_enable_m3_bonus(cfg.enable_m3_bonus)
      .set_m3_prior_visits(cfg.m3_prior_visits)
      .set_p_opt_weight(cfg.p_opt_weight)
      .build();
}

class ServiceImpl final : public Service {
 public:
  ServiceImpl(std::unique_ptr<NNInterface> nn_interface,
              eval::PlayerSearchConfig cfg, bool verbose);
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
  Response<game::Scores> GtpFinalScore(std::optional<int> id) override;
  Response<> GtpUndo(std::optional<int> id) override;
  Response<std::string> GtpExplainLastMove(std::optional<int> id) override;

  // Analysis methods. Call these from separate threads.
  Response<> GtpStartAnalysis(std::optional<int> id,
                              game::Color color) override;
  analysis::AnalysisSnapshot GtpAnalysisSnapshot(game::Color color) override;
  void GtpStopAnalysis() override;
  game::Loc GtpGenMoveAnalyze(game::Color color) override;

  // Private Commands.
  Response<std::string> GtpPlayDbg(std::optional<int> id,
                                   game::Move move) override;
  Response<std::string> GtpGenMoveDbg(std::optional<int> id,
                                      game::Color color) override;
  Response<std::array<float, BOARD_LEN * BOARD_LEN>> GtpOwnership(
      std::optional<int> id) override;
  Response<std::string> GtpSerializeSgfWithTrees(std::optional<int> id,
                                                 std::string filename) override;
  Response<> GtpLoadSgf(std::optional<int> id, std::string filename) override;

 private:
  GumbelResult GenMoveCommon(game::Color color);
  std::string BuildMoveDbgString(const GumbelResult& result, Color color);
  std::string BuildExplainComment();
  std::vector<std::pair<game::Loc, float>> GetTopPolicyMoves(
      const std::array<float, constants::kMaxMovesPerPosition>& pi);
  void ClearBoard();
  void MakeMove(Color color, Loc loc);
  inline TreeNode* current_root() const { return current_root_; }

  eval::PlayerSearchConfig cfg_;
  bool verbose_;
  float komi_ = 7.5f;
  Probability probability_;
  std::unique_ptr<NNInterface> nn_interface_;
  std::unique_ptr<GumbelEvaluator> gumbel_evaluator_;
  std::unique_ptr<Game> game_;
  std::unique_ptr<NodeTable> node_table_;
  std::optional<mcts::BiasCache> bias_cache_;
  TreeNode* current_root_;
  Color current_color_;
  std::string last_explain_comment_;

  std::atomic<bool> analysis_running_;
  std::thread analysis_thread_;
};

ServiceImpl::ServiceImpl(std::unique_ptr<NNInterface> nn_interface,
                         eval::PlayerSearchConfig cfg, bool verbose)
    : cfg_(cfg),
      verbose_(verbose),
      nn_interface_(std::move(nn_interface)),
      game_(std::make_unique<Game>(false)),
      node_table_(
          cfg.use_mcgs
              ? std::unique_ptr<NodeTable>(std::make_unique<McgsNodeTable>())
              : std::unique_ptr<NodeTable>(std::make_unique<MctsNodeTable>())),
      current_color_(BLACK),
      analysis_running_(false) {
  if (cfg_.use_bias_cache) {
    bias_cache_.emplace(cfg_.bias_cache_alpha, cfg_.bias_cache_lambda);
  }
  gumbel_evaluator_ = std::make_unique<GumbelEvaluator>(
      nn_interface_.get(), 0, MakeScoreUtilityParams(cfg_),
      bias_cache_ ? &*bias_cache_ : nullptr);
  current_root_ = node_table_->GetOrCreate(game_->board().hash(), BLACK, false);
}

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
  ClearBoard();
  return MakeResponse(id);
}

Response<> ServiceImpl::GtpKomi(std::optional<int> id, float komi) {
  komi_ = komi;
  game_->SetKomi(komi);
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
  if (V(current_root()) < -0.96f && game_->num_moves() > 50) {
    return MakeResponse(id, kNoopLoc);
  }

  last_explain_comment_ = BuildExplainComment();

  if (verbose_) {
    std::string dbg_str = BuildMoveDbgString(search_result, color);
    MakeMove(color, search_result.mcts_move);
    dbg_str += "\n" + GtpBoardString(game_->board());
    std::cerr << dbg_str << "\n";
  } else {
    MakeMove(color, search_result.mcts_move);
  }

  return MakeResponse(id, search_result.mcts_move);
}

Response<std::string> ServiceImpl::GtpPrintBoard(std::optional<int> id) {
  std::stringstream ss;
  ss << "\n" << GtpBoardString(game_->board());

  return MakeResponse(id, ss.str());
}

Response<game::Scores> ServiceImpl::GtpFinalScore(std::optional<int> id) {
  return MakeResponse(id, game_->GetScores());
}

Response<> ServiceImpl::GtpUndo(std::optional<int> id) {
  if (game_->num_moves() == 0) {
    return MakeErrorResponse(id, "Nothing to undo.");
  }

  // Replay from beginning, excluding last move.
  std::unique_ptr<Game> new_game = std::make_unique<Game>(false);
  new_game->SetKomi(komi_);
  for (int move_num = 0; move_num < game_->num_moves() - 1; ++move_num) {
    Move move = game_->move(move_num);
    new_game->PlayMove(move.loc, move.color);
  }
  game_ = std::move(new_game);

  if (game_->num_moves() == 0) {
    current_color_ = BLACK;
  } else {
    current_color_ =
        game::OppositeColor(game_->move(game_->num_moves() - 1).color);
  }

  // Allocate fresh root in a fresh table.
  node_table_ =
      cfg_.use_mcgs
          ? std::unique_ptr<NodeTable>(std::make_unique<McgsNodeTable>())
          : std::unique_ptr<NodeTable>(std::make_unique<MctsNodeTable>());
  current_root_ = node_table_->GetOrCreate(game_->board().hash(),
                                           current_color_, game_->IsGameOver());
  return MakeResponse(id);
}

Response<> ServiceImpl::GtpStartAnalysis(std::optional<int> id, Color color) {
  auto analysis_loop = [this, color]() {
    while (analysis_running_.load(std::memory_order_acquire)) {
      GenMoveCommon(color);
    }
  };

  analysis_running_.store(true, std::memory_order_release);
  analysis_thread_ = std::thread(analysis_loop);

  return MakeResponse(id);
}

analysis::AnalysisSnapshot ServiceImpl::GtpAnalysisSnapshot(Color color) {
  if (analysis_running_.load(std::memory_order_acquire)) {
    return analysis::ConstructAnalysisSnapshot(current_root());
  }

  return analysis::AnalysisSnapshot{};
}

void ServiceImpl::GtpStopAnalysis() {
  if (!analysis_running_.load(std::memory_order_acquire)) {
    return;
  }

  analysis_running_.store(false, std::memory_order_release);
  if (analysis_thread_.joinable()) {
    analysis_thread_.join();
  }
}

game::Loc ServiceImpl::GtpGenMoveAnalyze(game::Color color) {
  analysis_running_.store(true, std::memory_order_release);
  GumbelResult search_result = GenMoveCommon(color);
  analysis_running_.store(false, std::memory_order_release);

  last_explain_comment_ = BuildExplainComment();
  MakeMove(color, search_result.mcts_move);
  return search_result.mcts_move;
}

Response<std::string> ServiceImpl::GtpPlayDbg(std::optional<int> id,
                                              game::Move move) {
  if (!game_->IsValidMove(move.loc, move.color)) {
    return MakeErrorResponse<std::string>(id, "illegal move");
  }

  std::string dbg_str;
  dbg_str += "---- Move Num " + std::to_string(game_->num_moves()) + " ----\n";
  dbg_str += "\nTree Stats:\n  N: " +
             std::to_string(static_cast<int>(N(current_root()))) +
             "\n  V: " + std::to_string(V(current_root())) +
             "\n  Score: " + std::to_string(current_root()->init_score_est) +
             "\n";

  MakeMove(move.color, move.loc);
  dbg_str += "\n" + GtpBoardString(game_->board());

  return MakeResponse(id, dbg_str);
}

std::string ServiceImpl::BuildMoveDbgString(const GumbelResult& result,
                                            Color color) {
  TreeNode* root = current_root();

  float v_nn = root->init_util_est;
  float v_nn_adj =
      (root->bias_cache_entry && root->last_weight_term != 0.0f)
          ? v_nn - cfg_.bias_cache_lambda *
                       (root->last_obs_bias_term / root->last_weight_term)
          : v_nn;

  std::string s;
  s += "---- Move Num " + std::to_string(game_->num_moves()) + " ----\n";
  s += "NN"
       "  V=" +
       std::to_string(v_nn) + "  Vz=" + std::to_string(root->init_outcome_est) +
       "  Vadj=" + std::to_string(v_nn_adj) +
       "  Score=" + std::to_string(root->init_score_est) +
       "    Tree"
       "  N=" +
       std::to_string(static_cast<int>(N(root))) +
       "  V=" + std::to_string(V(root)) +
       "  Vz=" + std::to_string(VOutcome(root)) +
       "  Vstd=" + std::to_string(std::sqrt(VVar(root))) +
       "  Vzstd=" + std::to_string(std::sqrt(VOutcomeVar(root))) + "\n";

  s += "Top Policy Moves:\n";
  for (const auto& mv : GetTopPolicyMoves(root->move_probs)) {
    s += "  " + GtpValueString(mv.first) + " = " + std::to_string(mv.second) +
         "\n";
  }

  s += "Gumbel Selected Moves:\n";
  for (const auto& child : result.child_stats) {
    s += "  " + GtpValueString(child.move) + "  n=" + std::to_string(child.n) +
         "  p=" + std::to_string(child.prob) +
         "  p'=" + std::to_string(child.improved_policy) +
         "  q=" + std::to_string(child.q) + "  qz=" + std::to_string(child.qz) +
         "\n";
  }

#ifdef V_CATEGORICAL
  s += VCategoricalHistogram(root);
#endif

  return s;
}

Response<std::string> ServiceImpl::GtpExplainLastMove(std::optional<int> id) {
  return MakeResponse(id, last_explain_comment_);
}

std::string ServiceImpl::BuildExplainComment() {
  TreeNode* root = current_root();

  float v_tree = V(root);
  float vz_tree = VOutcome(root);
  float vstd = std::sqrt(VVar(root));
  float score = Score(root);
  float v_nn = root->init_util_est;
  float v_adj =
      (root->bias_cache_entry && root->last_weight_term != 0.0f)
          ? v_nn - cfg_.bias_cache_lambda *
                       (root->last_obs_bias_term / root->last_weight_term)
          : v_nn;

  std::stringstream ss;
  ss << std::fixed << std::setprecision(4);
  ss << "Vnn=" << v_nn << " Vadj=" << v_adj << " V=" << v_tree
     << " Score=" << score << " Vz=" << vz_tree << " Vstd=" << vstd << "\n";

  struct ChildInfo {
    int action;
    float lcb;
    int n;
    float q;
    float qstd;
    float prior;
    float opt_prior;
  };

  std::vector<ChildInfo> visited_children;
  for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
    if (root->child_visits[a] == 0) continue;
    visited_children.push_back(ChildInfo{
        a,
        Lcb(root, a),
        root->child_visits[a],
        Q(root, a),
        std::sqrt(QVar(root, a)),
        root->move_probs[a],
        root->opt_probs[a],
    });
  }

  std::sort(
      visited_children.begin(), visited_children.end(),
      [](const ChildInfo& x, const ChildInfo& y) { return x.lcb > y.lcb; });

  static constexpr int kNumTopMoves = 8;
  int num_to_show = std::min((int)visited_children.size(), kNumTopMoves);
  ss << "Top Moves (by LCB):\n";
  for (int i = 0; i < num_to_show; ++i) {
    const auto& c = visited_children[i];
    ss << "  " << GtpValueString(AsLoc(c.action)) << "  n=" << c.n
       << "  q=" << c.q << "  qstd=" << c.qstd << "  lcb=" << c.lcb
       << "  prior=" << c.prior << "  opt_prior=" << c.opt_prior << "\n";
  }

  return ss.str();
}

Response<std::string> ServiceImpl::GtpGenMoveDbg(std::optional<int> id,
                                                 game::Color color) {
  GumbelResult search_result = GenMoveCommon(color);
  std::string dbg_str = BuildMoveDbgString(search_result, color);

  if (V(current_root()) < -0.96f && game_->num_moves() > 50) {
    dbg_str += "p3achygo resigns :(";
    return MakeResponse(id, dbg_str);
  }

  MakeMove(color, search_result.mcts_move);
  dbg_str += "\n" + GtpBoardString(game_->board()) += "\n";
  return MakeResponse(id, dbg_str);
}

Response<std::array<float, BOARD_LEN * BOARD_LEN>> ServiceImpl::GtpOwnership(
    std::optional<int> id) {
  return MakeResponse(
      id, nn_interface_->LoadAndGetOwnership(0, *game_, current_color_));
}

Response<std::string> ServiceImpl::GtpSerializeSgfWithTrees(
    std::optional<int> id, std::string filename) {
  bool success =
      recorder::RecordSingleSgfWithTrees(filename, *game_, {current_root_});
  if (!success) {
    return MakeErrorResponse<std::string>(id, "Failed to write SGF file");
  }
  return MakeResponse<std::string>(id, filename);
}

Response<> ServiceImpl::GtpLoadSgf(std::optional<int> id,
                                   std::string filename) {
  absl::StatusOr<std::unique_ptr<sgf::SgfNode>> sgf_root =
      ParseSgfFile(filename);
  if (!sgf_root.ok()) {
    return MakeErrorResponse(id, std::string(sgf_root.status().message()));
  }

  GameInfo game_info = ExtractGameInfo(sgf_root->get());

  ClearBoard();
  for (const auto& mv : game_info.main_variation) {
    MakeMove(mv.color, mv.loc);
  }

  return MakeResponse(id);
}

GumbelResult ServiceImpl::GenMoveCommon(Color color) {
  if (!(color == current_color_)) {
    // We are moving twice consecutively. Fake a pass first.
    game_->PlayMove(kPassLoc, current_color_);
    TreeNode* next_root = current_root_->children[kPassLoc];
    if (!next_root) {
      next_root = node_table_->GetOrCreate(game_->board().hash(), color,
                                           game_->IsGameOver());
    }
    node_table_->Reap(next_root);
    if (bias_cache_) bias_cache_->PruneUnused();
    current_root_ = next_root;
    current_color_ = color;
  }

  GumbelResult search_result =
      cfg_.use_puct
          ? gumbel_evaluator_->SearchRootPuct(
                probability_, *game_, node_table_.get(), current_root(), color,
                cfg_.n, MakePuctParams(cfg_))
          : gumbel_evaluator_->SearchRoot(
                probability_, *game_, node_table_.get(), current_root(), color,
                mcts::GumbelSearchParams{cfg_.n, cfg_.k, cfg_.noise_scaling});
  return search_result;
}

std::vector<std::pair<game::Loc, float>> ServiceImpl::GetTopPolicyMoves(
    const std::array<float, constants::kMaxMovesPerPosition>& pi) {
  static constexpr int kNumTopMoves = 8;
  std::vector<std::pair<game::Loc, float>> top_policy_moves;
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
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

  TreeNode* next_root = current_root_->children[loc];
  if (!next_root) {
    next_root = node_table_->GetOrCreate(game_->board().hash(), current_color_,
                                         game_->IsGameOver());
  }
  node_table_->Reap(next_root);
  if (bias_cache_) bias_cache_->PruneUnused();
  current_root_ = next_root;
}

void ServiceImpl::ClearBoard() {
  game_ = std::make_unique<Game>(false);
  game_->SetKomi(komi_);
  node_table_ =
      cfg_.use_mcgs
          ? std::unique_ptr<NodeTable>(std::make_unique<McgsNodeTable>())
          : std::unique_ptr<NodeTable>(std::make_unique<MctsNodeTable>());
  current_color_ = BLACK;
  current_root_ = node_table_->GetOrCreate(game_->board().hash(), BLACK, false);
}
}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<Service>> Service::CreateService(
    std::string model_path, eval::PlayerSearchConfig cfg, bool verbose) {
  std::unique_ptr<nn::Engine> engine =
      nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path, 1,
                       nn::GetVersionFromModelPath(model_path));
  std::unique_ptr<NNInterface> nn_interface = std::make_unique<NNInterface>(
      1, std::numeric_limits<int64_t>::max(), 16384, std::move(engine));
  return std::make_unique<ServiceImpl>(std::move(nn_interface), cfg, verbose);
}

}  // namespace gtp