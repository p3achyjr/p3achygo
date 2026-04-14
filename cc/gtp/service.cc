#include "cc/gtp/service.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stop_token>
#include <thread>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "cc/analysis/analysis.h"
#include "cc/core/probability.h"
#include "cc/eval/player_config.h"
#include "cc/game/game.h"
#include "cc/gtp/time_control.h"
#include "cc/mcts/bias_cache.h"
#include "cc/mcts/gumbel.h"
#include "cc/mcts/node_table.h"
#include "cc/mcts/search.h"
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
using namespace ::eval;

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

class ServiceImpl final : public Service {
 public:
  ServiceImpl(std::unique_ptr<NNInterface> nn_interface,
              eval::PlayerSearchConfig cfg, bool verbose);

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
  Response<> GtpTimeSettings(std::optional<int> id, int main_time_secs,
                             int byoyomi_time_secs,
                             int byoyomi_periods) override;
  Response<> GtpTimeLeft(std::optional<int> id, Color color, int seconds,
                         int stones) override;

  // Analysis methods. Call these from separate threads.
  Response<> GtpStartAnalysis(std::optional<int> id,
                              game::Color color) override;
  analysis::AnalysisSnapshot GtpAnalysisSnapshot(game::Color color) override;
  void GtpStopAnalysis() override;
  game::Loc GtpGenMoveAnalyze(game::Color color) override;
  game::Loc GtpCgosGenmoveAnalyze(game::Color color,
                                  std::string* analysis_json) override;

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
  game::Loc GenMoveCommon(game::Color color);
  std::string BuildExplainComment();
  std::vector<std::pair<game::Loc, float>> GetTopPolicyMoves(
      const std::array<float, constants::kMaxMovesPerPosition>& pi);
  void ClearBoard();
  void MakeMove(Color color, Loc loc);
  void StopPondering();
  void StartPondering();
  void Ponder(std::stop_token token);
  inline TreeNode* current_root() const { return current_root_; }

  // RAII guard: stops pondering on construction, restarts on destruction.
  // Must not be created as a temporary (use [[nodiscard]] to catch this).
  struct [[nodiscard]] PonderPause {
    explicit PonderPause(ServiceImpl& s) : s_(s) { s_.StopPondering(); }
    ~PonderPause() { s_.StartPondering(); }
    PonderPause(const PonderPause&) = delete;
    PonderPause& operator=(const PonderPause&) = delete;

   private:
    ServiceImpl& s_;
  };

  const eval::PlayerSearchConfig cfg_;
  TimeControl time_control_;
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
  Search::Params search_params_;
  std::string last_explain_comment_;

  std::atomic<bool> analysis_running_;
  std::thread analysis_thread_;

  std::jthread ponder_thread_;
};

ServiceImpl::ServiceImpl(std::unique_ptr<NNInterface> nn_interface,
                         eval::PlayerSearchConfig cfg, bool verbose)
    : cfg_(cfg),
      time_control_(cfg.time_control_flags),
      verbose_(verbose),
      nn_interface_(std::move(nn_interface)),
      game_(std::make_unique<Game>(false)),
      node_table_(
          cfg.use_mcgs
              ? std::unique_ptr<NodeTable>(std::make_unique<McgsNodeTable>())
              : std::unique_ptr<NodeTable>(std::make_unique<MctsNodeTable>())),
      current_color_(BLACK),
      search_params_(MakeSearchParams(cfg)),
      analysis_running_(false) {
  if (cfg_.use_bias_cache) {
    bias_cache_.emplace(cfg_.bias_cache_alpha, cfg_.bias_cache_lambda);
  }
  gumbel_evaluator_ = std::make_unique<GumbelEvaluator>(
      nn_interface_.get(), 0, MakeScoreUtilityParams(cfg_),
      bias_cache_ ? &*bias_cache_ : nullptr);
  current_root_ = node_table_->GetOrCreate(game_->board().hash(), BLACK, false);
  StartPondering();
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
    if (!ls_str.empty()) ls_str += "\n";
    ls_str += GTPCodeToString(code);
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
  PonderPause pause(*this);
  ClearBoard();
  return MakeResponse(id);
}

Response<> ServiceImpl::GtpKomi(std::optional<int> id, float komi) {
  PonderPause pause(*this);
  komi_ = komi;
  game_->SetKomi(komi);
  return MakeResponse(id);
}

Response<> ServiceImpl::GtpPlay(std::optional<int> id, Move move) {
  if (!game_->IsValidMove(move.loc, move.color)) {
    return MakeErrorResponse(id, "illegal move");
  }
  if (verbose_) {
    std::cerr << "Playing " << move << "\n";
  }
  StopPondering();
  MakeMove(move.color, move.loc);
  return MakeResponse(id);
}

Response<Loc> ServiceImpl::GtpGenMove(std::optional<int> id, Color color) {
  PonderPause pause(*this);
  const auto mcts_move = GenMoveCommon(color);
  if (V(current_root()) < -0.96f && game_->num_moves() > 50) {
    return MakeResponse(id, kNoopLoc);
  }

  MakeMove(color, mcts_move);
  if (verbose_) {
    std::cerr << last_explain_comment_ << "\n"
              << GtpBoardString(game_->board()) << "\n";
  }

  return MakeResponse(id, mcts_move);
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
  PonderPause pause(*this);

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

// This is a shitty implementation, we should fix it.
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
  const auto mcts_move = GenMoveCommon(color);
  analysis_running_.store(false, std::memory_order_release);

  MakeMove(color, mcts_move);
  return mcts_move;
}

game::Loc ServiceImpl::GtpCgosGenmoveAnalyze(game::Color color,
                                             std::string* analysis_json) {
  PonderPause pause(*this);
  const auto mcts_move = GenMoveCommon(color);
  if (V(current_root()) < -0.96f && game_->num_moves() > 50) {
    return kNoopLoc;
  }

  TreeNode* root = current_root();
  float winrate = (VOutcome(root) + 1.0f) / 2.0f;
  float score = Score(root);

  std::stringstream ss;
  ss << std::fixed << std::setprecision(4);
  ss << "{\"winrate\":" << winrate << ",\"score\":" << score << "}";
  *analysis_json = ss.str();

  MakeMove(color, mcts_move);
  if (verbose_) {
    std::cerr << last_explain_comment_ << "\n"
              << GtpBoardString(game_->board()) << "\n";
  }
  return mcts_move;
}

Response<std::string> ServiceImpl::GtpPlayDbg(std::optional<int> id,
                                              game::Move move) {
  if (!game_->IsValidMove(move.loc, move.color)) {
    return MakeErrorResponse<std::string>(id, "illegal move");
  }
  StopPondering();

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
  PonderPause pause(*this);
  const auto mcts_move = GenMoveCommon(color);
  std::string dbg_str = last_explain_comment_;

  if (V(current_root()) < -0.96f && game_->num_moves() > 50) {
    dbg_str += "p3achygo resigns :(";
    return MakeResponse(id, dbg_str);
  }

  MakeMove(color, mcts_move);
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

  PonderPause pause(*this);
  ClearBoard();
  for (const auto& mv : game_info.main_variation) {
    MakeMove(mv.color, mv.loc);
  }

  return MakeResponse(id);
}

Response<> ServiceImpl::GtpTimeSettings(std::optional<int> id,
                                        int main_time_secs,
                                        int byoyomi_time_secs,
                                        int byoyomi_periods) {
  time_control_.Enable(true);
  time_control_.SetTimeSettings(main_time_secs, byoyomi_time_secs,
                                byoyomi_periods);
  return MakeResponse(id);
}

Response<> ServiceImpl::GtpTimeLeft(std::optional<int> id, Color color,
                                    int seconds, int stones) {
  time_control_.Enable(true);
  // stones == 0 means we are in main time; stones > 0 means byoyomi.
  if (stones == 0) {
    time_control_.SetTimeLeft(seconds, 0, 0);
  } else {
    time_control_.SetTimeLeft(0, seconds, stones);
  }

  if (verbose_) {
    std::cerr << "time_left=" << seconds << "s  byoyomi=" << stones << "\n";
  }
  return MakeResponse(id);
}

game::Loc ServiceImpl::GenMoveCommon(Color color) {
  if (color != current_color_) {
    // We are moving twice consecutively. Fake a pass first.
    MakeMove(color, kPassLoc);
  }

  game::Loc move;
  if (cfg_.num_threads_per_game > 1 || cfg_.time_ms != 0) {
    // requires parallel search.
    BiasCache* bias_cache = bias_cache_.has_value() ? &*bias_cache_ : nullptr;
    Search s(nn_interface_->MakeSlot(0), bias_cache);
    const bool use_visit_budget = cfg_.time_ms == 0;
    const bool use_auto_time = cfg_.time_ms == -1;
    const auto [time_auto, time_metadata] =
        time_control_.ComputeMoveTimeMs(*game_, current_root());
    const int time_ms = [&]() {
      if (use_visit_budget) return -1;
      if (!use_auto_time) return cfg_.time_ms;
      // auto time.
      if (!time_control_.IsEnabled()) return 1000;  // default 1s.
      return time_auto;
    }();
    if (verbose_) {
      std::cerr << "Thinking for " << time_auto << "ms\n";
      if (use_auto_time) {
        std::cerr << "base_ms=" << time_metadata.base_ms
                  << "  appx_moves_left=" << time_metadata.appx_moves_left
                  << "  appx_moves_left_by_q="
                  << time_metadata.appx_moves_left_by_q
                  << "  obv_move_factor=" << time_metadata.obv_move_factor
                  << "  stddev_factor=" << time_metadata.stddev_factor
                  << "  middlegame_factor=" << time_metadata.middlegame_factor
                  << "\n";
      }
    }
    Search::Params search_params = search_params_;
    search_params.total_visit_budget = use_visit_budget ? cfg_.n : 1 << 20;
    search_params.total_visit_time_ms = time_ms;
    Search::Result res = s.Run(probability_, *game_, node_table_.get(),
                               current_root(), color, search_params);
    if (verbose_) {
      std::cerr << "Thought for " << res.time_ms
                << "ms. Visits=" << res.num_visits << "\n";
    }
    move = res.move;
  } else {
    GumbelResult res =
        cfg_.use_puct
            ? gumbel_evaluator_->SearchRootPuct(
                  probability_, *game_, node_table_.get(), current_root(),
                  color, cfg_.n, MakePuctParams(cfg_))
            : gumbel_evaluator_->SearchRoot(
                  probability_, *game_, node_table_.get(), current_root(),
                  color,
                  mcts::GumbelSearchParams{cfg_.n, cfg_.k, cfg_.noise_scaling});
    move = res.mcts_move;
  }

  if (verbose_) {
    last_explain_comment_ = BuildExplainComment();
  }

  return move;
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
  const auto start = std::chrono::steady_clock::now();
  node_table_->Reap(next_root);
  if (bias_cache_) bias_cache_->PruneUnused();
  const auto end = std::chrono::steady_clock::now();
  if (verbose_) {
    const auto dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cerr << "Reap Took " << dur << "ms\n";
  }
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
  node_table_->Reap(current_root_);
  if (bias_cache_) bias_cache_->PruneUnused();
}

void ServiceImpl::StopPondering() {
  ponder_thread_ = {};  // request_stop() + join()
}

void ServiceImpl::StartPondering() {
  if (cfg_.enable_pondering) {
    ponder_thread_ = std::jthread(&ServiceImpl::Ponder, this);
  }
}

void ServiceImpl::Ponder(std::stop_token token) {
  BiasCache* bias_cache = bias_cache_.has_value() ? &*bias_cache_ : nullptr;
  Search s(nn_interface_->MakeSlot(0), bias_cache);
  // StopSearch() fires immediately when the jthread stop is requested.
  std::stop_callback stop_cb(token, [&s] { s.StopSearch(); });

  const auto start = std::chrono::steady_clock::now();
  const auto start_visits = current_root()->n;

  if (verbose_) {
    std::cerr << "Beginning Ponder...\n";
  }

  Search::Params params = search_params_;
  params.total_visit_budget = 1 << 17;
  params.total_visit_time_ms = -1;
  s.Run(probability_, *game_, node_table_.get(), current_root(), current_color_,
        params);

  if (verbose_) {
    const auto end = std::chrono::steady_clock::now();
    const auto end_visits = current_root()->n;
    const auto time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cerr << "Pondered for " << time_ms << "ms, "
              << (end_visits - start_visits) << " visits.\n";
  }
}
}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<Service>> Service::CreateService(
    std::string model_path, eval::PlayerSearchConfig cfg, bool verbose) {
  const int num_threads = cfg.num_threads_per_game;
  const bool uses_psearch = num_threads > 1 || cfg.time_ms != 0;
  std::unique_ptr<nn::Engine> engine =
      nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                       num_threads, nn::GetVersionFromModelPath(model_path));
  std::unique_ptr<NNInterface> nn_interface = std::make_unique<NNInterface>(
      num_threads, std::numeric_limits<int64_t>::max(), 16384,
      std::move(engine),
      uses_psearch ? NNInterface::SignalKind::kExplicit
                   : NNInterface::SignalKind::kAuto,
      1, NNInterface::WakeStrategy::kMutex);
  return std::make_unique<ServiceImpl>(std::move(nn_interface), cfg, verbose);
}

}  // namespace gtp