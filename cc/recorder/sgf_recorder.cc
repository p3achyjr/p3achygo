#include "cc/recorder/sgf_recorder.h"

#include <algorithm>
#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/filepath.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/mcts/tree.h"
#include "cc/sgf/sgf_serializer.h"
#include "cc/sgf/sgf_tree.h"

namespace recorder {
namespace {

using namespace ::game;
using namespace ::sgf;

using ::core::FilePath;
using ::mcts::TreeNode;

static constexpr int kMaxNumProperties = 32;
static constexpr char kSgfFormat[] = "gen%03d_b%03d_g%03d_%s.sgfs";
static constexpr char kSgfDoneFormat[] = "gen%03d_b%03d_g%03d_%s.done";
static constexpr char kSgfFullFormat[] = "FULL_gen%03d_b%03d_g%03d_%s.sgfs";
static constexpr char kSgfFullDoneFormat[] = "FULL_gen%03d_b%03d_g%03d_%s.done";

void PopulateHeader(SgfNode* root, float komi, game::Game::Result result,
                    const std::string& b_name, const std::string& w_name) {
  root->AddProperty(std::make_unique<SgfKomiProp>(komi));
  root->AddProperty(std::make_unique<SgfResultProp>(result));
  root->AddProperty(std::make_unique<SgfBPlayerProp>(b_name));
  root->AddProperty(std::make_unique<SgfWPlayerProp>(w_name));
}

void PopulateTree(SgfNode* sgf_node, TreeNode* node, Color color) {
  std::vector<std::pair<int, TreeNode*>> visited_children;
  for (int i = 0; i < constants::kMaxNumMoves; ++i) {
    if (node->children[i] && node->children[i]->n > 0) {
      visited_children.emplace_back(i, node->children[i].get());
    }
  }

  std::sort(std::begin(visited_children), std::end(visited_children),
            [](const std::pair<int, TreeNode*>& l,
               const std::pair<int, TreeNode*>& r) {
              return l.second->n < r.second->n;
            });
  std::string comment_string = absl::StrFormat(
      "Root Color: %s, N: %d, Q: %f, Q_z: %f, nn_outcome_est: %f, "
      "nn_score_est: %f",
      color == BLACK ? "B" : "W", node->n, node->q, node->q_outcome,
      node->outcome_est, node->score_est);

  sgf_node->AddProperty(std::make_unique<SgfCommentProp>(comment_string));
  for (const auto& [mv_index, child] : visited_children) {
    Loc loc = AsLoc(mv_index);
    std::unique_ptr<SgfNode> sgf_child = std::make_unique<SgfNode>();
    if (color == BLACK) {
      sgf_child->AddProperty(std::make_unique<SgfBMoveProp>(loc));
    } else {
      sgf_child->AddProperty(std::make_unique<SgfWMoveProp>(loc));
    }

    PopulateTree(sgf_child.get(), child, OppositeColor(color));
    sgf_node->AddChild(std::move(sgf_child));
  }
}

class SgfRecorderImpl final : public SgfRecorder {
 public:
  SgfRecorderImpl(std::string path, int num_threads, int gen,
                  std::string worker_id);
  ~SgfRecorderImpl() override = default;

  // Disable Copy and Move.
  SgfRecorderImpl(SgfRecorderImpl const&) = delete;
  SgfRecorderImpl& operator=(SgfRecorderImpl const&) = delete;
  SgfRecorderImpl(SgfRecorderImpl&&) = delete;
  SgfRecorderImpl& operator=(SgfRecorderImpl&&) = delete;

  // Recorder Impl.
  void RecordGame(
      int thread_id, const game::Game& game, std::string b_name,
      std::string w_name,
      std::vector<std::unique_ptr<mcts::TreeNode>>&& roots) override;
  void Flush() override;

 private:
  struct Record {
    Game game;
    int first_move;
    std::string b_name;
    std::string w_name;

    // Root node for each move of the game.
    // The child actually played at each root should be moved from.
    std::vector<std::unique_ptr<TreeNode>> roots;
  };
  std::unique_ptr<SgfNode> ToSgfNode(
      const Game& game, const std::string& b_name, const std::string& w_name,
      const std::vector<std::unique_ptr<mcts::TreeNode>>& roots);

  const std::string path_;
  std::array<std::vector<std::unique_ptr<Record>>, constants::kMaxNumThreads>
      records_;
  const int num_threads_;
  const int gen_;
  const std::string worker_id_;
  int batch_num_;
};

SgfRecorderImpl::SgfRecorderImpl(std::string path, int num_threads, int gen,
                                 std::string worker_id)
    : path_(path),
      num_threads_(num_threads),
      gen_(gen),
      worker_id_(worker_id),
      batch_num_(0) {}

void SgfRecorderImpl::RecordGame(
    int thread_id, const game::Game& game, std::string b_name,
    std::string w_name, std::vector<std::unique_ptr<mcts::TreeNode>>&& roots) {
  if (path_.empty()) {
    return;
  }

  CHECK(game.has_result());

  std::vector<std::unique_ptr<Record>>& thread_records = records_[thread_id];
  std::unique_ptr<Record> record = std::make_unique<Record>(
      Record{game, 0, b_name, w_name, std::move(roots)});
  thread_records.push_back(std::move(record));
}

// Only one thread can call this function. Additionally, no thread can call
// `RecordGame` while this function is running.
void SgfRecorderImpl::Flush() {
  if (path_.empty()) {
    return;
  }

  int games_in_batch = 0;
  int games_with_trees_in_batch = 0;
  std::string sgfs = "";
  std::string sgfs_with_trees = "";
  SgfSerializer serializer;
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    std::vector<std::unique_ptr<Record>>& thread_records = records_[thread_id];
    if (thread_records.empty()) {
      continue;
    }

    for (const auto& record : thread_records) {
      int& game_counter =
          record->roots.empty() ? games_in_batch : games_with_trees_in_batch;
      std::string& sgf_string = record->roots.empty() ? sgfs : sgfs_with_trees;
      sgf_string += serializer.Serialize(
          ToSgfNode(record->game, record->b_name, record->w_name, record->roots)
              .get());
      sgf_string += "\n";
      ++game_counter;
    }
    thread_records.clear();
  }

  auto flush = [](std::string path, std::string done_filename,
                  const std::string& sgf_string) {
    // Flush actual contents.
    FILE* const sgf_file = fopen(path.c_str(), "w");
    absl::FPrintF(sgf_file, "%s", sgf_string);
    fclose(sgf_file);

    // Flush lock file.
    FILE* const lock_file = fopen(done_filename.c_str(), "w");
    absl::FPrintF(lock_file, "");
    fclose(lock_file);
  };

  if (!(sgfs == "")) {
    // Flush regular SGFs.
    std::string path =
        FilePath(path_) / absl::StrFormat(kSgfFormat, gen_, batch_num_,
                                          games_in_batch, worker_id_);
    std::string done_filename =
        FilePath(path_) / absl::StrFormat(kSgfDoneFormat, gen_, batch_num_,
                                          games_in_batch, worker_id_);
    flush(path, done_filename, sgfs);
  }

  if (!(sgfs_with_trees == "")) {
    // Flush full SGFs.
    std::string path_fulls =
        FilePath(path_) / absl::StrFormat(kSgfFullFormat, gen_, batch_num_,
                                          games_with_trees_in_batch,
                                          worker_id_);
    std::string full_sgfs_done_filename =
        FilePath(path_) / absl::StrFormat(kSgfFullDoneFormat, gen_, batch_num_,
                                          games_with_trees_in_batch,
                                          worker_id_);
    flush(path_fulls, full_sgfs_done_filename, sgfs_with_trees);
  }

  ++batch_num_;
}

std::unique_ptr<SgfNode> SgfRecorderImpl::ToSgfNode(
    const Game& game, const std::string& b_name, const std::string& w_name,
    const std::vector<std::unique_ptr<mcts::TreeNode>>& roots) {
  std::unique_ptr<SgfNode> root_node = std::make_unique<SgfNode>();
  PopulateHeader(root_node.get(), game.komi(), game.result(), b_name, w_name);

  int num_moves = game.num_moves();
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

    if (!roots.empty()) PopulateTree(current_node, roots[i].get(), move.color);

    current_node = tmp;
  }

  return root_node;
}
}  // namespace

/* static */ std::unique_ptr<SgfRecorder> SgfRecorder::Create(
    std::string path, int num_threads, int gen, std::string worker_id) {
  return std::make_unique<SgfRecorderImpl>(path, num_threads, gen, worker_id);
}
}  // namespace recorder
