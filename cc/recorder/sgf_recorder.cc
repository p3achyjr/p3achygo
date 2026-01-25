#include "cc/recorder/sgf_recorder.h"

#include <algorithm>
#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "cc/constants/constants.h"
#include "cc/core/filepath.h"
#include "cc/data/filename_format.h"
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
static constexpr int kMaxCallDepth = 32;

// Currently broken for MCGS
#if 0
TreeNode* CloneTree(const TreeNode* node, std::optional<int> blocked_action,
                    int call_depth = 0) {
  if (node == nullptr) {
    return nullptr;
  }

  // Create a new node with the same board hash
  auto cloned = new TreeNode(node->board_hash);

  // Copy all scalar fields
  cloned->state = node->state;
  cloned->is_terminal = node->is_terminal;
  cloned->color_to_move = node->color_to_move;
  cloned->n = node->n;
  cloned->w = node->w;
  cloned->v = node->v;
#ifdef V_CATEGORICAL
  cloned->v_categorical = node->v_categorical;
#endif
  cloned->w_outcome = node->w_outcome;
  cloned->v_outcome = node->v_outcome;
  cloned->v_outcome_var = node->v_outcome_var;
  cloned->max_child_n = node->max_child_n;
  cloned->child_visits = node->child_visits;
  cloned->move_logits = node->move_logits;
  cloned->move_probs = node->move_probs;
  cloned->init_outcome_est = node->init_outcome_est;
  cloned->init_score_est = node->init_score_est;
  cloned->init_util_est = node->init_util_est;

  // Recursively clone children that have been visited (n > 0)
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    if (node->children[i] != nullptr && node->child_visits[i] > 0 &&
        i != blocked_action.value_or(-1)) {
      cloned->children[i] =
          CloneTree(node->children[i], std::nullopt, call_depth + 1);
    }
  }

  return cloned;
}

void DeleteClonedTree(TreeNode* node) {
  if (node == nullptr) {
    return;
  }
  for (int a = 0; a < node->children.size(); ++a) {
    TreeNode* child = node->children[a];
    DeleteClonedTree(child);
  }
  delete node;
}

std::vector<TreeNode*> ReconstructRoots(
    const game::Game& game, const std::vector<mcts::TreeNode*>& roots) {
  if (roots.empty()) return {};
  std::vector<TreeNode*> cloned_roots;
  Loc last_move = kNoopLoc;
  for (int move_num = 0; move_num < game.num_moves(); ++move_num) {
    game::Move move = game.move(move_num);
    TreeNode* root = CloneTree(roots[move_num], move.loc);
    if (!cloned_roots.empty()) {
      if (last_move >= 0) {
        CHECK(cloned_roots.back()->children[last_move] == nullptr);
      }
      cloned_roots.back()->children[move.loc] = root;
      last_move = move.loc;
    }
    cloned_roots.push_back(root);
  }

  return cloned_roots;
}
#endif

void PopulateHeader(SgfNode* root, float komi, game::Game::Result result,
                    const std::string& b_name, const std::string& w_name) {
  root->AddProperty(std::make_unique<SgfKomiProp>(komi));
  root->AddProperty(std::make_unique<SgfResultProp>(result));
  root->AddProperty(std::make_unique<SgfBPlayerProp>(b_name));
  root->AddProperty(std::make_unique<SgfWPlayerProp>(w_name));
}

void PopulateTree(SgfNode* sgf_node, TreeNode* node, Color color) {
  std::vector<std::pair<int, TreeNode*>> visited_children;
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    if (node->children[i] && node->children[i]->n > 0) {
      visited_children.emplace_back(i, node->children[i]);
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
      color == BLACK ? "B" : "W", node->n, node->v, node->v_outcome,
      node->init_outcome_est, node->init_score_est);

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

std::unique_ptr<SgfNode> ToSgfNode(const Game& game, const std::string& b_name,
                                   const std::string& w_name,
                                   const std::vector<TreeNode*>& roots) {
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

    if (!roots.empty()) PopulateTree(current_node, roots[i], move.color);

    current_node = tmp;
  }

  return root_node;
}

class SgfRecorderImpl final : public SgfRecorder {
 public:
  SgfRecorderImpl(std::string path, int num_threads, int gen,
                  std::string worker_id);
  SgfRecorderImpl();
  ~SgfRecorderImpl() override = default;

  // Disable Copy and Move.
  SgfRecorderImpl(SgfRecorderImpl const&) = delete;
  SgfRecorderImpl& operator=(SgfRecorderImpl const&) = delete;
  SgfRecorderImpl(SgfRecorderImpl&&) = delete;
  SgfRecorderImpl& operator=(SgfRecorderImpl&&) = delete;

  // Recorder Impl.
  void RecordGame(int thread_id, const game::Game& game, std::string b_name,
                  std::string w_name,
                  const std::vector<mcts::TreeNode*>& roots) override;
  void RecordEvalGame(int thread_id, const game::Game& game, std::string b_name,
                      std::string w_name) override;
  void Flush() override;

 private:
  struct Record {
    Game game;
    int first_move;
    std::string b_name;
    std::string w_name;

    // Root node for each move of the game.
    // These are cloned and local to the recorder.
    std::vector<TreeNode*> roots;

    ~Record() {
#if 0
      if (!roots.empty()) {
        DeleteClonedTree(roots.front());
      }
#endif
    }
  };

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

void SgfRecorderImpl::RecordGame(int thread_id, const game::Game& game,
                                 std::string b_name, std::string w_name,
                                 const std::vector<mcts::TreeNode*>& roots) {
  CHECK(game.has_result());

  // Deep copy the trees to keep them alive until flush (only if non-empty)
  std::vector<TreeNode*> cloned_roots;
#if 0
  if (!roots.empty()) {
    cloned_roots = ReconstructRoots(game, roots);
  }
#endif

  std::vector<std::unique_ptr<Record>>& thread_records = records_[thread_id];
  auto record =
      std::unique_ptr<Record>(new Record{.game = game,
                                         .first_move = 0,
                                         .b_name = std::move(b_name),
                                         .w_name = std::move(w_name),
                                         .roots = std::move(cloned_roots)});
  thread_records.push_back(std::move(record));
}

void SgfRecorderImpl::RecordEvalGame(int thread_id, const game::Game& game,
                                     std::string b_name, std::string w_name) {
  RecordGame(thread_id, game, b_name, w_name, /*roots=*/{});
}

// Only one thread can call this function. Additionally, no thread can call
// `RecordGame` while this function is running.
void SgfRecorderImpl::Flush() {
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

  if (sgfs != "") {
    // Flush regular SGFs.
    std::string path =
        FilePath(path_) / absl::StrFormat(data::kSgfFormat, gen_, batch_num_,
                                          games_in_batch, worker_id_);
    std::string done_filename =
        FilePath(path_) / absl::StrFormat(data::kSgfDoneFormat, gen_,
                                          batch_num_, games_in_batch,
                                          worker_id_);
    flush(path, done_filename, sgfs);
  }

  if (sgfs_with_trees != "") {
    // Flush full SGFs.
    std::string path_fulls =
        FilePath(path_) / absl::StrFormat(data::kSgfFullFormat, gen_,
                                          batch_num_, games_with_trees_in_batch,
                                          worker_id_);
    std::string full_sgfs_done_filename =
        FilePath(path_) / absl::StrFormat(data::kSgfFullDoneFormat, gen_,
                                          batch_num_, games_with_trees_in_batch,
                                          worker_id_);
    flush(path_fulls, full_sgfs_done_filename, sgfs_with_trees);
  }

  ++batch_num_;
}
}  // namespace

/* static */ std::unique_ptr<SgfRecorder> SgfRecorder::Create(
    std::string path, int num_threads, int gen, std::string worker_id) {
  return std::make_unique<SgfRecorderImpl>(path, num_threads, gen, worker_id);
}

bool RecordSingleSgfWithTrees(std::string path, const game::Game& game,
                              const std::vector<mcts::TreeNode*>& roots) {
  FILE* const sgf_file = fopen(path.c_str(), "w");
  if (sgf_file == nullptr) {
    return false;
  }

  SgfSerializer serializer;
  std::string sgf = serializer.Serialize(
      ToSgfNode(game, "player_b", "player_w", roots).get());
  absl::FPrintF(sgf_file, "%s", sgf);
  fclose(sgf_file);
  return true;
}
}  // namespace recorder
