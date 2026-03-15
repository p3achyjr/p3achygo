#pragma once

#include <sstream>
#include <string>

#include "cc/constants/constants.h"
#include "cc/core/lru_cache.h"
#include "cc/core/probability.h"
#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/mcts/tree.h"

namespace mcts {

static constexpr int kPatternLen = 5;
static constexpr int kNumPatternStates = 4;
class LocalPatternZobristTable final {
 public:
  LocalPatternZobristTable() {
    core::PRng prng;

    for (auto i = 0; i < kPatternLen; i++) {
      for (auto j = 0; j < kPatternLen; j++) {
        for (auto k = 1; k < kNumPatternStates; k++) {
          // leave EMPTY as 0.
          table_[i][j][k] = prng.next64();
        }
      }
    }
  }

  inline uint64_t hash_at(unsigned i, unsigned j, unsigned state) const {
    // BLACK = 1 OFF_BOARD = 2 WHITE = 3
    if (state == EMPTY) {
      return 0;
    }
    const unsigned zobrist_index = state == WHITE ? 3 : state;
    return table_[i][j][zobrist_index];
  }

  static const LocalPatternZobristTable& get() {
    static LocalPatternZobristTable zobrist;
    return zobrist;
  }

 private:
  uint64_t table_[kPatternLen][kPatternLen][kNumPatternStates]{};
};

struct LocalPattern final {
  uint64_t grid_hash = 0;
  uint64_t atari_hash = 0;
  uint64_t ko_hash = 0;
  std::array<game::Color, kPatternLen * kPatternLen> grid{};
  std::array<game::Color, kPatternLen * kPatternLen> atari{};
  std::array<game::Color, kPatternLen * kPatternLen> ko{};
  game::Move last_move;
  game::Move two_moves_ago;

  static inline int lin_index(int i, int j) { return i * kPatternLen + j; }

  static std::optional<LocalPattern> FromCurrentPosition(
      const game::Game& game) {
    constexpr int kGridOff = kPatternLen / 2;
    LocalPattern pattern;
    pattern.last_move = game.move(game.num_moves() - 1);
    pattern.two_moves_ago = game.move(game.num_moves() - 2);
    game::Loc last_loc = pattern.last_move.loc;
    if (pattern.two_moves_ago.loc == game::kNoopLoc ||
        last_loc == game::kNoopLoc || last_loc == game::kPassLoc) {
      return std::nullopt;
    }

    for (int grid_i = 0; grid_i < kPatternLen; ++grid_i) {
      for (int grid_j = 0; grid_j < kPatternLen; ++grid_j) {
        const int i = last_loc.i + grid_i - kGridOff;
        const int j = last_loc.j + grid_j - kGridOff;
        if (i < 0 || i >= BOARD_LEN || j < 0 || j >= BOARD_LEN) {
          pattern.grid[lin_index(grid_i, grid_j)] = OFF_BOARD;
        } else {
          pattern.grid[lin_index(grid_i, grid_j)] = game.board().at(i, j);
          if (game.board().at(i, j) == EMPTY &&
              !game.IsValidMove(game::Loc{i, j},
                                game::OppositeColor(pattern.last_move.color))) {
            pattern.ko[lin_index(grid_i, grid_j)] = 1;
          }

          if (game.board().IsInAtari(game::Loc{i, j})) {
            pattern.atari[lin_index(grid_i, grid_j)] = 1;
          }
        }
      }
    }

    // create hash.
    uint64_t grid_hash = 0;
    uint64_t atari_hash = 0;
    uint64_t ko_hash = 0;
    const LocalPatternZobristTable& zobrist = LocalPatternZobristTable::get();
    for (int grid_i = 0; grid_i < kPatternLen; ++grid_i) {
      for (int grid_j = 0; grid_j < kPatternLen; ++grid_j) {
        grid_hash ^= zobrist.hash_at(grid_i, grid_j,
                                     pattern.grid[lin_index(grid_i, grid_j)]);
        atari_hash ^= zobrist.hash_at(grid_i, grid_j,
                                      pattern.atari[lin_index(grid_i, grid_j)]);
        ko_hash ^= zobrist.hash_at(grid_i, grid_j,
                                   pattern.ko[lin_index(grid_i, grid_j)]);
      }
    }
    pattern.grid_hash = grid_hash;
    pattern.atari_hash = atari_hash;
    pattern.ko_hash = ko_hash;
    return pattern;
  }
};

/*
 * I am making a career out of stealing katago's ideas.
 */
class BiasCache final {
 public:
  // (last_player, last_move, two_moves_ago, pattern_hash, atari_hash, ko_hash)
  using Key = std::tuple<game::Color, game::Loc, game::Loc, uint64_t, uint64_t,
                         uint64_t>;
  // (weighted L1 error, weighted visits)
  using Entry = std::pair<float, float>;
  BiasCache(float alpha = 0.8f, float lambda = 0.4f)
      : alpha_(alpha), lambda_(lambda){};
  ~BiasCache() = default;

  BiasCache(const BiasCache&) = delete;
  BiasCache& operator=(const BiasCache&) = delete;

  inline std::shared_ptr<Entry> GetOrCreate(const LocalPattern& local_pattern) {
    absl::MutexLock l(&mu_);
    Key key = std::make_tuple(
        local_pattern.last_move.color, local_pattern.last_move.loc,
        local_pattern.two_moves_ago.loc, local_pattern.grid_hash,
        local_pattern.atari_hash, local_pattern.ko_hash);

    if (cache_.contains(key)) {
      return cache_[key];
    }

    std::shared_ptr<Entry> entry = std::make_shared<Entry>(0.0f, 0.0f);
    cache_[key] = entry;
    return entry;
  }

  // Updates the node's cache entry and last observed terms, and returns the
  // weighted overall bias.
  inline float UpdateAndFetch(TreeNode* node) {
    absl::MutexLock l(&mu_);
    // assumes node is fully updated.
    float weighted_child_utility = 0;
    for (int a = 0; a < constants::kMaxMovesPerPosition; ++a) {
      const int n_a = node->child_visits[a];
      if (n_a == 0) {
        continue;
      }

      weighted_child_utility += n_a * -(node->child(a)->v);
    }

    const int child_visits = node->n - 1;
    const float obs_err =
        node->init_util_est - (weighted_child_utility / child_visits);

    // recompute terms
    const float weight_term = std::pow(child_visits, alpha_);
    const float obs_bias_term = obs_err * weight_term;
    const float weight_term_delta = weight_term - node->last_weight_term;
    const float obs_bias_term_delta = obs_bias_term - node->last_obs_bias_term;

    std::shared_ptr<Entry>& entry = node->bias_cache_entry;
    entry->first += obs_bias_term_delta;
    entry->second += weight_term_delta;
    node->last_weight_term = weight_term;
    node->last_obs_bias_term = obs_bias_term;
    return lambda_ * (entry->first / entry->second);
  }

  inline uint32_t PruneUnused() {
    return absl::erase_if(cache_, [&](const auto& entry) {
      return entry.second.use_count() <= 1;
    });
  }

 private:
  absl::flat_hash_map<Key, std::shared_ptr<Entry>> cache_;
  const float alpha_;
  const float lambda_;
  absl::Mutex mu_;
};

inline std::string ToString(const LocalPattern& pattern) {
  constexpr int kCenter = kPatternLen / 2;
  auto grid_char = [](game::Color c) -> const char* {
    switch (c) {
      case BLACK:
        return "○";
      case WHITE:
        return "●";
      case EMPTY:
        return "⋅";
      default:
        return "x";  // OFF_BOARD
    }
  };

  std::stringstream ss;
  ss << "Grid (last_move=" << pattern.last_move
     << " two_moves_ago=" << pattern.two_moves_ago;
  ss << std::hex;
  ss << " grid_hash=0x" << pattern.grid_hash << " atari_hash=0x"
     << pattern.atari_hash << " ko_hash=0x" << pattern.ko_hash << "):\n";
  ss << std::dec;
  for (int i = 0; i < kPatternLen; ++i) {
    for (int j = 0; j < kPatternLen; ++j) {
      const char* ch = grid_char(pattern.grid[LocalPattern::lin_index(i, j)]);
      if (i == kCenter && j == kCenter) {
        ss << "[" << ch << "]";
      } else {
        ss << " " << ch << " ";
      }
    }
    ss << "\n";
  }

  ss << "Atari:\n";
  for (int i = 0; i < kPatternLen; ++i) {
    for (int j = 0; j < kPatternLen; ++j) {
      ss << (pattern.atari[LocalPattern::lin_index(i, j)] ? " 1 " : " ⋅ ");
    }
    ss << "\n";
  }

  ss << "Ko:\n";
  for (int i = 0; i < kPatternLen; ++i) {
    for (int j = 0; j < kPatternLen; ++j) {
      ss << (pattern.ko[LocalPattern::lin_index(i, j)] ? " 1 " : " ⋅ ");
    }
    ss << "\n";
  }

  return ss.str();
}
}  // namespace mcts
