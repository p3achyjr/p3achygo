#if 0
#pragma once

namespace mcts {

class Search final {
 public:
  Search(const int num_threads);
  // Disable Copy and Move.
  Search(Search const&) = delete;
  Search& operator=(Search const&) = delete;
  Search(Search&&) = delete;
  Search& operator=(Search&&) = delete;

 private:
};

}  // namespace mcts
#endif
