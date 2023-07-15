#ifndef CC_MCTS_CONSTANTS_H_
#define CC_MCTS_CONSTANTS_H_

namespace mcts {

static constexpr float kDefaultScoreWeight = .5f;
static constexpr float kMaxQ = 1.0f + kDefaultScoreWeight;
static constexpr float kMinQ = -1.0f - kDefaultScoreWeight;

}  // namespace mcts

#endif
