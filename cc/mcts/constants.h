#ifndef MCTS_CONSTANTS_H_
#define MCTS_CONSTANTS_H_

namespace mcts {

static constexpr float kDefaultScoreWeight = .5f;
static constexpr float kMaxQ = 1.0f + kDefaultScoreWeight;
static constexpr float kMinQ = -1.0f - kDefaultScoreWeight;

static constexpr int kNumVBuckets = 51;
static constexpr float kBucketRange = 2.0f / kNumVBuckets;

}  // namespace mcts

#endif
