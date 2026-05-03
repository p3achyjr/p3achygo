#ifndef MCTS_CONSTANTS_H_
#define MCTS_CONSTANTS_H_

#include "cc/constants/constants.h"

namespace mcts {

static constexpr float kDefaultScoreWeight = .5f;
static constexpr float kMaxQ = 1.0f + kDefaultScoreWeight;
static constexpr float kMinQ = -1.0f - kDefaultScoreWeight;

static constexpr int kNumVBuckets = constants::kNumVBuckets;
static constexpr float kBucketRange = constants::kBucketRange;
static constexpr int kVBucketMidpoint = constants::kVBucketMidpoint;

}  // namespace mcts

#endif
