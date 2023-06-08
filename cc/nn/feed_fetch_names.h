#ifndef __CC_NN_FEED_FETCH_NAMES_H_
#define __CC_NN_FEED_FETCH_NAMES_H_

#include <string>
#include <vector>

namespace nn {
extern const std::vector<std::string> kInputNames;
extern const std::vector<std::string> kOutputNames;
static constexpr int kNNPolicyIndex = 0;
static constexpr int kNNOutcomeIndex = 1;
static constexpr int kNNOwnIndex = 2;
static constexpr int kNNScoreIndex = 3;
static constexpr int kNNGammaIndex = 4;
}  // namespace nn

#endif
