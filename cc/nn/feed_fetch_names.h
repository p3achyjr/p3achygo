#ifndef __NN_FEED_FETCH_NAMES_H_
#define __NN_FEED_FETCH_NAMES_H_

#include <string>
#include <vector>

namespace nn {
extern const std::vector<std::string> kInputNames;
extern const std::vector<std::string> kOutputNames;
static constexpr int kPiLogitsIndex = 0;
static constexpr int kPiProbsIndex = 1;
static constexpr int kOutcomeLogitsIndex = 2;
static constexpr int kOutcomeIndex = 3;
static constexpr int kOwnIndex = 4;
static constexpr int kScoreLogitsIndex = 5;
static constexpr int kScoreProbsIndex = 6;
static constexpr int kGammaIndex = 7;
}  // namespace nn

#endif
