#ifndef NN_FEED_FETCH_NAMES_H_
#define NN_FEED_FETCH_NAMES_H_

#include <string>
#include <vector>

namespace nn {
extern const std::vector<std::string> kInputNames;
extern const std::vector<std::string> kOutputNames;

// Keep in sync with //python/model.py:call
static constexpr int kPiLogitsIndex = 0;
static constexpr int kQ30Index = 1;
static constexpr int kQ100Index = 2;
static constexpr int kQ200Index = 3;
static constexpr int kPiProbsIndex = 4;
static constexpr int kOutcomeLogitsIndex = 5;
static constexpr int kOutcomeIndex = 6;
static constexpr int kOwnIndex = 7;
static constexpr int kScoreLogitsIndex = 8;
static constexpr int kScoreProbsIndex = 9;
static constexpr int kGammaIndex = 10;
static constexpr int kPiLogitsAuxIndex = 11;
}  // namespace nn

#endif
