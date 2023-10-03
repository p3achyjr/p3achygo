#ifndef NN_ENGINE_TRT_NAMES_H_
#define NN_ENGINE_TRT_NAMES_H_

namespace nn {
namespace trt {
namespace input {

// Keep these in sync with //python/scripts:convert_to_onnx.py
static constexpr char kPlanesName[] = "board_state";
static constexpr char kFeaturesName[] = "game_state";
static constexpr char kScoresName[] = "scores";
}  // namespace input

namespace output {
static constexpr char kPiLogitsName[] = "00:pi_logits";
static constexpr char kPiProbsName[] = "01:pi";
static constexpr char kOutcomeName[] = "03:outcome";
static constexpr char kOwnershipName[] = "04:own";
static constexpr char kScoreName[] = "06:score_probs";
}  // namespace output
}  // namespace trt
}  // namespace nn

#endif
