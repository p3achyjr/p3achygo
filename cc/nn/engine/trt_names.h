#ifndef NN_ENGINE_TRT_NAMES_H_
#define NN_ENGINE_TRT_NAMES_H_

namespace nn {
namespace trt {
namespace input {
static constexpr char kPlanesName[] = "args_0";    //"input_planes";
static constexpr char kFeaturesName[] = "args_1";  //"input_features";
}  // namespace input

namespace output {
static constexpr char kPolicyCombinedName[] = "policy_combined";
static constexpr char kZqName[] = "zq";
static constexpr char kOwnershipName[] = "ownership";
static constexpr char kScoreName[] = "score";
}  // namespace output
}  // namespace trt
}  // namespace nn

#endif
