#include "cc/nn/nn_wrapper.h"

namespace nn {

NNWrapper::NNWrapper()
    : session_options_(SessionOptions()), run_options_(RunOptions()) {}

NNWrapper::LoadFromPath(std::string&& path) {
  SavedModelBundleLite model_bundle;
  SessionOptions session_options = SessionOptions();
  RunOptions run_options = RunOptions();
  Status status = LoadSavedModel(session_options_, run_options_, path,
                                 {kSavedModelTagServe}, &model_bundle_);
}

}  // namespace nn