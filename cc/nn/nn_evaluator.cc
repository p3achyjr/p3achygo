#include "cc/nn/nn_evaluator.h"

using ::tensorflow::RunOptions;
using ::tensorflow::SessionOptions;
using ::tensorflow::Status;
using ::tensorflow::Tensor;

namespace nn {

NNEvaluator::NNEvaluator()
    : session_options_(SessionOptions()), run_options_(RunOptions()) {
  auto* device_count = session_options_.config.mutable_device_count();
  device_count->insert({"GPU", 1});
}

Status NNEvaluator::InitFromPath(std::string&& path) {
  return LoadSavedModel(session_options_, run_options_, path,
                        {tensorflow::kSavedModelTagServe}, &model_bundle_);
}

Status NNEvaluator::Infer(
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_names,
    std::vector<Tensor>* output_buf) {
  return model_bundle_.GetSession()->Run(inputs, output_names, {}, output_buf);
}

}  // namespace nn