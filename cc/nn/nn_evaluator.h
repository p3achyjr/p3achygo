#ifndef __NN_WRAPPER_H_
#define __NN_WRAPPER_H_

#include <string>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

namespace nn {

class NNEvaluator final {
 public:
  NNEvaluator();
  ~NNEvaluator() = default;

  // Disable Copy
  NNEvaluator(NNEvaluator const&) = delete;
  NNEvaluator& operator=(NNEvaluator const&) = delete;

  ::tensorflow::Status InitFromPath(std::string&& path);
  ::tensorflow::Status Infer(
      const std::vector<std::pair<std::string, ::tensorflow::Tensor>>& inputs,
      const std::vector<std::string>& output_names,
      std::vector<::tensorflow::Tensor>* output_buf);

 private:
  ::tensorflow::SavedModelBundleLite model_bundle_;
  ::tensorflow::SessionOptions session_options_;
  ::tensorflow::RunOptions run_options_;
};

}  // namespace nn

#endif  // __NN_WRAPPER_H_