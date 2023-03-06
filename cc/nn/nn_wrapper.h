#ifndef __NN_WRAPPER_H_
#define __NN_WRAPPER_H_

#include <string>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

namespace nn {

class NNWrapper final {
 public:
  NNWrapper();
  ~NNWrapper() = default;

  // Disable Copy
  NNWrapper(NNWrapper const&) = delete;
  NNWrapper& operator=(NNWrapper const&) = delete;

  void LoadFromPath(std::string&& path);

 private:
  SavedModelBundleLite model_bundle_;
  SessionOptions session_options_;
  RunOptions run_options_;
};

}  // namespace nn

#endif  // __NN_WRAPPER_H_