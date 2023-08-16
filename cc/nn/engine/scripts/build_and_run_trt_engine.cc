#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "cc/nn/engine/build_trt_engine.h"
#include "cc/nn/engine/parse_h5.h"
#include "cc/nn/engine/trt_names.h"

namespace {
namespace nv = ::nvinfer1;
using namespace ::nn;
}  // namespace

ABSL_FLAG(std::string, weights_path, "", "Path to model weights.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string weights_path = absl::GetFlag(FLAGS_weights_path);
  if (weights_path == "") {
    LOG(ERROR) << "No Weights Path (--weights_path) specified.\n";
    return 1;
  }

  std::unique_ptr<model_arch::Model> model = ParseFromH5(weights_path);
  nv::IHostMemory* serialized_engine = nn::trt::BuildEngine(model.get(), 64);

  return 0;
}
