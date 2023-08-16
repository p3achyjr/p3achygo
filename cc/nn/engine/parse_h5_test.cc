#include "cc/nn/engine/parse_h5.h"

#include <filesystem>

#include "absl/log/log.h"
#include "cc/core/doctest_include.h"
#include "cc/nn/engine/validate_model_arch.h"

namespace nn {
namespace {
using namespace ::nn::model_arch;
namespace fs = std::filesystem;
}  // namespace

TEST_CASE("ParseHdf5Test") {
  fs::path current_file_path = fs::path(__FILE__);
  fs::path pwd = current_file_path.parent_path();
  fs::path rel_model_path("__testdata__/model.h5");
  fs::path model_path = pwd / rel_model_path;

  SUBCASE("Parse Succeeds") {
    std::unique_ptr<model_arch::Model> model = ParseFromH5(model_path);
  }

  SUBCASE("Model is Sane") {
    std::unique_ptr<model_arch::Model> model = ParseFromH5(model_path);
    ValidateModelArch(model.get());
  }
}
}  // namespace nn
