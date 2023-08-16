#ifndef NN_BACKEND_MODEL_ARCH_H_
#define NN_BACKEND_MODEL_ARCH_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace nn {
namespace model_arch {

struct TensorDesc {
  std::vector<int> shape;
  std::vector<float> weights;
};

enum class ActivationKind : uint8_t {
  kUnknown = 0,
  kLinear = 1,
  kRelu = 2,
  kTanh = 3,
  kSoftplus = 4,
};

struct Layer {
  static constexpr char kKernel[] = "kernel";
  static constexpr char kBias[] = "bias";
  static constexpr char kMovingMean[] = "moving_mean";
  static constexpr char kMovingVariance[] = "moving_variance";
  static constexpr char kBeta[] = "beta";
  static constexpr char kEpsilon[] = "epsilon";
  static constexpr char kActivation[] = "activation";

  struct Property {
    enum Kind : uint8_t {
      kUnknown = 0,
      kWeights = 1,
      kActivation = 2,
      kScalar = 3,
    };

    Kind kind;
    std::optional<TensorDesc> weights;
    std::optional<ActivationKind> activation;
    std::optional<float> scalar;
  };

  enum class Kind : uint8_t {
    kUnknown = 0,
    kConv = 1,
    kDense = 2,
    kBatchNorm = 3,
    kActivation = 4,
  };

  std::string name;
  Kind kind;
  absl::flat_hash_map<absl::string_view, Property> properties;
};

struct Block {
  enum class Kind : uint8_t {
    kUnknown = 0,
    kConv = 1,
    kBroadcast = 2,
  };

  std::string name;
  Kind kind;
  std::vector<Layer> layers;
};

struct ResidualBlock {
  std::string name;
  enum class Kind : uint8_t {
    kUnknown = 0,
    kBottleneck = 1,
    kBroadcast = 2,
  };

  Kind kind;
  std::vector<Block> blocks;
};

struct GlobalPool {
  std::string name;
  int c;
  int h;
  int w;
};

struct GlobalPoolBias {
  std::string name;
  Layer batch_norm_g;
  GlobalPool gpool;
  Layer dense;
};

struct Trunk {
  std::string name;
  std::vector<ResidualBlock> res_blocks;
};

struct PolicyHead {
  std::string name;
  Layer conv_policy;
  Layer conv_global;
  GlobalPoolBias gpool_bias;
  Layer batch_norm;
  Layer conv_moves;
  Layer dense_pass;
};

struct ValueHead {
  std::string name;
  Layer conv_value;
  GlobalPool gpool;
  Layer dense_outcome_pre;
  Layer dense_outcome;
  Layer conv_ownership;
  Layer dense_gamma_pre;
  Layer dense_gamma;
  TensorDesc scores;
  Layer dense_score_pre;
  Layer dense_score;
};

struct Model {
  std::string name;
  int num_input_planes;
  int num_input_features;
  int num_blocks;
  int num_channels;
  int num_bottleneck_channels;
  int num_head_channels;
  int num_value_channels;
  int bottleneck_length;

  Block init_conv;
  Layer init_game_state;
  Trunk trunk;
  PolicyHead policy_head;
  ValueHead value_head;
};

}  // namespace model_arch
}  // namespace nn

#endif
