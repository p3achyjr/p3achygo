#include "cc/nn/engine/validate_model_arch.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/core/util.h"

namespace nn {
namespace model_arch {
namespace {
Layer& GetLayerFromConvBlock(Block& conv_block, Layer::Kind kind) {
  for (Layer& layer : conv_block.layers) {
    if (layer.kind == kind) {
      return layer;
    }
  }

  LOG(FATAL) << "Could not get conv layer from conv block";
}

int ShapeSize(const std::vector<int>& shape) {
  int size = 1;
  for (const int& dim : shape) {
    size *= dim;
  }

  return size;
}

void ValidateConvLayer(Layer& conv_layer, int c_in, int c_out) {
  CHECK(conv_layer.kind == Layer::Kind::kConv) << conv_layer.name;
  TensorDesc& kernel = conv_layer.properties[Layer::kKernel].weights.value();
  TensorDesc& bias = conv_layer.properties[Layer::kBias].weights.value();
  CHECK(kernel.shape[0] == c_out) << conv_layer.name;
  CHECK(kernel.shape[1] == c_in) << conv_layer.name;
  CHECK(kernel.shape[2] == kernel.shape[3]) << conv_layer.name;
  CHECK(kernel.weights.size() == ShapeSize(kernel.shape)) << conv_layer.name;
  CHECK(bias.shape[0] == c_out) << conv_layer.name;
  CHECK(bias.weights.size() == ShapeSize(bias.shape)) << conv_layer.name;
}

void ValidateDenseLayer(Layer& dense_layer, int d_in, int d_out) {
  CHECK(dense_layer.kind == Layer::Kind::kDense) << dense_layer.name;
  TensorDesc& kernel = dense_layer.properties[Layer::kKernel].weights.value();
  TensorDesc& bias = dense_layer.properties[Layer::kBias].weights.value();
  CHECK(kernel.shape[0] == d_in) << dense_layer.name;
  CHECK(kernel.shape[1] == d_out) << dense_layer.name;
  CHECK(kernel.weights.size() == ShapeSize(kernel.shape)) << dense_layer.name;
  CHECK(bias.shape[0] == d_out) << dense_layer.name;
  CHECK(bias.weights.size() == ShapeSize(bias.shape)) << dense_layer.name;
}

void ValidateBottleneckResBlock(const Model* model, ResidualBlock& res_block) {
  for (size_t i = 0; i < res_block.blocks.size(); ++i) {
    Block& block = res_block.blocks[i];
    CHECK(block.kind == Block::Kind::kConv) << res_block.name;

    int c_in, c_out;
    if (i == 0) {
      c_in = model->num_channels;
      c_out = model->num_bottleneck_channels;
    } else if (i == res_block.blocks.size() - 1) {
      c_in = model->num_bottleneck_channels;
      c_out = model->num_channels;
    } else {
      c_in = model->num_bottleneck_channels;
      c_out = model->num_bottleneck_channels;
    }

    ValidateConvLayer(GetLayerFromConvBlock(block, Layer::Kind::kConv), c_in,
                      c_out);
  }
}

void ValidateBroadcastResBlock(const Model* model, ResidualBlock& res_block) {
  CHECK(res_block.blocks.size() == 3) << res_block.name;

  // First Conv.
  CHECK(res_block.blocks[0].kind == Block::Kind::kConv) << res_block.name;
  ValidateConvLayer(
      GetLayerFromConvBlock(res_block.blocks[0], Layer::Kind::kConv),
      model->num_channels, model->num_channels);

  // TODO: Validate Middle Block. We do not know the board length for the model.
  // Last Conv.
  CHECK(res_block.blocks[2].kind == Block::Kind::kConv) << res_block.name;
  ValidateConvLayer(
      GetLayerFromConvBlock(res_block.blocks[2], Layer::Kind::kConv),
      model->num_channels, model->num_channels);
}
}  // namespace

/*
 * Sanity checks model. Aborts on failure.
 */
void ValidateModelArch(Model* model) {
  // Check board length matches what binary supports.
  CHECK(model->board_len == BOARD_LEN);

  // Check init conv dims match model config.
  ValidateConvLayer(GetLayerFromConvBlock(model->init_conv, Layer::Kind::kConv),
                    model->num_input_planes, model->num_channels);

  // Check init game state dims match model config.
  ValidateDenseLayer(model->init_game_state, model->num_input_features,
                     model->num_channels);

  // Check Trunk Blocks
  for (ResidualBlock& res_block : model->trunk.res_blocks) {
    if (res_block.kind == ResidualBlock::Kind::kBottleneck) {
      ValidateBottleneckResBlock(model, res_block);
    } else if (res_block.kind == ResidualBlock::Kind::kBroadcast) {
      ValidateBroadcastResBlock(model, res_block);
    } else {
      LOG(FATAL) << "Unknown Residual Block Kind: "
                 << static_cast<int>(res_block.kind);
    }
  }

  // Policy Head
  ValidateConvLayer(model->policy_head.conv_policy, model->num_channels,
                    model->num_head_channels);
  ValidateConvLayer(model->policy_head.conv_global, model->num_channels,
                    model->num_head_channels);
  ValidateConvLayer(model->policy_head.conv_moves, model->num_head_channels, 2);
  ValidateDenseLayer(model->policy_head.dense_pass,
                     2 * model->num_head_channels, 2);

  // Value Head
  ValidateConvLayer(model->value_head.conv_value, model->num_channels,
                    model->num_head_channels);
  ValidateDenseLayer(model->value_head.dense_outcome, model->num_value_channels,
                     5);
  ValidateConvLayer(model->value_head.conv_ownership, model->num_head_channels,
                    1);
  ValidateDenseLayer(model->value_head.dense_score, model->num_value_channels,
                     1);
}

}  // namespace model_arch
}  // namespace nn
