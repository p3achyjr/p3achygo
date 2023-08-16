#include "cc/nn/engine/trt_engine_builder.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/nn/engine/trt_calibrator.h"
#include "cc/nn/engine/trt_logger.h"
#include "cc/nn/engine/trt_names.h"

namespace nn {
namespace trt {
namespace {
namespace nv = ::nvinfer1;
using namespace ::nn::model_arch;

struct GlobalPoolBiasOutput {
  nv::ITensor* biased_tensor;
  nv::ITensor* pool_tensor;
};

struct PolicyHeadOutput {
  nv::ITensor* policy_combined;
};

struct ValueHeadOutput {
  nv::ITensor* zq;
  nv::ITensor* ownership;
  nv::ITensor* score_probs;
};

std::vector<std::string>& layer_names() {
  // A hack to ensure that layer names live long enough.
  static std::vector<std::string> names;
  return names;
}

int ShapeSize(const std::vector<int>& shape) {
  int size = 1;
  for (const int& dim : shape) {
    size *= dim;
  }

  return size;
}

nv::Dims Dims1(int d) {
  nv::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = d;
  return dims;
}

nv::ITensor* StaticShapeWithBatch(nv::INetworkDefinition* network,
                                  nv::ITensor* runtime_tensor,
                                  const int* static_shape, int num_elems) {
  nv::IShapeLayer* runtime_shape = network->addShape(*runtime_tensor);
  nv::ISliceLayer* batch_size_slice = network->addSlice(
      *runtime_shape->getOutput(0), Dims1(0), Dims1(1), Dims1(1));
  nv::ITensor* batch_size_singleton = batch_size_slice->getOutput(0);
  nv::Weights slice_size_weights{nv::DataType::kINT32, static_shape, num_elems};
  nv::IConstantLayer* slice_size_layer =
      network->addConstant(Dims1(2), slice_size_weights);
  nv::ITensor* size_concat_elems[] = {batch_size_singleton,
                                      slice_size_layer->getOutput(0)};
  nv::IConcatenationLayer* size_concat_layer =
      network->addConcatenation(size_concat_elems, num_elems);
  size_concat_layer->setAxis(0);

  return size_concat_layer->getOutput(0);
}

void SetName(nv::ILayer* layer, std::string name) {
  std::vector<std::string>& names = layer_names();
  names.emplace_back(name);
  layer->setName(names.back().c_str());
}

nv::IConvolutionLayer* AddConvNd(nv::INetworkDefinition* network,
                                 nv::ITensor* input_tensor, int out_channels,
                                 nv::Dims kernel_dims,
                                 nv::Weights kernel_weights,
                                 nv::Weights bias_weights, std::string name) {
  nv::IConvolutionLayer* layer = network->addConvolutionNd(
      *input_tensor, out_channels, kernel_dims, kernel_weights, bias_weights);
  layer->setPaddingMode(nv::PaddingMode::kSAME_UPPER);
  SetName(layer, name.c_str());
  return layer;
}

nv::IMatrixMultiplyLayer* AddMatmul(nv::INetworkDefinition* network,
                                    nv::ITensor* input_tensor,
                                    nv::ITensor* kernel_tensor,
                                    std::string name) {
  nv::IMatrixMultiplyLayer* layer =
      network->addMatrixMultiply(*input_tensor, nv::MatrixOperation::kNONE,
                                 *kernel_tensor, nv::MatrixOperation::kNONE);
  SetName(layer, name.c_str());
  return layer;
}

nv::IShuffleLayer* AddShuffle(nv::INetworkDefinition* network,
                              nv::ITensor* input_tensor, std::string name) {
  nv::IShuffleLayer* layer = network->addShuffle(*input_tensor);
  SetName(layer, name.c_str());
  return layer;
}

nv::IConstantLayer* AddConstant(nv::INetworkDefinition* network, nv::Dims dims,
                                nv::Weights weights, std::string name) {
  nv::IConstantLayer* layer = network->addConstant(dims, weights);
  SetName(layer, name.c_str());
  return layer;
}

/*
 * Adds convolutional layer to network. Returns single output tensor.
 *
 * `input_tensor`: (N, C, H, W)
 * `layer.kernel`: (k, k, C, C')
 * `layer.bias`: (C')
 * Returns: `tensor`: (N, C', H, W)
 */
nv::ITensor* BuildConvLayer(nv::INetworkDefinition* network,
                            nv::ITensor* input_tensor, Layer& layer) {
  auto validate = [&layer, &input_tensor]() {
    CHECK(input_tensor->getDimensions().nbDims == 4);
    CHECK(layer.kind == Layer::Kind::kConv);
    CHECK(layer.properties.contains(Layer::kKernel));
    CHECK(layer.properties[Layer::kKernel].kind == Layer::Property::kWeights);
    CHECK(layer.properties.contains(Layer::kBias));
    CHECK(layer.properties[Layer::kBias].kind == Layer::Property::kWeights);
  };

  logger().log(Logger::Severity::kINFO, "Building Conv Layer: " + layer.name);

  validate();
  const std::vector<int>& conv_shape =
      layer.properties[Layer::kKernel].weights->shape;
  int out_channels = conv_shape[0];
  int kh = conv_shape[2];
  int kw = conv_shape[3];

  const std::vector<float>& kernel_data =
      layer.properties[Layer::kKernel].weights->weights;
  nv::Weights kernel_weights{nv::DataType::kFLOAT, kernel_data.data(),
                             static_cast<int64_t>(kernel_data.size())};
  const std::vector<float>& bias_data =
      layer.properties[Layer::kBias].weights->weights;
  nv::Weights bias_weights{nv::DataType::kFLOAT, bias_data.data(),
                           static_cast<int64_t>(bias_data.size())};
  nv::IConvolutionLayer* conv_layer =
      AddConvNd(network, input_tensor, out_channels, nv::DimsHW(kh, kw),
                kernel_weights, bias_weights, layer.name);
  conv_layer->setPaddingMode(nv::PaddingMode::kSAME_UPPER);

  return conv_layer->getOutput(0);
}

/*
 * Adds dense layer to network. Returns single output tensor.
 *
 * `input_tensor`: (N, K)
 * `layer.kernel`: (N, K')
 * `layer.bias`: (K',)
 * Returns: `tensor`: (N, K')
 */
nv::ITensor* BuildDenseLayer(nv::INetworkDefinition* network,
                             nv::ITensor* input_tensor, Layer& layer) {
  auto validate = [&layer]() {
    CHECK(layer.kind == Layer::Kind::kDense);
    CHECK(layer.properties.contains(Layer::kKernel));
    CHECK(layer.properties[Layer::kKernel].kind == Layer::Property::kWeights);
    CHECK(layer.properties.contains(Layer::kBias));
    CHECK(layer.properties[Layer::kBias].kind == Layer::Property::kWeights);
  };

  logger().log(nv::ILogger::Severity::kINFO,
               "Building Dense Layer: " + layer.name);

  validate();
  const std::vector<int>& kernel_shape =
      layer.properties[Layer::kKernel].weights->shape;
  int in_dim = kernel_shape[0];
  int out_dim = kernel_shape[1];

  std::cerr << "DENSE. IN DIM: " << in_dim << ", OUT_DIM: " << out_dim << "\n";

  // Parse Kernel Weights
  const std::vector<float>& kernel_data =
      layer.properties[Layer::kKernel].weights->weights;
  nv::Weights kernel_weights{nv::DataType::kFLOAT, kernel_data.data(),
                             static_cast<int64_t>(kernel_data.size())};

  for (const float& weight : kernel_data) {
    std::cerr << " " << weight;
  }
  std::cerr << "\n";

  // Parse Bias Weights.
  const std::vector<float>& bias_data =
      layer.properties[Layer::kBias].weights->weights;
  nv::Weights bias_weights{nv::DataType::kFLOAT, bias_data.data(),
                           static_cast<int64_t>(bias_data.size())};

  // Add Matmul.
  nv::ITensor* kernel_tensor =
      AddConstant(network, nv::Dims2(in_dim, out_dim), kernel_weights,
                  layer.name + ":kernel")
          ->getOutput(0);
  nv::IMatrixMultiplyLayer* matmul_layer =
      AddMatmul(network, input_tensor, kernel_tensor, layer.name + ":matmul");
  nv::ITensor* matmul_output = matmul_layer->getOutput(0);

  // Add Bias.
  nv::Dims2 bias_dims(1, out_dim);
  nv::ITensor* bias_tensor =
      AddConstant(network, bias_dims, bias_weights, layer.name + ":bias")
          ->getOutput(0);
  nv::IElementWiseLayer* bias_add_layer = network->addElementWise(
      *matmul_output, *bias_tensor, nv::ElementWiseOperation::kSUM);
  nv::ITensor* bias_add_output = bias_add_layer->getOutput(0);

  return bias_add_output;
}

/*
 * Adds batchnorm layer to network. Returns single output tensor.
 *
 * `input_tensor`: (N, C, H, W)
 * `layer.beta`: (C,)
 * `layer.moving_mean`: (C,)
 * `layer.moving_variance`: (C,)
 * Returns: `tensor`: (N, C, H, W)
 */
nv::ITensor* BuildBatchNormLayer(nv::INetworkDefinition* network,
                                 nv::ITensor* input_tensor, Layer& layer) {
  auto validate = [&layer]() {
    CHECK(layer.kind == Layer::Kind::kBatchNorm);
    CHECK(layer.properties.contains(Layer::kMovingMean));
    CHECK(layer.properties[Layer::kMovingMean].kind ==
          Layer::Property::kWeights);
    CHECK(layer.properties.contains(Layer::kMovingVariance));
    CHECK(layer.properties[Layer::kMovingVariance].kind ==
          Layer::Property::kWeights);
    CHECK(layer.properties.contains(Layer::kBeta));
    CHECK(layer.properties[Layer::kBeta].kind == Layer::Property::kWeights);
    CHECK(layer.properties.contains(Layer::kEpsilon));
    CHECK(layer.properties[Layer::kEpsilon].kind == Layer::Property::kScalar);

    CHECK(layer.properties[Layer::kMovingMean].weights->shape.size() == 1);
    CHECK(layer.properties[Layer::kMovingMean].weights->shape ==
          layer.properties[Layer::kMovingVariance].weights->shape);
    CHECK(layer.properties[Layer::kMovingVariance].weights->shape ==
          layer.properties[Layer::kBeta].weights->shape);
  };

  logger().log(nv::ILogger::Severity::kINFO,
               "Building BN Layer: " + layer.name);

  validate();

  // Fetch Moving Mean.
  TensorDesc& moving_mean_desc =
      layer.properties[Layer::kMovingMean].weights.value();
  const std::vector<float>& moving_mean_data = moving_mean_desc.weights;

  // Fetch Moving Variance.
  TensorDesc& moving_variance_desc =
      layer.properties[Layer::kMovingVariance].weights.value();
  const std::vector<float>& moving_variance_data = moving_variance_desc.weights;
  std::cerr << "VARIANCE: ";
  for (const auto& weight : moving_variance_desc.weights) {
    std::cerr << " " << weight;
  }
  std::cerr << std::endl;

  // Fetch Beta.
  TensorDesc& beta_desc = layer.properties[Layer::kBeta].weights.value();
  const std::vector<float>& beta_data = beta_desc.weights;

  // Shared Shape.
  nv::Dims4 bn_dims(1, moving_mean_desc.shape[0], 1, 1);
  std::cerr << "BN DIMS: " << bn_dims.d[1] << "\n";

  // Fetch Epsilon
  const float* epsilon = &layer.properties[Layer::kEpsilon].scalar.value();
  std::cerr << "EPSILON: " << *epsilon << std::endl;

  // Moving Mean Weights.
  nv::Weights moving_mean_weights{
      nv::DataType::kFLOAT, moving_mean_data.data(),
      static_cast<int64_t>(moving_mean_data.size())};
  nv::ITensor* moving_mean_tensor =
      AddConstant(network, bn_dims, moving_mean_weights,
                  layer.name + ":moving_mean")
          ->getOutput(0);

  // Moving Variance Weights.
  nv::Weights moving_variance_weights{
      nv::DataType::kFLOAT, moving_variance_data.data(),
      static_cast<int64_t>(moving_variance_data.size())};
  nv::ITensor* moving_variance_tensor =
      AddConstant(network, bn_dims, moving_variance_weights,
                  layer.name + ":moving_variance")
          ->getOutput(0);

  // Beta Weights.
  nv::Weights beta_weights{nv::DataType::kFLOAT, beta_data.data(),
                           static_cast<int64_t>(beta_data.size())};
  nv::ITensor* beta_tensor =
      AddConstant(network, bn_dims, beta_weights, layer.name + ":beta")
          ->getOutput(0);

  // Epsilon Weights.
  nv::Weights eps_weights{nv::DataType::kFLOAT, epsilon, 1};
  nv::ITensor* eps_tensor = AddConstant(network, nv::Dims4(1, 1, 1, 1),
                                        eps_weights, layer.name + ":eps")
                                ->getOutput(0);

  // Compute Denominator.
  nv::IElementWiseLayer* add_var_eps_layer = network->addElementWise(
      *moving_variance_tensor, *eps_tensor, nv::ElementWiseOperation::kSUM);
  nv::ITensor* var_eps_tensor = add_var_eps_layer->getOutput(0);
  nv::IUnaryLayer* sqrt_layer =
      network->addUnary(*moving_variance_tensor, nv::UnaryOperation::kSQRT);
  nv::ITensor* denom_tensor = sqrt_layer->getOutput(0);

  // Compute Numerator.
  nv::IElementWiseLayer* sub_mean_layer = network->addElementWise(
      *input_tensor, *moving_mean_tensor, nv::ElementWiseOperation::kSUB);
  nv::ITensor* numer_tensor = sub_mean_layer->getOutput(0);

  // Divide.
  nv::IElementWiseLayer* div_layer = network->addElementWise(
      *numer_tensor, *denom_tensor, nv::ElementWiseOperation::kDIV);
  nv::ITensor* input_normed_tensor = div_layer->getOutput(0);

  // Add Beta.
  nv::IElementWiseLayer* add_beta_layer = network->addElementWise(
      *input_normed_tensor, *beta_tensor, nv::ElementWiseOperation::kSUM);

  return add_beta_layer->getOutput(0);
}

/*
 * Adds activation layer to network. Returns single output tensor.
 *
 * `input_tensor`: (N, C, H, W)
 * `layer.activation`: model_arch::ActivationType
 * Returns: `tensor`: (N, C, H, W)
 */
nv::ITensor* BuildActivationLayer(nv::INetworkDefinition* network,
                                  nv::ITensor* input_tensor, Layer& layer) {
  auto validate = [&layer]() {
    CHECK(layer.kind == Layer::Kind::kActivation);
    CHECK(layer.properties.contains(Layer::kActivation));
    CHECK(layer.properties[Layer::kActivation].kind ==
          Layer::Property::Kind::kActivation);
  };

  logger().log(nv::ILogger::Severity::kINFO,
               "Building Activation Layer: " + layer.name);

  validate();
  ActivationKind act_kind =
      layer.properties[Layer::kActivation].activation.value();
  if (act_kind == ActivationKind::kLinear) {
    // Nothing to do.
    return input_tensor;
  }

  nv::ActivationType nv_activation;
  if (act_kind == ActivationKind::kRelu) {
    nv_activation = nv::ActivationType::kRELU;
  } else if (act_kind == ActivationKind::kTanh) {
    nv_activation = nv::ActivationType::kTANH;
  } else if (act_kind == ActivationKind::kSoftplus) {
    nv_activation = nv::ActivationType::kSOFTPLUS;
  } else {
    LOG(FATAL) << "Unsupported Activation Type: `" << static_cast<int>(act_kind)
               << "`";
  }

  nv::IActivationLayer* act_layer =
      network->addActivation(*input_tensor, nv_activation);
  return act_layer->getOutput(0);
}

/*
 * Adds a list of layers sequentially to the network. Returns output tensor.
 */
nv::ITensor* BuildLayers(nv::INetworkDefinition* network, nv::ITensor* tensor,
                         std::vector<Layer>& layers) {
  for (Layer& layer : layers) {
    if (layer.kind == Layer::Kind::kConv) {
      tensor = BuildConvLayer(network, tensor, layer);
    } else if (layer.kind == Layer::Kind::kDense) {
      tensor = BuildDenseLayer(network, tensor, layer);
    } else if (layer.kind == Layer::Kind::kBatchNorm) {
      tensor = BuildBatchNormLayer(network, tensor, layer);
    } else if (layer.kind == Layer::Kind::kActivation) {
      tensor = BuildActivationLayer(network, tensor, layer);
    } else {
      LOG(FATAL) << "Unknown Layer Type: `" << static_cast<int>(layer.kind)
                 << "`";
    }
  }

  return tensor;
}

/*
 * Adds a convolutional block to the network. Returns output tensor.
 *
 * `input_tensor`: (N, C, H, W)
 * Returns: `output_tensor`, (N, C', H, W)
 */
nv::ITensor* BuildConvBlock(nv::INetworkDefinition* network,
                            nv::ITensor* input_tensor, Block& conv_block) {
  auto validate = [&conv_block]() {
    CHECK(conv_block.kind == Block::Kind::kConv);
    CHECK(conv_block.layers.size() == 3);
    CHECK(conv_block.layers[0].kind == Layer::Kind::kConv);
    CHECK(conv_block.layers[1].kind == Layer::Kind::kBatchNorm);
    CHECK(conv_block.layers[2].kind == Layer::Kind::kActivation);
  };

  validate();
  return BuildLayers(network, input_tensor, conv_block.layers);
}

/*
 * Adds a broadcast block to the network. Returns output tensor.
 *
 * `input_tensor`: (N, C, H, W)
 * Returns: `output_tensor`, (N, C, H, W)
 */
nv::ITensor* BuildBroadcastBlock(nv::INetworkDefinition* network,
                                 nv::ITensor* input_tensor,
                                 Block& broadcast_block) {
  auto validate = [&broadcast_block, &input_tensor]() {
    CHECK(input_tensor->getDimensions().nbDims == 4);
    CHECK(broadcast_block.kind == Block::Kind::kBroadcast);
    CHECK(broadcast_block.layers.size() == 2);
    CHECK(broadcast_block.layers[0].kind == Layer::Kind::kDense);
    CHECK(broadcast_block.layers[1].kind == Layer::Kind::kActivation);
  };

  validate();
  nv::Dims dims = input_tensor->getDimensions();
  int c = dims.d[1];
  int h = dims.d[2];
  int w = dims.d[3];

  // Flatten input. (N, C, H, W) -> (N*C, H*W)
  nv::IShuffleLayer* flatten_layer = AddShuffle(
      network, input_tensor, broadcast_block.name + ":flatten_pre_dense");
  flatten_layer->setReshapeDimensions(nv::Dims2(-1, h * w));
  nv::ITensor* flattened_tensor = flatten_layer->getOutput(0);

  // FC layer.
  nv::ITensor* fc_output =
      BuildDenseLayer(network, flattened_tensor, broadcast_block.layers[0]);

  // Activation.
  nv::ITensor* act_output =
      BuildActivationLayer(network, fc_output, broadcast_block.layers[1]);

  // Un-flatten input. (-1, H*W) -> (-1, C, H, W)
  nv::IShuffleLayer* unflatten_layer = AddShuffle(
      network, act_output, broadcast_block.name + ":expand_post_dense");
  unflatten_layer->setReshapeDimensions(nv::Dims4(-1, c, h, w));
  return unflatten_layer->getOutput(0);
}

/*
 * Adds a bottleneck residual block to the network. Returns output tensor.
 *
 * `input_tensor`: (N, C, H, W)
 * Returns: `output_tensor`, (N, C, H, W)
 */
nv::ITensor* BuildBottleneckResBlock(nv::INetworkDefinition* network,
                                     nv::ITensor* input_tensor,
                                     ResidualBlock& res_block) {
  auto validate = [&res_block] {
    CHECK(res_block.kind == ResidualBlock::Kind::kBottleneck);
    for (Block& block : res_block.blocks) {
      CHECK(block.kind == Block::Kind::kConv);
    }
  };

  logger().log(nv::ILogger::Severity::kINFO,
               "Building Bottleneck Residual Block: " + res_block.name);

  validate();
  nv::ITensor* tensor = input_tensor;
  for (Block& block : res_block.blocks) {
    // Since we have validated, we know these are all conv blocks.
    tensor = BuildConvBlock(network, tensor, block);
  }

  nv::IElementWiseLayer* res_add_layer = network->addElementWise(
      *input_tensor, *tensor, nv::ElementWiseOperation::kSUM);
  return res_add_layer->getOutput(0);
}

/*
 * Adds a broadcast residual block to the network. Returns output tensor.
 *
 * `input_tensor`: (N, C, H, W)
 * Returns: `output_tensor`, (N, C, H, W)
 */
nv::ITensor* BuildBroadcastResBlock(nv::INetworkDefinition* network,
                                    nv::ITensor* input_tensor,
                                    ResidualBlock& res_block) {
  auto validate = [&res_block] {
    CHECK(res_block.kind == ResidualBlock::Kind::kBroadcast);
    CHECK(res_block.blocks.size() == 3);
    CHECK(res_block.blocks[0].kind == Block::Kind::kConv);
    CHECK(res_block.blocks[1].kind == Block::Kind::kBroadcast);
    CHECK(res_block.blocks[2].kind == Block::Kind::kConv);
  };

  logger().log(nv::ILogger::Severity::kINFO,
               "Building Broadcast Residual Block: " + res_block.name);

  validate();
  nv::ITensor* tensor;

  // Perform Operations.
  tensor = BuildConvBlock(network, input_tensor, res_block.blocks[0]);
  tensor = BuildBroadcastBlock(network, tensor, res_block.blocks[1]);
  tensor = BuildConvBlock(network, tensor, res_block.blocks[2]);

  // Residual Add.
  nv::IElementWiseLayer* res_add_layer = network->addElementWise(
      *input_tensor, *tensor, nv::ElementWiseOperation::kSUM);
  tensor = res_add_layer->getOutput(0);

  // ReLU.
  nv::IActivationLayer* relu_layer =
      network->addActivation(*tensor, nv::ActivationType::kRELU);
  return relu_layer->getOutput(0);
}

/*
 * Builds trunk and returns tensor representing the end of the trunk.
 */
nv::ITensor* BuildTrunk(nv::INetworkDefinition* network, nv::ITensor* tensor,
                        Trunk& trunk) {
  logger().log(nv::ILogger::Severity::kINFO, "Building Trunk: " + trunk.name);
  for (ResidualBlock& res_block : trunk.res_blocks) {
    if (res_block.kind == ResidualBlock::Kind::kBottleneck) {
      tensor = BuildBottleneckResBlock(network, tensor, res_block);
    } else if (res_block.kind == ResidualBlock::Kind::kBroadcast) {
      tensor = BuildBroadcastResBlock(network, tensor, res_block);
    } else {
      LOG(FATAL) << "Unknown Residual Block Kind: `"
                 << static_cast<int>(res_block.kind) << "`.";
    }
  }
  return tensor;
}

/*
 * Builds GlobalPool. Returns mean/max tensor (N, 2C).
 *
 * `input_tensor`: (N, C, H, W)
 * Returns: `output_tensor`, (N, 2C)
 */
nv::ITensor* BuildGlobalPool(nv::INetworkDefinition* network,
                             nv::ITensor* input_tensor, GlobalPool& gpool) {
  auto validate = [&gpool, &input_tensor]() {
    nv::Dims dims = input_tensor->getDimensions();
    nv::Dims4 gpool_dims(-1, gpool.c, gpool.h, gpool.w);

    CHECK(dims.nbDims == 4);
    CHECK(dims.d[1] == gpool_dims.d[1]);
    CHECK(dims.d[2] == gpool_dims.d[2]);
    CHECK(dims.d[3] == gpool_dims.d[3]);
  };

  validate();

  // Flatten. (N, C, H, W) -> (N, 2C)
  nv::Dims input_dims = input_tensor->getDimensions();
  nv::Dims3 flattened_dims(-1, input_dims.d[1],
                           input_dims.d[2] * input_dims.d[3]);
  nv::IShuffleLayer* flatten_layer =
      AddShuffle(network, input_tensor, gpool.name + ":flatten");
  flatten_layer->setReshapeDimensions(flattened_dims);
  nv::ITensor* flat_tensor = flatten_layer->getOutput(0);

  // Channel-wise mean.
  nv::IReduceLayer* mean_layer =
      network->addReduce(*flat_tensor, nv::ReduceOperation::kAVG, 1U << 2,
                         false /* keep_dimensions */);
  nv::ITensor* mean_tensor = mean_layer->getOutput(0);

  // Channel-wise max.
  nv::IReduceLayer* max_layer =
      network->addReduce(*flat_tensor, nv::ReduceOperation::kMAX, 1U << 2,
                         false /* keep_dimensions */);
  nv::ITensor* max_tensor = max_layer->getOutput(0);

  // Concatenate.
  nv::ITensor* concat_tensors[2] = {mean_tensor, max_tensor};
  nv::IConcatenationLayer* concat_layer =
      network->addConcatenation(concat_tensors, 2);
  concat_layer->setAxis(1);  // concat along channel dimension.
  return concat_layer->getOutput(0);
}

/*
 * Builds GlobalPoolBias.
 *
 * `x_tensor`: (N, C, H, W)
 * `g_tensor`: (N, C, H, W)
 *
 * Returns:
 * `biased_tensor`: (N, C, H, W)
 * `pool_tensor`: (N, 2C)
 */
GlobalPoolBiasOutput BuildGlobalPoolBias(nv::INetworkDefinition* network,
                                         nv::ITensor* x_tensor,
                                         nv::ITensor* g_tensor,
                                         GlobalPoolBias& gpool_bias) {
  auto validate = [&gpool_bias]() {
    CHECK(gpool_bias.batch_norm_g.kind == Layer::Kind::kBatchNorm);
    CHECK(gpool_bias.dense.kind == Layer::Kind::kDense);
  };

  validate();

  // Normalize g_tensor.
  g_tensor = BuildBatchNormLayer(network, g_tensor, gpool_bias.batch_norm_g);
  nv::IActivationLayer* relu_layer =
      network->addActivation(*g_tensor, nv::ActivationType::kRELU);
  g_tensor = relu_layer->getOutput(0);

  // Pool (Get Mean/Max)
  nv::ITensor* g_pooled_tensor =
      BuildGlobalPool(network, g_tensor, gpool_bias.gpool);

  // Dense. (N, 2C) -> (N, C)
  nv::ITensor* g_bias_tensor =
      BuildDenseLayer(network, g_pooled_tensor, gpool_bias.dense);

  // Reshape g_bias_tensor for compatibility.
  nv::IShuffleLayer* reshape_layer =
      AddShuffle(network, g_bias_tensor, gpool_bias.name + ":reshape_g_bias");
  nv::Dims4 reshape_dims(-1, g_bias_tensor->getDimensions().d[1], 1, 1);
  reshape_layer->setReshapeDimensions(reshape_dims);

  // Add.
  nv::IElementWiseLayer* bias_layer = network->addElementWise(
      *x_tensor, *reshape_layer->getOutput(0), nv::ElementWiseOperation::kSUM);
  return GlobalPoolBiasOutput{bias_layer->getOutput(0), g_pooled_tensor};
}

/*
 * Builds policy head. Returns 1D tensors representing policy/aux policy.
 *
 * `input_tensor`: (N, C, H, W)
 * Returns: `policy`, `policy_aux`: (N, H*W+1)
 */
PolicyHeadOutput BuildPolicyHead(nv::INetworkDefinition* network,
                                 nv::ITensor* input_tensor,
                                 PolicyHead& policy_head,
                                 int num_head_channels) {
  static const float kPassOffset = 3.0f;
  auto validate = [&]() {
    CHECK(input_tensor->getDimensions().nbDims == 4);
    CHECK(policy_head.conv_policy.kind == Layer::Kind::kConv);
    CHECK(policy_head.conv_global.kind == Layer::Kind::kConv);
    CHECK(policy_head.batch_norm.kind == Layer::Kind::kBatchNorm);
    CHECK(policy_head.conv_moves.kind == Layer::Kind::kConv);
    CHECK(policy_head.dense_pass.kind == Layer::Kind::kDense);

    std::vector<int> conv_policy_shape =
        policy_head.conv_policy.properties[Layer::kKernel].weights->shape;
    std::vector<int> conv_global_shape =
        policy_head.conv_policy.properties[Layer::kKernel].weights->shape;
    CHECK(conv_policy_shape.size() == 4);
    CHECK(conv_policy_shape == conv_global_shape);

    std::vector<int> conv_moves_shape =
        policy_head.conv_moves.properties[Layer::kKernel].weights->shape;
    // Two output channels: one for policy, one for aux policy.
    CHECK(conv_moves_shape.size() == 4 && conv_moves_shape[0] == 2);

    std::vector<int> dense_pass_shape =
        policy_head.dense_pass.properties[Layer::kKernel].weights->shape;
    CHECK(dense_pass_shape.size() == 2 && dense_pass_shape[1] == 2);
  };

  logger().log(nv::ILogger::Severity::kINFO,
               "Building Policy Head: " + policy_head.name);

  validate();

  // Initial Convolutions.
  nv::ITensor* p_tensor =
      BuildConvLayer(network, input_tensor, policy_head.conv_policy);
  nv::ITensor* g_tensor =
      BuildConvLayer(network, input_tensor, policy_head.conv_global);

  // Global Pool.
  GlobalPoolBiasOutput gpool_output =
      BuildGlobalPoolBias(network, p_tensor, g_tensor, policy_head.gpool_bias);

  // BN + ReLU on policy conv.
  p_tensor = BuildBatchNormLayer(network, gpool_output.biased_tensor,
                                 policy_head.batch_norm);
  nv::IActivationLayer* act_layer =
      network->addActivation(*p_tensor, nv::ActivationType::kRELU);
  p_tensor = act_layer->getOutput(0);

  // Board Policy. (N, 2, 19, 19)
  nv::ITensor* board_policy =
      BuildConvLayer(network, p_tensor, policy_head.conv_moves);

  // Pass Policy. (N, 2)
  nv::ITensor* pass_policy = BuildDenseLayer(network, gpool_output.pool_tensor,
                                             policy_head.dense_pass);
  nv::IConstantLayer* const3 =
      AddConstant(network, nv::Dims2(1, 1),
                  nv::Weights{nv::DataType::kFLOAT, &kPassOffset, 1},
                  policy_head.name + ":const3");
  nv::IElementWiseLayer* pass_sub = network->addElementWise(
      *pass_policy, *const3->getOutput(0), nv::ElementWiseOperation::kSUB);
  pass_policy = pass_sub->getOutput(0);

  // Flatten Board Policy. (N, 2, 361)
  nv::Dims board_policy_dims = board_policy->getDimensions();
  nv::Dims3 flat_dims(
      -1, board_policy_dims.d[1],
      board_policy_dims.d[2] * board_policy_dims.d[3]);  // (N, 2, H*W)
  nv::IShuffleLayer* flatten_layer = AddShuffle(
      network, board_policy, policy_head.name + ":flatten_board_policy");
  flatten_layer->setReshapeDimensions(flat_dims);
  board_policy = flatten_layer->getOutput(0);

  // Expand Pass Policy. (N, 2, 1)
  nv::Dims pass_policy_dims = pass_policy->getDimensions();
  nv::Dims3 expanded_dims(-1, pass_policy_dims.d[1], 1);
  nv::IShuffleLayer* expand_layer = AddShuffle(
      network, pass_policy, policy_head.name + ":expand_pass_policy");
  expand_layer->setReshapeDimensions(expanded_dims);
  pass_policy = expand_layer->getOutput(0);

  // Concat. (N, 2, 362)
  nv::ITensor* policy_arr[] = {board_policy, pass_policy};
  nv::IConcatenationLayer* policy_concat_layer =
      network->addConcatenation(policy_arr, 2);
  policy_concat_layer->setAxis(2);
  nv::ITensor* policy_combined = policy_concat_layer->getOutput(0);

  // Return both policy and policy aux. Shape: (N, 2, 362)
  return PolicyHeadOutput{policy_combined};
}

/*
 * Builds value head. Returns outcome/score distributions, and ownership.
 *
 * `input_tensor`: (N, C, H, W)
 * Returns:
 * - `outcome`: (N, 2)
 * - `ownership`: (N, 1, 19, 19)
 * - `score`: (N, 800)
 */
ValueHeadOutput BuildValueHead(nv::INetworkDefinition* network,
                               nv::ITensor* input_tensor, ValueHead& value_head,
                               int num_head_channels, int num_value_channels) {
  static const int kScoreDimsPartial[] = {constants::kNumScoreLogits, 1};
  static const float kZero = 0;
  auto validate = [&]() {
    CHECK(input_tensor->getDimensions().nbDims == 4);
    CHECK(value_head.conv_value.kind == Layer::Kind::kConv);
    CHECK(value_head.dense_outcome_pre.kind == Layer::Kind::kDense);
    CHECK(value_head.dense_outcome.kind == Layer::Kind::kDense);
    CHECK(value_head.conv_ownership.kind == Layer::Kind::kConv);
    CHECK(value_head.dense_gamma_pre.kind == Layer::Kind::kDense);
    CHECK(value_head.dense_gamma.kind == Layer::Kind::kDense);
    CHECK(value_head.dense_score_pre.kind == Layer::Kind::kDense);
    CHECK(value_head.dense_score.kind == Layer::Kind::kDense);
    CHECK(value_head.conv_value.properties[Layer::kKernel].weights->shape[0] ==
          num_head_channels);
    CHECK(ShapeSize(value_head.scores.shape) == constants::kNumScoreLogits);
  };

  logger().log(nv::ILogger::Severity::kINFO,
               "Building Value Head: " + value_head.name);

  validate();

  // Init Value Conv.
  nv::ITensor* conv_v =
      BuildConvLayer(network, input_tensor, value_head.conv_value);
  nv::ITensor* gpool_v = BuildGlobalPool(network, conv_v, value_head.gpool);

  // [zl, zw, q6, q16, q50]
  nv::ITensor* zq_pre =
      BuildDenseLayer(network, gpool_v, value_head.dense_outcome_pre);
  nv::IActivationLayer* zq_relu_layer =
      network->addActivation(*zq_pre, nv::ActivationType::kRELU);
  nv::ITensor* zq = BuildDenseLayer(network, zq_relu_layer->getOutput(0),
                                    value_head.dense_outcome);

  // Ownership. (N, 1, 19, 19)
  nv::ITensor* ownership =
      BuildConvLayer(network, conv_v, value_head.conv_ownership);
  nv::IShuffleLayer* squeeze_ownership_layer = AddShuffle(
      network, ownership,
      value_head.name + ":squeeze_ownership");  // (N, 1, 19, 19) -> (N, 19, 19)
  squeeze_ownership_layer->setReshapeDimensions(
      nv::Dims3(-1, BOARD_LEN, BOARD_LEN));
  nv::IActivationLayer* ownership_tanh_layer = network->addActivation(
      *squeeze_ownership_layer->getOutput(0), nv::ActivationType::kTANH);
  ownership = ownership_tanh_layer->getOutput(0);

  // Gamma. (N, 1)
  nv::ITensor* gamma;
  gamma = BuildDenseLayer(network, gpool_v, value_head.dense_gamma_pre);
  nv::IActivationLayer* gamma_relu_layer =
      network->addActivation(*gamma, nv::ActivationType::kRELU);
  gamma = BuildDenseLayer(network, gamma_relu_layer->getOutput(0),
                          value_head.dense_gamma);
  nv::IActivationLayer* gamma_softplus_layer =
      network->addActivation(*gamma, nv::ActivationType::kSOFTPLUS);
  gamma = gamma_softplus_layer->getOutput(0);

  // Convert gpool_v from (N, K,) to (N, 1, K)
  nv::Dims gpool_dims = gpool_v->getDimensions();
  nv::Dims3 gpool_expand_dims(-1, 1, gpool_dims.d[1]);
  nv::IShuffleLayer* gpool_expand_layer =
      AddShuffle(network, gpool_v, value_head.name + ":gpool_v_expand");
  gpool_expand_layer->setReshapeDimensions(gpool_expand_dims);

  // Convert gpool_v from (N, 1, K) to (N, 800, K).
  std::vector<nv::ITensor*> gpool_vs;
  for (int i = 0; i < constants::kNumScoreLogits; ++i) {
    // Tile gpool 800 times.
    gpool_vs.emplace_back(gpool_expand_layer->getOutput(0));
  }
  nv::IConcatenationLayer* gpool_tile_layer =
      network->addConcatenation(gpool_vs.data(), constants::kNumScoreLogits);
  gpool_tile_layer->setAxis(1);
  nv::ITensor* gpool_tiled = gpool_tile_layer->getOutput(0);  // (N, 800, K)

  // Constant Score Tensor (1, 800, 1).
  nv::Dims3 score_dims(1, constants::kNumScoreLogits, 1);
  nv::Weights score_weights{
      nv::DataType::kFLOAT, value_head.scores.weights.data(),
      static_cast<int64_t>(value_head.scores.weights.size())};
  nv::IConstantLayer* scores_layer = AddConstant(
      network, score_dims, score_weights, value_head.name + ":scores");
  nv::ITensor* scores = scores_layer->getOutput(0);

  // Reshape score tensor by adding it to a tensor of shape (N, 800, 1) of all
  // zeroes.
  nv::Dims3 zero_dims(1, 1, 1);
  nv::Weights zero_weights{nv::DataType::kFLOAT, &kZero, 1};
  nv::ITensor* zero =
      AddConstant(network, zero_dims, zero_weights, value_head.name + ":zeroes")
          ->getOutput(0);
  nv::ITensor* scores_runtime_shape =
      StaticShapeWithBatch(network, gpool_tiled, kScoreDimsPartial, 2);
  nv::IResizeLayer* zero_resize = network->addResize(*zero);
  zero_resize->setInput(1, *scores_runtime_shape);
  nv::ITensor* zeroes = zero_resize->getOutput(0);
  scores =
      network->addElementWise(*scores, *zeroes, nv::ElementWiseOperation::kSUM)
          ->getOutput(0);

  // Concat scores, from (N, 800, K) to (N, 800, K+1).
  nv::ITensor* gpool_score_arr[] = {gpool_tiled, scores};
  nv::IConcatenationLayer* concat_score_layer =
      network->addConcatenation(gpool_score_arr, 2);
  concat_score_layer->setAxis(2);
  nv::ITensor* gpool_scores = concat_score_layer->getOutput(0);

  // Flatten to prepare for dense layer. (N, 800, K+1) -> (N*800, K+1)
  nv::IShuffleLayer* gpool_scores_flatten = AddShuffle(
      network, gpool_scores, value_head.name + ":gpool_scores_flatten");
  gpool_scores_flatten->setReshapeDimensions(
      nv::Dims2(-1, gpool_dims.d[1] + 1));
  gpool_scores = gpool_scores_flatten->getOutput(0);

  // First dense layer. (N*800, K+1) -> (N*800, K)
  nv::ITensor* score_pre = BuildDenseLayer(
      network, gpool_scores,
      value_head.dense_score_pre);  // (N*800, K+1) -> (N*800, K)
  nv::IActivationLayer* scores_relu_layer =
      network->addActivation(*score_pre, nv::ActivationType::kRELU);

  // Second dense.
  nv::ITensor* score_logits =
      BuildDenseLayer(network, scores_relu_layer->getOutput(0),
                      value_head.dense_score);  // (N*800, K) -> (N*800, 1)

  // Reshape. (N*800, 1) -> (N, 800)
  nv::IShuffleLayer* squeeze_score_layer = AddShuffle(
      network, score_logits,
      value_head.name + ":squeeze_score_layer");  // (N, 800, 1) -> (N, 800)
  squeeze_score_layer->setReshapeDimensions(
      nv::Dims2(-1, constants::kNumScoreLogits));

  // Scale by gamma.
  nv::IElementWiseLayer* gamma_scale_layer =
      network->addElementWise(*gamma, *squeeze_score_layer->getOutput(0),
                              nv::ElementWiseOperation::kPROD);
  score_logits = gamma_scale_layer->getOutput(0);

  // Final score distribution.
  nv::ISoftMaxLayer* score_softmax_layer = network->addSoftMax(*score_logits);
  nv::ITensor* score_probs = score_softmax_layer->getOutput(0);

  return ValueHeadOutput{zq, ownership, score_probs};
}

void BuildFromModel(nv::INetworkDefinition* network, Model* model_arch) {
  // Sanity check board length.
  CHECK(model_arch->board_len == BOARD_LEN);

  // Input Planes.
  nv::ITensor* input_planes = network->addInput(
      input::kPlanesName, nv::DataType::kFLOAT,
      nv::Dims4(-1, model_arch->num_input_planes, BOARD_LEN, BOARD_LEN));

  // Input Features.
  nv::Dims input_features_dims = nv::Dims2(-1, model_arch->num_input_features);
  nv::ITensor* input_features = network->addInput(
      input::kFeaturesName, nv::DataType::kFLOAT, input_features_dims);

  // Init Conv. (N, 13, 19, 19) -> (N, C, 19, 19)
  nv::ITensor* init_conv_output =
      BuildConvBlock(network, input_planes, model_arch->init_conv);

  // Init Game State. (N, 7) -> (N, C)
  nv::ITensor* init_game_state_output =
      BuildDenseLayer(network, input_features, model_arch->init_game_state);

  // Reshape Game State. (N, C) -> (N, C, 1, 1)
  nv::IShuffleLayer* reshape_state_layer =
      network->addShuffle(*init_game_state_output);
  reshape_state_layer->setReshapeDimensions(
      nv::Dims4(-1, model_arch->num_channels, 1, 1));
  nv::ITensor* init_conv_biases = reshape_state_layer->getOutput(0);

  // Bias Init Conv.
  nv::IElementWiseLayer* add_biases_layer = network->addElementWise(
      *init_conv_output, *init_conv_biases, nv::ElementWiseOperation::kSUM);
  nv::ITensor* trunk_input = add_biases_layer->getOutput(0);

  // Trunk.
  nv::ITensor* trunk_output =
      BuildTrunk(network, trunk_input, model_arch->trunk);

  // Policy Head.
  PolicyHeadOutput policy_head_output =
      BuildPolicyHead(network, trunk_output, model_arch->policy_head,
                      model_arch->num_head_channels);

  // Value Head.
  ValueHeadOutput value_head_output = BuildValueHead(
      network, trunk_output, model_arch->value_head,
      model_arch->num_head_channels, model_arch->num_value_channels);

  network->markOutput(*policy_head_output.policy_combined);
  network->markOutput(*value_head_output.zq);
  network->markOutput(*value_head_output.ownership);
  network->markOutput(*value_head_output.score_probs);

  policy_head_output.policy_combined->setName(output::kPolicyCombinedName);
  policy_head_output.policy_combined->setType(nv::DataType::kFLOAT);
  value_head_output.zq->setName(output::kZqName);
  value_head_output.zq->setType(nv::DataType::kFLOAT);
  value_head_output.ownership->setName(output::kOwnershipName);
  value_head_output.ownership->setType(nv::DataType::kFLOAT);
  value_head_output.score_probs->setName(output::kScoreName);
  value_head_output.score_probs->setType(nv::DataType::kFLOAT);
}
}  // namespace

nv::IHostMemory* BuildEngine(Model* model_arch, size_t batch_size) {
  nv::IBuilder* builder = nv::createInferBuilder(logger());
  nv::INetworkDefinition* network = builder->createNetworkV2(
      1U << static_cast<uint32_t>(
          nv::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

  BuildFromModel(network, model_arch);

  // Build Optimization Profile.
  nv::IOptimizationProfile* profile = builder->createOptimizationProfile();

  // Min
  profile->setDimensions(
      input::kPlanesName, nv::OptProfileSelector::kMIN,
      nv::Dims4(1, model_arch->num_input_planes, BOARD_LEN, BOARD_LEN));
  profile->setDimensions(input::kFeaturesName, nv::OptProfileSelector::kMIN,
                         nv::Dims2(1, model_arch->num_input_features));

  // Opt
  profile->setDimensions(input::kPlanesName, nv::OptProfileSelector::kOPT,
                         nv::Dims4(batch_size, model_arch->num_input_planes,
                                   BOARD_LEN, BOARD_LEN));
  profile->setDimensions(input::kFeaturesName, nv::OptProfileSelector::kOPT,
                         nv::Dims2(batch_size, model_arch->num_input_features));

  // Max
  profile->setDimensions(input::kPlanesName, nv::OptProfileSelector::kMAX,
                         nv::Dims4(batch_size, model_arch->num_input_planes,
                                   BOARD_LEN, BOARD_LEN));
  profile->setDimensions(input::kFeaturesName, nv::OptProfileSelector::kMAX,
                         nv::Dims2(batch_size, model_arch->num_input_features));

  // Create Config.
  nv::IBuilderConfig* config = builder->createBuilderConfig();
  config->addOptimizationProfile(profile);
  logger().log(
      Logger::Severity::kINFO,
      "Has Fast FP16? " + std::to_string(builder->platformHasFastFp16()));
  logger().log(
      Logger::Severity::kINFO,
      "Has Fast INT8? " + std::to_string(builder->platformHasFastInt8()));
  if (builder->platformHasFastFp16()) {
    config->setFlag(nv::BuilderFlag::kFP16);
  }

  std::unique_ptr<Int8Calibrator> calibrator = Int8Calibrator::Create(
      batch_size, "/tmp/p3achygo/val.tfrecord.zz", "/tmp/int8_cache.trt");
  if (builder->platformHasFastInt8()) {
    config->setFlag(nv::BuilderFlag::kINT8);
    config->setInt8Calibrator(calibrator.get());
    config->setCalibrationProfile(profile);
  }
  nv::IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);

  delete config;
  delete network;
  delete builder;

  return engine;
}

void WriteEngineToDisk(nvinfer1::IHostMemory* serialized_engine,
                       std::string path) {
  std::ofstream file(path, std::ios::binary);

  if (!file) {
    LOG(ERROR) << "Cannot open engine file for writing: " << path << std::endl;
    return;
  }

  file.write(static_cast<char*>(serialized_engine->data()),
             serialized_engine->size());
  file.close();
}

}  // namespace trt
}  // namespace nn
