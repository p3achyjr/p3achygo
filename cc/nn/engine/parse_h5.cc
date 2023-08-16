#include "cc/nn/engine/parse_h5.h"

#include <hdf5/serial/H5Cpp.h>

#include <algorithm>
#include <regex>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/nn/engine/model_arch.h"

namespace nn {

// Keep in sync with //python/export_weights.py
namespace tags {
namespace model {
static constexpr char kModel[] = "model";
static constexpr char kInitConv[] = "init_conv";
static constexpr char kInitGameState[] = "init_game_state";
static constexpr char kTrunk[] = "trunk";
static constexpr char kPolicyHead[] = "policy_head";
static constexpr char kValueHead[] = "value_head";
};  // namespace model

namespace metadata {
static constexpr char kNLayers[] = "nlayers";
static constexpr char kNChannels[] = "nchannels";
static constexpr char kNBtlChannels[] = "nbtl_channels";
static constexpr char kNHeadChannels[] = "nhead_channels";
static constexpr char kNValChannels[] = "nval_channels";
static constexpr char kNBtlLength[] = "nbtl";
static constexpr char kNInputPlanes[] = "ninput_planes";
static constexpr char kNInputFeatures[] = "ninput_features";
static constexpr char kOrder[] = "order";
}  // namespace metadata

namespace policy_head {
static constexpr char kConvPolicy[] = "conv_policy";
static constexpr char kConvGlobal[] = "conv_global";
static constexpr char kConvMoves[] = "conv_moves";
static constexpr char kDensePass[] = "dense_pass";
}  // namespace policy_head

namespace value_head {
static constexpr char kConvValue[] = "conv_value";
static constexpr char kDenseOutcomePre[] = "dense_outcome_pre";
static constexpr char kDenseOutcome[] = "dense_outcome";
static constexpr char kOwnership[] = "ownership";
static constexpr char kDenseGammaPre[] = "dense_gamma_pre";
static constexpr char kDenseGamma[] = "dense_gamma";
static constexpr char kActivationGamma[] = "activation_gamma";
static constexpr char kDenseScoresPre[] = "dense_scores_pre";
static constexpr char kDenseScores[] = "dense_scores";
static constexpr char kScores[] = "scores";
}  // namespace value_head

namespace block {
static constexpr char kBottleneckRes[] = "bottleneck_res";
static constexpr char kBroadcastRes[] = "broadcast_res";
static constexpr char kBroadcast[] = "broadcast";
static constexpr char kGlobalPool[] = "global_pool";
static constexpr char kGlobalPoolBias[] = "global_pool_bias";
static constexpr char kConvBlock[] = "conv_block";
}  // namespace block

namespace layer {
static constexpr char kConv[] = "conv";
static constexpr char kDense[] = "dense";
static constexpr char kBatchNorm[] = "batch_norm";
static constexpr char kActivation[] = "activation";
}  // namespace layer

namespace dataset {
static constexpr char kKernel[] = "kernel";
static constexpr char kBias[] = "bias";
static constexpr char kMovingMean[] = "moving_mean";
static constexpr char kMovingVariance[] = "moving_variance";
static constexpr char kBeta[] = "beta";
static constexpr char kEpsilon[] = "epsilon";
}  // namespace dataset
}  // namespace tags

namespace h5_activations {
// Keep in sync with //python/export_weights.py
static constexpr int kLinear = 0;
static constexpr int kRelu = 1;
static constexpr int kTanh = 2;
static constexpr int kSoftplus = 3;
}  // namespace h5_activations

namespace {
using namespace model_arch;
bool ExistsGroup(const H5::Group& group, const std::string& name) {
  return H5Lexists(group.getId(), name.c_str(), H5P_DEFAULT);
}

std::string GetFullName(const H5::Group& group) { return group.getObjName(); }

int ReadIntAttr(const H5::Attribute& attr) {
  H5::DataType datatype = attr.getDataType();
  CHECK(datatype == H5::PredType::NATIVE_INT ||
        datatype == H5::PredType::NATIVE_INT64 ||
        datatype == H5::PredType::NATIVE_UINT ||
        datatype == H5::PredType::NATIVE_UINT64);

  int data;
  attr.read(datatype, &data);

  return data;
}

float ReadFloatAttr(const H5::Attribute& attr) {
  H5::DataType datatype = attr.getDataType();
  CHECK(datatype == H5::PredType::NATIVE_FLOAT ||
        datatype == H5::PredType::NATIVE_DOUBLE);

  float data;
  attr.read(datatype, &data);

  return data;
}

std::vector<std::string> GetOrderedGroupKeys(const H5::Group& group) {
  hsize_t num_objs = group.getNumObjs();
  std::vector<std::string> keys;
  for (hsize_t i = 0; i < num_objs; ++i) {
    keys.push_back(group.getObjnameByIdx(i));
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

std::string GetBaseKey(std::string key) {
  static const std::regex ordered_key_re("\\d+:(.+)");

  std::smatch match;
  if (!std::regex_match(key, match, ordered_key_re)) {
    return key;
  }

  CHECK(match.size() == 2);
  return match[1].str();
}

std::vector<int> GetDataspaceShape(const H5::DataSpace& dataspace) {
  int ndims = dataspace.getSimpleExtentNdims();
  std::vector<hsize_t> hdims(ndims);
  dataspace.getSimpleExtentDims(hdims.data(), NULL);

  std::vector<int> dims(ndims);
  std::transform(hdims.begin(), hdims.end(), dims.begin(),
                 [](const hsize_t& n) { return static_cast<int>(n); });
  return dims;
}

int GetNumElems(const std::vector<int>& shape) {
  int num_elems = 1;
  for (const auto& d : shape) {
    num_elems *= d;
  }

  return num_elems;
}

ActivationKind ConvertActivation(int h5_act) {
  if (h5_act == h5_activations::kLinear) {
    return ActivationKind::kLinear;
  } else if (h5_act == h5_activations::kRelu) {
    return ActivationKind::kRelu;
  } else if (h5_act == h5_activations::kTanh) {
    return ActivationKind::kTanh;
  } else if (h5_act == h5_activations::kSoftplus) {
    return ActivationKind::kSoftplus;
  }

  LOG(FATAL) << "Unknown Activation Type `" << h5_act << "`.";
}
}  // namespace

TensorDesc ParseTensor(H5::DataSet ds) {
  H5::DataSpace dataspace = ds.getSpace();
  TensorDesc desc;
  desc.shape = GetDataspaceShape(dataspace);
  desc.weights.resize(GetNumElems(desc.shape));
  ds.read(desc.weights.data(), H5::PredType::NATIVE_FLOAT);
  return desc;
}

Layer::Property ParseWeightsProperty(H5::DataSet weights_ds) {
  Layer::Property prop;
  prop.kind = Layer::Property::Kind::kWeights;
  prop.weights = ParseTensor(weights_ds);
  return prop;
}

Layer ParseConvLayer(H5::Group conv_group) {
  Layer layer;
  layer.name = GetFullName(conv_group);
  layer.kind = Layer::Kind::kConv;
  layer.properties[Layer::kKernel] =
      ParseWeightsProperty(conv_group.openDataSet(tags::dataset::kKernel));
  layer.properties[Layer::kBias] =
      ParseWeightsProperty(conv_group.openDataSet(tags::dataset::kBias));

  CHECK(layer.properties[Layer::kKernel].weights->shape.size() == 4);
  CHECK(layer.properties[Layer::kBias].weights->shape.size() == 1);
  CHECK(layer.properties[Layer::kKernel].weights->shape[3] ==
        layer.properties[Layer::kBias].weights->shape[0]);

  return layer;
}

Layer ParseDenseLayer(H5::Group dense_group) {
  Layer layer;
  layer.name = GetFullName(dense_group);
  layer.kind = Layer::Kind::kDense;
  layer.properties[Layer::kKernel] =
      ParseWeightsProperty(dense_group.openDataSet(tags::dataset::kKernel));
  layer.properties[Layer::kBias] =
      ParseWeightsProperty(dense_group.openDataSet(tags::dataset::kBias));

  CHECK(layer.properties[Layer::kKernel].weights->shape.size() == 2);
  CHECK(layer.properties[Layer::kBias].weights->shape.size() == 1);
  CHECK(layer.properties[Layer::kKernel].weights->shape[1] ==
        layer.properties[Layer::kBias].weights->shape[0]);

  return layer;
}

Layer ParseBNLayer(H5::Group bn_group) {
  Layer layer;
  layer.name = GetFullName(bn_group);
  layer.kind = Layer::Kind::kBatchNorm;
  layer.properties[Layer::kBeta] =
      ParseWeightsProperty(bn_group.openDataSet(tags::dataset::kBeta));
  layer.properties[Layer::kMovingMean] =
      ParseWeightsProperty(bn_group.openDataSet(tags::dataset::kMovingMean));
  layer.properties[Layer::kMovingVariance] = ParseWeightsProperty(
      bn_group.openDataSet(tags::dataset::kMovingVariance));
  layer.properties[Layer::kMovingVariance] = ParseWeightsProperty(
      bn_group.openDataSet(tags::dataset::kMovingVariance));

  float eps = ReadFloatAttr(bn_group.openAttribute(tags::dataset::kEpsilon));
  layer.properties[Layer::kEpsilon] = Layer::Property{
      Layer::Property::Kind::kScalar, std::nullopt, std::nullopt, eps};

  CHECK(layer.properties[Layer::kBeta].weights->shape.size() == 1);
  CHECK(layer.properties[Layer::kMovingMean].weights->shape.size() == 1);
  CHECK(layer.properties[Layer::kMovingVariance].weights->shape.size() == 1);
  CHECK(layer.properties[Layer::kBeta].weights->shape[0] ==
            layer.properties[Layer::kMovingMean].weights->shape[0] &&
        layer.properties[Layer::kBeta].weights->shape[0] ==
            layer.properties[Layer::kMovingVariance].weights->shape[0]);
  return layer;
}

Layer ParseActivationLayer(H5::Group act_group) {
  Layer layer;
  layer.name = GetFullName(act_group);
  layer.kind = Layer::Kind::kActivation;

  H5::Attribute act_attr = act_group.openAttribute(tags::layer::kActivation);
  ActivationKind activation = ConvertActivation(ReadIntAttr(act_attr));

  layer.properties[Layer::kActivation] =
      Layer::Property{Layer::Property::Kind::kActivation, std::nullopt,
                      activation, std::nullopt};
  return layer;
}

Layer ParseLayer(H5::Group parent_group, std::string layer_group_key) {
  std::string layer_key = GetBaseKey(layer_group_key);
  Layer layer;
  H5::Group layer_group = parent_group.openGroup(layer_group_key.c_str());

  if (layer_key == tags::layer::kConv) {
    layer = ParseConvLayer(layer_group);
  } else if (layer_key == tags::layer::kDense) {
    layer = ParseDenseLayer(layer_group);
  } else if (layer_key == tags::layer::kBatchNorm) {
    layer = ParseBNLayer(layer_group);
  } else if (layer_key == tags::layer::kActivation) {
    layer = ParseActivationLayer(layer_group);
  } else {
    LOG(FATAL) << "Unknown Layer Key: `" << layer_key << "`.";
  }

  return layer;
}

void FillBlockLayers(Block& block, H5::Group block_group) {
  std::vector<std::string> ordered_groups = GetOrderedGroupKeys(block_group);
  for (const auto& layer_group_key : ordered_groups) {
    block.layers.emplace_back(ParseLayer(block_group, layer_group_key));
  }
}

Block ParseConvBlock(H5::Group conv_block_group) {
  Block conv_block;
  conv_block.name = GetFullName(conv_block_group);
  conv_block.kind = Block::Kind::kConv;
  FillBlockLayers(conv_block, conv_block_group);
  CHECK(conv_block.layers.size() == 3);
  return conv_block;
}

Block ParseBroadcastBlock(H5::Group broadcast_block_group) {
  Block broadcast_block;
  broadcast_block.name = GetFullName(broadcast_block_group);
  broadcast_block.kind = Block::Kind::kBroadcast;
  FillBlockLayers(broadcast_block, broadcast_block_group);
  CHECK(broadcast_block.layers.size() == 2);
  return broadcast_block;
}

Block ParseBlock(H5::Group parent_group, std::string block_group_key) {
  std::string block_key = GetBaseKey(block_group_key);
  Block block;
  H5::Group block_group = parent_group.openGroup(block_group_key);
  if (block_key == tags::block::kConvBlock) {
    block = ParseConvBlock(block_group);
  } else if (block_key == tags::block::kBroadcast) {
    block = ParseBroadcastBlock(block_group);
  }

  return block;
}

void FillResidualBlockLayers(ResidualBlock& res_block,
                             H5::Group res_block_group) {
  std::vector<std::string> ordered_groups =
      GetOrderedGroupKeys(res_block_group);
  for (const auto& block_group_key : ordered_groups) {
    res_block.blocks.emplace_back(ParseBlock(res_block_group, block_group_key));
  }
}

ResidualBlock ParseBottleneckResBlock(H5::Group bottleneck_res_group) {
  ResidualBlock bottleneck_res_block;
  bottleneck_res_block.name = GetFullName(bottleneck_res_group);
  bottleneck_res_block.kind = ResidualBlock::Kind::kBottleneck;
  FillResidualBlockLayers(bottleneck_res_block, bottleneck_res_group);
  return bottleneck_res_block;
}

ResidualBlock ParseBroadcastResBlock(H5::Group broadcast_res_group) {
  ResidualBlock broadcast_res_block;
  broadcast_res_block.name = GetFullName(broadcast_res_group);
  broadcast_res_block.kind = ResidualBlock::Kind::kBroadcast;
  FillResidualBlockLayers(broadcast_res_block, broadcast_res_group);
  return broadcast_res_block;
}

ResidualBlock ParseResidualBlock(H5::Group parent_group,
                                 std::string res_block_group_key) {
  std::string res_block_key = GetBaseKey(res_block_group_key);
  ResidualBlock res_block;
  H5::Group res_block_group = parent_group.openGroup(res_block_group_key);
  if (res_block_key == tags::block::kBottleneckRes) {
    res_block = ParseBottleneckResBlock(res_block_group);
  } else if (res_block_key == tags::block::kBroadcastRes) {
    res_block = ParseBroadcastResBlock(res_block_group);
  } else {
    LOG(FATAL) << "Unknown Residual Block Key: `" << res_block_key << "`.";
  }

  return res_block;
}

GlobalPool ParseGlobalPool(H5::Group gpool_group) {
  GlobalPool gpool;
  gpool.name = GetFullName(gpool_group);
  gpool.c = ReadIntAttr(gpool_group.openAttribute("c"));
  gpool.h = ReadIntAttr(gpool_group.openAttribute("h"));
  gpool.w = ReadIntAttr(gpool_group.openAttribute("w"));
  return gpool;
}

GlobalPoolBias ParseGlobalPoolBias(H5::Group gpool_bias_group) {
  GlobalPoolBias gpool_bias;
  gpool_bias.name = GetFullName(gpool_bias_group);
  gpool_bias.batch_norm_g =
      ParseLayer(gpool_bias_group, tags::layer::kBatchNorm);
  gpool_bias.gpool =
      ParseGlobalPool(gpool_bias_group.openGroup(tags::block::kGlobalPool));
  gpool_bias.dense = ParseLayer(gpool_bias_group, tags::layer::kDense);
  return gpool_bias;
}

Trunk ParseTrunk(H5::Group trunk_group) {
  Trunk trunk;
  trunk.name = GetFullName(trunk_group);

  std::vector<std::string> ordered_groups = GetOrderedGroupKeys(trunk_group);
  for (const auto& group_key : ordered_groups) {
    LOG(INFO) << "  Parsing Trunk/" << group_key << "...";
    trunk.res_blocks.emplace_back(ParseResidualBlock(trunk_group, group_key));
  }

  return trunk;
}

PolicyHead ParsePolicyHead(H5::Group policy_head_group) {
  PolicyHead policy_head;
  policy_head.name = GetFullName(policy_head_group);

  LOG(INFO) << "  Parsing Policy Head/Init Policy Conv...";
  policy_head.conv_policy =
      ParseLayer(policy_head_group.openGroup(tags::policy_head::kConvPolicy),
                 tags::layer::kConv);

  LOG(INFO) << "  Parsing Policy Head/Init Global Conv...";
  policy_head.conv_global =
      ParseLayer(policy_head_group.openGroup(tags::policy_head::kConvGlobal),
                 tags::layer::kConv);

  LOG(INFO) << "  Parsing Policy Head/Global Pool Bias...";
  policy_head.gpool_bias = ParseGlobalPoolBias(
      policy_head_group.openGroup(tags::block::kGlobalPoolBias));

  LOG(INFO) << "  Parsing Policy Head/Batch Norm...";
  policy_head.batch_norm =
      ParseLayer(policy_head_group, tags::layer::kBatchNorm);

  LOG(INFO) << "  Parsing Policy Head/Conv Board Moves...";
  policy_head.conv_moves =
      ParseLayer(policy_head_group.openGroup(tags::policy_head::kConvMoves),
                 tags::layer::kConv);

  LOG(INFO) << "  Parsing Policy Head/Dense Pass...";
  policy_head.dense_pass =
      ParseLayer(policy_head_group.openGroup(tags::policy_head::kDensePass),
                 tags::layer::kDense);
  return policy_head;
}

ValueHead ParseValueHead(H5::Group value_head_group) {
  ValueHead value_head;
  value_head.name = GetFullName(value_head_group);

  LOG(INFO) << "  Parsing Value Head/Init Value Conv...";
  value_head.conv_value =
      ParseLayer(value_head_group.openGroup(tags::value_head::kConvValue),
                 tags::layer::kConv);

  LOG(INFO) << "  Parsing Value Head/GPool...";
  value_head.gpool =
      ParseGlobalPool(value_head_group.openGroup(tags::block::kGlobalPool));

  LOG(INFO) << "  Parsing Value Head/Dense Outcome Pre...";
  value_head.dense_outcome_pre =
      ParseLayer(value_head_group.openGroup(tags::value_head::kDenseOutcomePre),
                 tags::layer::kDense);

  LOG(INFO) << "  Parsing Value Head/Dense Outcome...";
  value_head.dense_outcome =
      ParseLayer(value_head_group.openGroup(tags::value_head::kDenseOutcome),
                 tags::layer::kDense);

  LOG(INFO) << "  Parsing Value Head/Conv Ownership...";
  value_head.conv_ownership =
      ParseLayer(value_head_group.openGroup(tags::value_head::kOwnership),
                 tags::layer::kConv);

  LOG(INFO) << "  Parsing Value Head/Dense Gamma Pre...";
  value_head.dense_gamma_pre =
      ParseLayer(value_head_group.openGroup(tags::value_head::kDenseGammaPre),
                 tags::layer::kDense);

  LOG(INFO) << "  Parsing Value Head/Dense Gamma...";
  value_head.dense_gamma =
      ParseLayer(value_head_group.openGroup(tags::value_head::kDenseGamma),
                 tags::layer::kDense);

  LOG(INFO) << "  Parsing Value Head/Scores...";
  value_head.scores =
      ParseTensor(value_head_group.openDataSet(tags::value_head::kScores));

  LOG(INFO) << "  Parsing Value Head/Dense Scores Pre...";
  value_head.dense_score_pre =
      ParseLayer(value_head_group.openGroup(tags::value_head::kDenseScoresPre),
                 tags::layer::kDense);

  LOG(INFO) << "  Parsing Value Head/Dense Scores...";
  value_head.dense_score =
      ParseLayer(value_head_group.openGroup(tags::value_head::kDenseScores),
                 tags::layer::kDense);
  return value_head;
}

std::unique_ptr<Model> ParseModel(H5::Group model_group) {
  std::unique_ptr<Model> model = std::make_unique<Model>();
  model->name = GetFullName(model_group);
  model->num_input_planes =
      ReadIntAttr(model_group.openAttribute(tags::metadata::kNInputPlanes));
  model->num_input_features =
      ReadIntAttr(model_group.openAttribute(tags::metadata::kNInputFeatures));
  model->num_blocks =
      ReadIntAttr(model_group.openAttribute(tags::metadata::kNLayers));
  model->num_channels =
      ReadIntAttr(model_group.openAttribute(tags::metadata::kNChannels));
  model->num_bottleneck_channels =
      ReadIntAttr(model_group.openAttribute(tags::metadata::kNBtlChannels));
  model->num_head_channels =
      ReadIntAttr(model_group.openAttribute(tags::metadata::kNHeadChannels));
  model->num_value_channels =
      ReadIntAttr(model_group.openAttribute(tags::metadata::kNValChannels));
  model->bottleneck_length =
      ReadIntAttr(model_group.openAttribute(tags::metadata::kNBtlLength));

  LOG(INFO) << "Parsing Init Conv...";
  H5::Group init_conv_block_group =
      model_group.openGroup(tags::model::kInitConv)
          .openGroup(tags::block::kConvBlock);
  model->init_conv = ParseConvBlock(init_conv_block_group);

  LOG(INFO) << "Parsing Init Game State...";
  model->init_game_state = ParseLayer(
      model_group.openGroup(tags::model::kInitGameState), tags::layer::kDense);

  LOG(INFO) << "Parsing Trunk...";
  model->trunk = ParseTrunk(model_group.openGroup(tags::model::kTrunk));

  LOG(INFO) << "Parsing Policy Head...";
  model->policy_head =
      ParsePolicyHead(model_group.openGroup(tags::model::kPolicyHead));

  LOG(INFO) << "Parsing Value Head...";
  model->value_head =
      ParseValueHead(model_group.openGroup(tags::model::kValueHead));

  return model;
}

std::unique_ptr<Model> ParseFromH5(std::string path) {
  H5::H5File file(path, H5F_ACC_RDONLY);
  CHECK(ExistsGroup(file, tags::model::kModel))
      << "Missing Top-Level Tag `Model`";

  std::unique_ptr<Model> model =
      ParseModel(file.openGroup(tags::model::kModel));

  return model;
}

}  // namespace nn
