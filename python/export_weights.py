'''
Script to export a model's weights into a custom H5 format.

Allows us to write custom model backends.
'''
from __future__ import annotations

import numpy as np
import h5py
import tensorflow as tf

from model import *


class ModelTags:
  # Top Level Descriptors.
  MODEL = 'model'
  INIT_CONV = 'init_conv'
  INIT_GAME_STATE = 'init_game_state'
  TRUNK = 'trunk'
  POLICY_HEAD = 'policy_head'
  VALUE_HEAD = 'value_head'


# Keep in sync with CC.
class MetadataTags:

  # Metadata
  NLAYERS = 'nlayers'
  NCHANNELS = 'nchannels'
  NBTL_CHANNELS = 'nbtl_channels'
  NHEAD_CHANNELS = 'nhead_channels'
  NVAL_CHANNELS = 'nval_channels'
  NBTL = 'nbtl'
  NINPUT_PLANES = 'ninput_planes'
  NINPUT_FEATURES = 'ninput_features'
  ORDER = 'order'  # Generic layer ordering tag.


class PolicyHeadTags:
  # Policy Head Tags
  CONV_POLICY = 'conv_policy'
  CONV_GLOBAL = 'conv_global'
  CONV_MOVES = 'conv_moves'
  DENSE_PASS = 'dense_pass'


class ValueHeadTags:
  # Value Head Tags
  CONV_VALUE = 'conv_value'
  DENSE_OUTCOME_PRE = 'dense_outcome_pre'
  DENSE_OUTCOME = 'dense_outcome'
  OWNERSHIP = 'ownership'
  DENSE_GAMMA_PRE = 'dense_gamma_pre'
  DENSE_GAMMA = 'dense_gamma'
  ACTIVATION_GAMMA = 'activation_gamma'
  DENSE_SCORES_PRE = 'dense_scores_pre'
  DENSE_SCORES = 'dense_scores'
  SCORES = 'scores'


class BlockTags:
  # Block Types.
  BOTTLENECK_RES = 'bottleneck_res'
  BROADCAST_RES = 'broadcast_res'
  BROADCAST = 'broadcast'
  GLOBAL_POOL = 'global_pool'
  GLOBAL_POOL_BIAS = 'global_pool_bias'
  CONV_BLOCK = 'conv_block'


class LayerTags:
  # Layer Types
  CONV = 'conv'
  DENSE = 'dense'
  BATCH_NORM = 'batch_norm'
  ACTIVATION = 'activation'


class DatasetTags:
  # Dataset Tags
  KERNEL = 'kernel'
  BIAS = 'bias'
  MOVING_MEAN = 'moving_mean'
  MOVING_VARIANCE = 'moving_variance'
  BETA = 'beta'
  EPSILON = 'epsilon'


class Activations:
  LINEAR = 0
  RELU = 1
  TANH = 2
  SOFTPLUS = 3

  MAPPING = {
      'linear': LINEAR,
      'relu': RELU,
      'tanh': TANH,
      'softplus': SOFTPLUS,
  }


def create_top_level_model_group(model: P3achyGoModel, h5: h5py.File):
  model_group = h5.create_group(ModelTags.MODEL)
  model_group.attrs[MetadataTags.NINPUT_PLANES] = model.num_input_planes
  model_group.attrs[MetadataTags.NINPUT_FEATURES] = model.num_input_features
  model_group.attrs[MetadataTags.NLAYERS] = model.num_blocks
  model_group.attrs[MetadataTags.NCHANNELS] = model.num_channels
  model_group.attrs[MetadataTags.NBTL_CHANNELS] = model.num_bottleneck_channels
  model_group.attrs[MetadataTags.NHEAD_CHANNELS] = model.num_head_channels
  model_group.attrs[MetadataTags.NVAL_CHANNELS] = model.c_val
  model_group.attrs[MetadataTags.NBTL] = model.bottleneck_length - 2

  return model_group


def create_trunk_group(model: P3achyGoModel, model_group: h5py.Group):
  trunk_group = model_group.create_group(ModelTags.TRUNK)
  trunk_group.attrs[MetadataTags.NLAYERS] = len(model.blocks)
  return trunk_group


def create_ordered_group(parent_group: h5py.Group, tag: str, order: int):
  tag = f'{order:02d}:{tag}'
  group = parent_group.create_group(tag)
  return group


def fill_conv_layer(conv_layer: tf.keras.layers.Conv2D, group: h5py.Group):
  assert isinstance(
      conv_layer,
      tf.keras.layers.Conv2D), f'Invalid Layer Type: {type(conv_layer)}'
  group.create_dataset(DatasetTags.KERNEL,
                       data=conv_layer.kernel.numpy(),
                       dtype=np.float32)
  group.create_dataset(DatasetTags.BIAS,
                       data=conv_layer.bias.numpy(),
                       dtype=np.float32)


def fill_dense_layer(dense_layer: tf.keras.layers.Dense, group: h5py.Group):
  assert isinstance(
      dense_layer,
      tf.keras.layers.Dense), f'Invalid Layer Type: {type(dense_layer)}'
  group.create_dataset(DatasetTags.KERNEL,
                       data=dense_layer.kernel.numpy(),
                       dtype=np.float32)
  group.create_dataset(DatasetTags.BIAS,
                       data=dense_layer.bias.numpy(),
                       dtype=np.float32)


def fill_bn_layer(bn_layer: tf.keras.layers.BatchNormalization,
                  group: h5py.Group):
  assert isinstance(bn_layer, tf.keras.layers.BatchNormalization
                   ), f'Invalid Layer Type: {type(bn_layer)}'
  group.create_dataset(DatasetTags.MOVING_MEAN,
                       data=bn_layer.moving_mean,
                       dtype=np.float32)
  group.create_dataset(DatasetTags.MOVING_VARIANCE,
                       data=bn_layer.moving_variance.numpy(),
                       dtype=np.float32)
  group.create_dataset(DatasetTags.BETA,
                       data=bn_layer.beta.numpy(),
                       dtype=np.float32)
  group.attrs[DatasetTags.EPSILON] = bn_layer.epsilon


def fill_act_layer(activation: str, group: h5py.Group):
  assert activation in Activations.MAPPING.keys(
  ), f'Unknown Activation: {activation}'
  group.attrs[LayerTags.ACTIVATION] = Activations.MAPPING[activation]


def fill_conv_block(conv_block: ConvBlock, group: h5py.Group):
  fill_conv_layer(conv_block.conv,
                  create_ordered_group(group, LayerTags.CONV, order=0))
  fill_bn_layer(conv_block.batch_norm,
                create_ordered_group(group, LayerTags.BATCH_NORM, order=1))
  fill_act_layer(tf.keras.activations.serialize(conv_block.activation),
                 create_ordered_group(group, LayerTags.ACTIVATION, order=2))


def fill_init_conv(model: P3achyGoModel, group: h5py.Group):
  fill_conv_block(model.init_board_conv,
                  group.create_group(BlockTags.CONV_BLOCK))


def fill_init_game_state(model: P3achyGoModel, group: h5py.Group):
  fill_dense_layer(model.init_game_layer, group.create_group(LayerTags.DENSE))


def fill_bottleneck_block(block: BottleneckResidualConvBlock,
                          group: h5py.Group):
  group.attrs[MetadataTags.NLAYERS] = len(block.blocks)
  for i, block in enumerate(block.blocks):
    assert isinstance(
        block, ConvBlock
    ), f'Invalid Block Type in BottleneckResidualConvBlock: {i}, {type(block)}'
    fill_conv_block(block,
                    create_ordered_group(group, BlockTags.CONV_BLOCK, order=i))


def fill_broadcast_block(block: BroadcastResidualBlock, group: h5py.Group):

  def fill_broadcast_layer(block: BroadcastResidualBlock.Broadcast,
                           group: h5py.Group):
    fill_dense_layer(block.dense,
                     create_ordered_group(group, LayerTags.DENSE, order=0))
    fill_act_layer('relu',
                   create_ordered_group(group, LayerTags.ACTIVATION, order=1))

  group.attrs[MetadataTags.NLAYERS] = len(block.blocks)
  for i, block in enumerate(block.blocks):
    if isinstance(block, ConvBlock):
      fill_conv_block(
          block, create_ordered_group(group, BlockTags.CONV_BLOCK, order=i))
    elif isinstance(block, BroadcastResidualBlock.Broadcast):
      fill_broadcast_layer(
          block, create_ordered_group(group, BlockTags.BROADCAST, order=i))
    else:
      raise Exception(
          f'Invalid Block Type in BroadcastResidualBlock: {i}, {type(block)}')


def fill_global_pool(block: GlobalPool, group: h5py.Group):
  group.attrs['c'] = block.c
  group.attrs['h'] = block.h
  group.attrs['w'] = block.w


def fill_global_pool_bias(block: GlobalPoolBias, group: h5py.Group):
  fill_bn_layer(block.batch_norm_g, group.create_group(LayerTags.BATCH_NORM))
  fill_global_pool(block.gpool, group.create_group(BlockTags.GLOBAL_POOL))
  fill_dense_layer(block.dense, group.create_group(LayerTags.DENSE))


def fill_trunk(model: P3achyGoModel, group: h5py.Group):
  for i, block in enumerate(model.blocks):
    if isinstance(block, BottleneckResidualConvBlock):
      fill_bottleneck_block(
          block, create_ordered_group(group, BlockTags.BOTTLENECK_RES, order=i))
    elif isinstance(block, BroadcastResidualBlock):
      fill_broadcast_block(
          block, create_ordered_group(group, BlockTags.BROADCAST_RES, order=i))
    else:
      raise Exception(f'Invalid Block Type at Trunk Index {i}, {type(block)}')


def fill_policy_head(model: P3achyGoModel, group: h5py.Group):
  policy_head = model.policy_head

  conv_policy_group = group.create_group(PolicyHeadTags.CONV_POLICY)
  fill_conv_layer(policy_head.conv_p,
                  conv_policy_group.create_group(LayerTags.CONV))

  conv_global_group = group.create_group(PolicyHeadTags.CONV_GLOBAL)
  fill_conv_layer(policy_head.conv_g,
                  conv_global_group.create_group(LayerTags.CONV))

  fill_global_pool_bias(policy_head.gpool,
                        group.create_group(BlockTags.GLOBAL_POOL_BIAS))
  fill_bn_layer(policy_head.batch_norm,
                group.create_group(LayerTags.BATCH_NORM))

  conv_moves_group = group.create_group(PolicyHeadTags.CONV_MOVES)
  fill_conv_layer(policy_head.output_moves,
                  conv_moves_group.create_group(LayerTags.CONV))

  dense_pass_group = group.create_group(PolicyHeadTags.DENSE_PASS)
  fill_dense_layer(policy_head.output_pass,
                   dense_pass_group.create_group(LayerTags.DENSE))


def fill_value_head(model: P3achyGoModel, group: h5py.Group):
  value_head = model.value_head
  conv_value_group = group.create_group(ValueHeadTags.CONV_VALUE)

  # prep
  fill_conv_layer(value_head.conv,
                  conv_value_group.create_group(LayerTags.CONV))
  fill_global_pool(value_head.gpool, group.create_group(BlockTags.GLOBAL_POOL))

  # outcome
  dense_outcome_pre_group = group.create_group(ValueHeadTags.DENSE_OUTCOME_PRE)
  fill_dense_layer(value_head.outcome_q_biases,
                   dense_outcome_pre_group.create_group(LayerTags.DENSE))
  dense_outcome_group = group.create_group(ValueHeadTags.DENSE_OUTCOME)
  fill_dense_layer(value_head.outcome_q_output,
                   dense_outcome_group.create_group(LayerTags.DENSE))

  # ownership
  ownership_group = group.create_group(ValueHeadTags.OWNERSHIP)
  fill_conv_layer(value_head.conv_ownership,
                  ownership_group.create_group(LayerTags.CONV))

  # gamma
  dense_gamma_pre_group = group.create_group(ValueHeadTags.DENSE_GAMMA_PRE)
  fill_dense_layer(value_head.gamma_pre,
                   dense_gamma_pre_group.create_group(LayerTags.DENSE))
  dense_gamma_group = group.create_group(ValueHeadTags.DENSE_GAMMA)
  fill_dense_layer(value_head.gamma_output,
                   dense_gamma_group.create_group(LayerTags.DENSE))

  # score
  group.create_dataset(ValueHeadTags.SCORES, data=value_head.scores.numpy())
  dense_score_pre_group = group.create_group(ValueHeadTags.DENSE_SCORES_PRE)
  fill_dense_layer(value_head.score_pre,
                   dense_score_pre_group.create_group(LayerTags.DENSE))
  dense_score_group = group.create_group(ValueHeadTags.DENSE_SCORES)
  fill_dense_layer(value_head.score_output,
                   dense_score_group.create_group(LayerTags.DENSE))


def export_h5(model: P3achyGoModel, path: str):
  with h5py.File(path, 'w') as h5:
    model_group = create_top_level_model_group(model, h5)
    init_conv_group = model_group.create_group(ModelTags.INIT_CONV)
    init_game_state_group = model_group.create_group(ModelTags.INIT_GAME_STATE)
    trunk_group = create_trunk_group(model, model_group)
    policy_head_group = model_group.create_group(ModelTags.POLICY_HEAD)
    value_head_group = model_group.create_group(ModelTags.VALUE_HEAD)

    fill_init_conv(model, init_conv_group)
    fill_init_game_state(model, init_game_state_group)
    fill_trunk(model, trunk_group)
    fill_policy_head(model, policy_head_group)
    fill_value_head(model, value_head_group)


def export(model: P3achyGoModel, path: str):
  with tf.device('/CPU:0'):
    export_h5(model, path)
