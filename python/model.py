'''
Model definition for p3achyGo.

Input:

At move k, pass in 7 19 x 19 binary feature planes containing:

1. Location has own stone
2. Location has opponent stone
3. {k - 5}th move (one hot)
4. {k - 4}th move
5. {k - 3}rd move
6. {k - 2}nd move
7. {k - 1}st move

Output:

One (19, 19) feature plane of logits, where softmax(logits) = policy

Architecture:

We mimic the architecture in https://openreview.net/pdf?id=bERaNdoegnO.
'''

import tensorflow as tf

from model_config import ModelConfig


class ConvBlock(tf.keras.layers.Layer):
  ''' 
  Basic convolutional block.
  '''

  def __init__(self,
               output_channels: int,
               conv_size: int,
               activation=tf.keras.activations.relu,
               name=None):
    super(ConvBlock, self).__init__(name=name)
    self.conv = tf.keras.layers.Conv2D(output_channels,
                                       conv_size,
                                       activation=None,
                                       padding='same')
    self.batch_norm = tf.keras.layers.BatchNormalization(scale=False,
                                                         momentum=.999,
                                                         eps=1e-3)
    self.activation = activation

  def call(self, x):
    x = self.conv(x)
    x = self.batch_norm(x)
    x = self.activation(x)

    return x


class ResidualBlock(tf.keras.layers.Layer):
  ''' 
  Generalized residual block.

  Input:

  1. A series of operations (blocks)
  2. An activation (activation)

  ResidualBlock(x) = activation(blocks(x) + x)
  '''

  def __init__(self,
               inner_blocks: list[tf.keras.layers.Layer],
               activation=tf.keras.activations.linear,
               name=None):
    super(ResidualBlock, self).__init__(name=name)
    self.blocks = inner_blocks
    self.activation = activation

  def call(self, x):
    res = x
    for block in self.blocks:
      x = block(x)

    return self.activation(x + res)


class BottleneckResidualConvBlock(ResidualBlock):
  '''
  Bottleneck block that reduces dimension, performs inner convolutions, and
  expands dimension.

  Does so via `num_conv_stacks` number of:

  1. A 1x1 convolution outputting `bottleneck_channels` number of channels.
  2. `inner_stack_size` number of `conv_size` convolutions.
  3. A 1x1 convolution expanding to `output_channels` number of channels.

  '''

  def __init__(self,
               output_channels: int,
               bottleneck_channels: int,
               conv_size: int,
               stack_size=3,
               name=None):
    blocks = []
    self.blocks.append(
        ConvBlock(bottleneck_channels, 1, name=f'res_id_reduce_dim_begin'))
    for i in range(stack_size - 2):
      self.blocks.append(
          ConvBlock(bottleneck_channels, conv_size, name=f'res_id_inner_{i}'))
    self.blocks.append(
        ConvBlock(output_channels, 1, name=f'res_id_expand_dim_end'))
    super(BottleneckResidualConvBlock, self).__init__(blocks, name=name)


class BroadcastResidualBlock(ResidualBlock):
  '''
  Block that mixes data across channels, and globally, within each channel.

  Does so via:

  1. A 1x1 convolution (mix across channels)
  2. A linear layer `BroadcastResidualBlock.Broadcast`
  3. An (additional) 1x1 convolution.

  The input tensor is added to the result of the series of layers.
  '''

  class Broadcast(tf.keras.layers.Layer):
    '''
    Block that, per channel, mixes global state.

    Does this via a linear layer from the flattened channel to an output
    with the same dimensions.
    '''

    def __init__(self, name=None):
      super(BroadcastResidualBlock.Broadcast, self).__init__(name=name)

    def call(self, x):
      assert (len(x.shape) == 4)
      n, h, w, c = x.shape

      x = tf.transpose(x, perm=(0, 3, 1, 2))  # NHWC -> NCHW
      x = tf.reshape(x, (n, c, h * w))  # flatten
      x = tf.keras.layers.Dense(h * w, name='broadcast_linear')(x)  # mix
      x = tf.keras.activations.relu(x)
      x = tf.reshape(x, (n, c, h, w))  # expand
      x = tf.transpose(x, perm=(0, 2, 3, 1))  # NCHW -> NHWC

      return x

  def __init__(self,
               output_channels: int,
               activation=tf.keras.activations.linear,
               name=None):
    blocks = [
        ConvBlock(output_channels, 1, name='broadcast_conv_first'),
        BroadcastResidualBlock.Broadcast(name='broadcast_mix'),
        ConvBlock(output_channels, 1, name='broadcast_conv_last')
    ]

    super(BroadcastResidualBlock, self).__init__(blocks, activation=activation)


class GlobalPoolBias(tf.keras.layers.Layer):

  class GlobalPool(tf.keras.layers.Layer):
    '''
    Computes mean and max of each channel. Given a tensor with shape (n, h, w, c),
    outputs a tensor of shape (n, 2c).
    '''

    def __init__(self, name=None):
      super(GlobalPoolBias.GlobalPool, self).__init__(name=name)

    def call(self, x):
      assert (len(x.shape) == 4)
      n, h, w, c = x.shape

      x = tf.transpose(x, perm=(0, 3, 1, 2))  # NHWC -> NCHW
      x = tf.reshape(x, (n, c, h * w))  # flatten
      x_mean = tf.math.reduce_mean(x, axis=2)
      x_max = tf.math.reduce_max(x, axis=2)

      return tf.concat([x_mean, x_max], axis=1)

  def __init__(self, channels: int, name=None):
    super(GlobalPoolBias, self).__init__(name=name)
    self.channels = channels
    self.batch_norm_g = tf.keras.layers.BatchNormalization(
        scale=False, momentum=.999, eps=1e-3, name='batch_norm_gpool')
    self.gpool = GlobalPoolBias.GlobalPool(name='gpool')
    self.dense = tf.keras.layers.Dense(channels)

  def call(self, x, g):
    assert (x.shape == g.shape)
    assert (len(x.shape) == 4)
    assert (x.shape[3] == g.shape[3])

    g = self.batch_norm_g(g)
    g = tf.keras.activations.relu(g)
    g_pooled = self.gpool(g)
    g_biases = self.dense(g_pooled)

    x = x + g_biases

    return (x, g_pooled)


class PolicyHead(tf.keras.layers.Layer):

  def __init__(self, channels=32, name=None):
    super(PolicyHead, self).__init__(name=name)
    self.conv_v = tf.keras.layers.Conv2D(channels,
                                         1,
                                         padding='same',
                                         name='policy_conv_v')
    self.conv_g = tf.keras.layers.Conv2D(channels,
                                         1,
                                         padding='same',
                                         name='policy_conv_g')
    self.gpool = GlobalPoolBias(channels)
    self.batch_norm = tf.keras.layers.BatchNormalization(
        scale=False, momentum=.999, eps=1e-3, name='batch_norm_gpool')
    self.output_moves = tf.keras.layers.Conv2D(1,
                                               1,
                                               padding='same',
                                               name='policy_output_moves')
    self.output_pass = tf.keras.layers.Dense(1, name='policy_output_pass')

  def call(self, x):
    p = self.conv_v(x)
    g = self.conv_g(x)

    (p, g_pooled) = self.gpool(p, g)
    p = self.batch_norm(p)
    p = tf.keras.activations.relu(p)

    p = self.output_moves(p)

    pass_logit = self.output_pass(g_pooled)

    p = tf.keras.layers.Flatten(tf.squeeze(p))

    return tf.concat([p, pass_logit], axis=1)


class P3achyGoModel(tf.keras.Model):

  def __init__(self, config=ModelConfig(), name=None):
    assert (config.kBlocks > 1)

    super(P3achyGoModel, self).__init__(name=name)

    self.blocks = []
    for i in range(config.kBlocks - 1):
      if i == 0:
        self.blocks.append(
            ConvBlock(config.kChannels, config.kConvSize, name='init_conv'))
      elif i % config.kBroadcastInterval == 0:
        self.blocks.append(
            BroadcastResidualBlock(config.kChannels,
                                   activation=tf.keras.activations.relu,
                                   name=f'broadcast_res_{i}'))
      else:
        self.blocks.append(
            BottleneckResidualConvBlock(
                config.kChannels,
                config.kBottleneckChannels,
                config.kConvSize,
                inner_stack_size=config.kBottleneckLength - 2,
                name=f'bottleneck_res_{i}'))

    self.policy_head = PolicyHead(config.kPolicyHeadChannels)
