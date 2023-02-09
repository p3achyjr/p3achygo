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

L2 = tf.keras.regularizers.L2

C_L2 = 1e-4


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
                                       kernel_regularizer=L2(C_L2),
                                       padding='same')
    self.batch_norm = tf.keras.layers.BatchNormalization(scale=False,
                                                         momentum=.999,
                                                         epsilon=1e-3)
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
    blocks.append(
        ConvBlock(bottleneck_channels, 1, name=f'res_id_reduce_dim_begin'))
    for i in range(stack_size - 2):
      blocks.append(
          ConvBlock(bottleneck_channels, conv_size, name=f'res_id_inner_{i}'))
    blocks.append(ConvBlock(output_channels, 1, name=f'res_id_expand_dim_end'))
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

    def __init__(self, n: int, c: int, h: int, w: int, name=None):
      super(BroadcastResidualBlock.Broadcast, self).__init__(name=name)
      self.channel_flatten = tf.keras.layers.Reshape((c, h * w),
                                                     name='broadcast_flatten')
      self.dense = tf.keras.layers.Dense(
          h * w,
          name='broadcast_linear',
          kernel_regularizer=L2(C_L2),
      )
      self.channel_expand = tf.keras.layers.Reshape((c, h, w),
                                                    name='broadcast_expand')

    def call(self, x):
      assert (len(x.shape) == 4)
      n, h, w, c = x.shape

      x = tf.transpose(x, perm=(0, 3, 1, 2))  # NHWC -> NCHW
      # x = tf.reshape(x, (n, c, h * w))  # flatten
      x = self.channel_flatten(x)
      x = self.dense(x)  # mix
      x = tf.keras.activations.relu(x)
      # x = tf.reshape(x, (n, c, h, w))  # expand
      x = self.channel_expand(x)
      x = tf.transpose(x, perm=(0, 2, 3, 1))  # NCHW -> NHWC

      return x

  def __init__(self,
               output_channels: int,
               activation=tf.keras.activations.linear,
               name=None):
    blocks = [
        ConvBlock(output_channels, 1, name='broadcast_conv_first'),
        BroadcastResidualBlock.Broadcast(32,
                                         output_channels,
                                         19,
                                         19,
                                         name='broadcast_mix'),
        ConvBlock(output_channels, 1, name='broadcast_conv_last')
    ]

    super(BroadcastResidualBlock, self).__init__(blocks, activation=activation)


class GlobalPoolBias(tf.keras.layers.Layer):

  class GlobalPool(tf.keras.layers.Layer):
    '''
    Computes mean and max of each channel. Given a tensor with shape (n, h, w, c),
    outputs a tensor of shape (n, 2c).
    '''

    def __init__(self, c: int, h: int, w: int, name=None):
      super(GlobalPoolBias.GlobalPool, self).__init__(name=name)
      self.channel_flatten = tf.keras.layers.Reshape((c, h * w),
                                                     name='gpool_flatten')

    def call(self, x):
      assert (len(x.shape) == 4)
      n, h, w, c = x.shape

      x = tf.transpose(x, perm=(0, 3, 1, 2))  # NHWC -> NCHW
      x = self.channel_flatten(x)  # flatten
      x_mean = tf.math.reduce_mean(x, axis=2)
      x_max = tf.math.reduce_max(x, axis=2)

      return tf.concat([x_mean, x_max], axis=1)

  def __init__(self, channels: int, name=None):
    super(GlobalPoolBias, self).__init__(name=name)
    self.channels = channels
    self.batch_norm_g = tf.keras.layers.BatchNormalization(
        scale=False, momentum=.999, epsilon=1e-3, name='batch_norm_gpool')
    self.gpool = GlobalPoolBias.GlobalPool(channels, 19, 19, name='gpool')
    self.dense = tf.keras.layers.Dense(channels)

  def call(self, x, g):
    assert (x.shape == g.shape)
    assert (len(x.shape) == 4)
    assert (x.shape[3] == g.shape[3])

    g = self.batch_norm_g(g)
    g = tf.keras.activations.relu(g)
    g_pooled = self.gpool(g)
    g_biases = self.dense(g_pooled)  # shape = (N, C)

    x = tf.transpose(x, (1, 2, 0, 3))  # NHWC -> HWNC

    x = x + g_biases

    x = tf.transpose(x, (2, 0, 1, 3))  # HWNC -> NHWC

    return (x, g_pooled)


class PolicyHead(tf.keras.layers.Layer):
  '''
  Implementation of policy head from KataGo.

  Input: b x b x c feature matrix
  Output: b x b + 1 length vector indicating logits for each move (incl. pass)

  Layers:

  1) 2 parallel convolutions outputting P, G, tensors of dim b x b x `channels`
  2) A global pooling bias layer biasing G to P (i.e. P = P + gpool(G))
  3) Batch normalization of P
  4) A 1x1 convolution outputting a single b x b matrix with move logits
  5) A dense layer for gpool(G) to a single output containing the pass logit
  '''

  def __init__(self, channels=32, name=None):
    super(PolicyHead, self).__init__(name=name)
    self.conv_p = tf.keras.layers.Conv2D(channels,
                                         1,
                                         padding='same',
                                         kernel_regularizer=L2(C_L2),
                                         name='policy_conv_v')
    self.conv_g = tf.keras.layers.Conv2D(channels,
                                         1,
                                         padding='same',
                                         kernel_regularizer=L2(C_L2),
                                         name='policy_conv_g')
    self.gpool = GlobalPoolBias(channels)
    self.batch_norm = tf.keras.layers.BatchNormalization(
        scale=False, momentum=.999, epsilon=1e-3, name='batch_norm_gpool')
    self.flatten = tf.keras.layers.Flatten()
    self.output_moves = tf.keras.layers.Conv2D(1,
                                               1,
                                               padding='same',
                                               kernel_regularizer=L2(C_L2),
                                               name='policy_output_moves')
    self.output_pass = tf.keras.layers.Dense(
        1,
        name='policy_output_pass',
        kernel_regularizer=L2(C_L2),
    )
    self.scaling_pass = tf.keras.layers.Rescaling(3e-1,
                                                  name='policy_output_scale')

  def call(self, x):
    p = self.conv_p(x)
    g = self.conv_g(x)

    (p, g_pooled) = self.gpool(p, g)
    p = self.batch_norm(p)
    p = tf.keras.activations.relu(p)

    p = self.output_moves(p)

    pass_logit = self.output_pass(g_pooled)
    pass_logit = self.scaling_pass(pass_logit)

    p = self.flatten(tf.squeeze(p))

    return tf.concat([p, pass_logit], axis=1)


class P3achyGoModel(tf.keras.Model):
  '''
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
  '''

  def __init__(self, config=ModelConfig(), shape=(19, 19, 7), name=None):
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
            BottleneckResidualConvBlock(config.kChannels,
                                        config.kBottleneckChannels,
                                        config.kConvSize,
                                        stack_size=config.kBottleneckLength,
                                        name=f'bottleneck_res_{i}'))

    self.policy_head = PolicyHead(config.kPolicyHeadChannels)

  def call(self, x):
    for block in self.blocks:
      x = block(x)

    pi_logits = self.policy_head(x)

    return pi_logits

  def summary(self):
    x = tf.keras.layers.Input(shape=(19, 19, 7))
    model = tf.keras.Model(inputs=[x], outputs=self.call(x))
    return model.summary()
