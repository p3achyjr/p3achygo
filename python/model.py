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

as well as a scalar vector consisting of

(komi / 15.0,)

Output:

A pi = (19, 19) feature plane of logits, where softmax(pi) = policy
A v_outcome = (2, ) logit vector, where softmax(v_outcome) = p(win, lose)
A v_ownership = (19, 19) feature plane representing board ownership in [-1, 1]
A v_score = (800, ) logit vector, where softmax(v_score) = p(each score)

Architecture:

We mimic the architecture in https://openreview.net/pdf?id=bERaNdoegnO.
'''

from __future__ import annotations

import tensorflow as tf

from constants import *
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
    self.conv = tf.keras.layers.Conv2D(
        output_channels,
        conv_size,
        activation=None,  # defer until later
        kernel_regularizer=L2(C_L2),
        padding='same')
    self.batch_norm = tf.keras.layers.BatchNormalization(scale=False,
                                                         momentum=.999,
                                                         epsilon=1e-3)
    self.activation = activation

    # save for serialization
    self.output_channels = output_channels
    self.conv_size = conv_size

  def call(self, x, training=False):
    x = self.conv(x)
    x = self.batch_norm(x, training=training)
    x = self.activation(x)

    return x

  def get_config(self):
    return {
        'output_channels': self.output_channels,
        'conv_size': self.conv_size,
        'activation': tf.keras.activations.serialize(self.activation),
        'name': self.name
    }

  @classmethod
  def from_config(cls, config):
    cls(config['output_channels'],
        config['conv_size'],
        tf.keras.activations.deserialize(config['activation']),
        name=config['name'])


class ResidualBlock(tf.keras.layers.Layer):
  ''' 
  Generalized residual block.

  Input:

  1. A series of operations (blocks)
  2. An activation (activation)

  ResidualBlock(x) = activation(blocks(x) + x)

  IMPORTANT: This block is impossible to serialize. Calling this block
  directly in a model definition will cause issues.
  '''

  def __init__(self,
               inner_blocks: list[tf.keras.layers.Layer],
               activation=tf.keras.activations.linear,
               name=None):
    super(ResidualBlock, self).__init__(name=name)
    self.blocks = inner_blocks
    self.activation = activation

  def call(self, x, training=False):
    res = x
    for block in self.blocks:
      x = block(x, training=training)

    return self.activation(res + x)


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

    # save for serialization
    self.output_channels = output_channels
    self.bottleneck_channels = bottleneck_channels
    self.conv_size = conv_size
    self.stack_size = stack_size

  def get_config(self):
    return {
        'output_channels': self.output_channels,
        'bottleneck_channels': self.bottleneck_channels,
        'conv_size': self.conv_size,
        'stack_size': self.stack_size,
        'name': self.name,
    }


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

    def __init__(self, c: int, h: int, w: int, name=None):
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

      # save for serialization
      self.c, self.h, self.w = c, h, w

    def call(self, x, training=False):
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

    def get_config(self):
      return {'c': self.c, 'h': self.h, 'w': self.w, 'name': self.name}

  def __init__(self,
               output_channels: int,
               board_len: int,
               activation=tf.keras.activations.linear,
               name=None):
    blocks = [
        ConvBlock(output_channels, 1, name='broadcast_conv_first'),
        BroadcastResidualBlock.Broadcast(output_channels,
                                         board_len,
                                         board_len,
                                         name='broadcast_mix'),
        ConvBlock(output_channels, 1, name='broadcast_conv_last')
    ]

    super(BroadcastResidualBlock, self).__init__(blocks,
                                                 activation=activation,
                                                 name=name)

    # save for serialization
    self.output_channels = output_channels
    self.activation_serialized = tf.keras.activations.serialize(activation)

  def get_config(self):
    return {
        'output_channels': self.output_channels,
        'activation': self.activation_serialized,
        'name': self.name,
    }

  @classmethod
  def from_config(cls, config):
    return cls(config['output_channels'],
               tf.keras.activations.deserialize(config['activation']),
               config['name'])


class GlobalPool(tf.keras.layers.Layer):
  '''
  Computes mean and max of each channel. Given a tensor with shape (n, h, w, c),
  outputs a tensor of shape (n, 2c).
  '''

  def __init__(self, c: int, h: int, w: int, name=None):
    super(GlobalPool, self).__init__(name=name)
    self.channel_flatten = tf.keras.layers.Reshape((c, h * w),
                                                   name='gpool_flatten')

    # save parameters for serialization
    self.c, self.h, self.w = c, h, w

  def call(self, x, training=False):
    assert (len(x.shape) == 4)
    n, h, w, c = x.shape

    x = tf.transpose(x, perm=(0, 3, 1, 2))  # NHWC -> NCHW
    x = self.channel_flatten(x)  # flatten
    x_mean = tf.math.reduce_mean(x, axis=2)
    x_max = tf.math.reduce_max(x, axis=2)

    return tf.concat([x_mean, x_max], axis=1)

  def get_config(self):
    return {
        'c': self.c,
        'h': self.h,
        'w': self.w,
        'name': self.name,
    }


class GlobalPoolBias(tf.keras.layers.Layer):
  '''
  Takes in two vectors (x, y), and returns x + dense(gpool(y)), where gpool(y) is
  a vector of the concatenated mean and max of each channel, and dense is a 
  fully connected layer to the number of channels in x (so that channelwise addition
  works).
  '''

  def __init__(self, channels: int, board_len=BOARD_LEN, name=None):
    super(GlobalPoolBias, self).__init__(name=name)
    self.batch_norm_g = tf.keras.layers.BatchNormalization(
        scale=False, momentum=.999, epsilon=1e-3, name='batch_norm_gpool')
    self.gpool = GlobalPool(channels, board_len, board_len, name='gpool')
    self.dense = tf.keras.layers.Dense(channels)

    # save for serialization
    self.channels = channels

  def call(self, x, g, training=False):
    assert (x.shape == g.shape)
    assert (len(x.shape) == 4)
    assert (x.shape[3] == g.shape[3])

    g = self.batch_norm_g(g, training=training)
    g = tf.keras.activations.relu(g)
    g_pooled = self.gpool(g)
    g_biases = self.dense(g_pooled)  # shape = (N, C)

    x = tf.transpose(x, (1, 2, 0, 3))  # NHWC -> HWNC

    x = x + g_biases

    x = tf.transpose(x, (2, 0, 1, 3))  # HWNC -> NHWC

    return (x, g_pooled)

  def get_config(self):
    return {'channels': self.channels, 'name': self.name}


class PolicyHead(tf.keras.layers.Layer):
  '''
  Implementation of policy head from KataGo.

  Input: b x b x c feature matrix
  Output: b x b + 1 length vector indicating logits for each move (incl. pass)

  Layers:

  1) Broadcast Residual Block
  2) Batch normalization of P
  3) A 1x1x1 Convolution to logits on the board.
  4) An FC layer from gpool to pass logit.
  '''

  def __init__(self, channels=32, board_len=BOARD_LEN, name=None):
    super(PolicyHead, self).__init__(name=name)
    self.conv_p = tf.keras.layers.Conv2D(channels,
                                         1,
                                         padding='same',
                                         kernel_regularizer=L2(C_L2),
                                         name='policy_conv_p')
    self.conv_g = tf.keras.layers.Conv2D(channels,
                                         1,
                                         padding='same',
                                         kernel_regularizer=L2(C_L2),
                                         name='policy_conv_g')
    self.gpool = GlobalPoolBias(channels, board_len=board_len)
    self.batch_norm = tf.keras.layers.BatchNormalization(scale=False,
                                                         momentum=.999,
                                                         epsilon=1e-3,
                                                         name='policy_bn')
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

    # save parameters for serialization
    self.channels = channels
    self.board_len = board_len

  def call(self, x, training=False):
    p = self.conv_p(x)
    g = self.conv_g(x)

    (p, g_pooled) = self.gpool(p, g)
    p = self.batch_norm(p, training=training)
    p = tf.keras.activations.relu(p)

    p = self.output_moves(p)

    # Hacky, but forces model to learn when to pass, rather than to learn when
    # not to.
    pass_logit = self.output_pass(g_pooled) - 5

    p = self.flatten(tf.squeeze(p, axis=3))

    return tf.concat([p, pass_logit], axis=1)

  def get_config(self):
    return {
        'channels': self.channels,
        'board_len': self.board_len,
        'name': self.name,
    }


class ValueHead(tf.keras.layers.Layer):
  '''
  Implementation of KataGo value head.

  Input: b x b x c feature matrix 
  Output:

  - (2, ) logits for {win, loss}
  - (b x b) ownership matrix
  - (800, ) logits representing score difference
  '''

  def __init__(self,
               channels=32,
               c_val=64,
               board_len=BOARD_LEN,
               score_range=SCORE_RANGE,
               name=None):
    super(ValueHead, self).__init__(name=name)

    ## Initialize Model Layers ##
    self.conv = tf.keras.layers.Conv2D(channels,
                                       1,
                                       padding='same',
                                       kernel_regularizer=L2(C_L2),
                                       name='value_conv')
    self.gpool = GlobalPool(channels, board_len, board_len, name='value_gpool')

    # Game Outcome Subhead
    self.outcome_biases = tf.keras.layers.Dense(c_val,
                                                kernel_regularizer=L2(C_L2),
                                                name='value_outcome_biases')
    self.outcome_output = tf.keras.layers.Dense(2,
                                                kernel_regularizer=L2(C_L2),
                                                name='value_outcome_output')

    # Ownership Subhead
    self.conv_ownership = tf.keras.layers.Conv2D(1,
                                                 1,
                                                 padding='same',
                                                 kernel_regularizer=L2(C_L2),
                                                 name='value_conv_ownership')

    # Score Distribution Subhead
    self.gamma_pre = tf.keras.layers.Dense(c_val,
                                           kernel_regularizer=L2(C_L2),
                                           name='value_gamma_pre')
    self.gamma_output = tf.keras.layers.Dense(1,
                                              kernel_regularizer=L2(C_L2),
                                              name='value_gamma_output')

    self.score_range = score_range
    self.score_min, self.score_max = -score_range // 2, score_range // 2
    self.scores = .05 * tf.range(self.score_min + .5,
                                 self.score_max + .5)  # [-399.5 ... 399.5]
    self.score_identity = tf.keras.layers.Activation(
        'linear')  # need for mixed-precision
    self.score_pre = tf.keras.layers.Dense(c_val,
                                           kernel_regularizer=L2(C_L2),
                                           name='score_distribution_pre')
    self.score_output = tf.keras.layers.Dense(1,
                                              kernel_regularizer=L2(C_L2),
                                              name='score_distribution_output')

    # Save for serialization
    self.channels = channels
    self.c_val = c_val
    self.board_len = board_len
    self.score_range = score_range

  def call(self, x):
    v = self.conv(x)
    v_pooled = self.gpool(v)

    # Compute Game Output
    game_outcome = self.outcome_biases(v_pooled)
    game_outcome = tf.keras.activations.relu(game_outcome)
    game_outcome = self.outcome_output(game_outcome)

    # Compute Game Ownership
    game_ownership = self.conv_ownership(v)
    game_ownership = tf.keras.activations.tanh(game_ownership)

    # Compute Score Distribution
    gamma = self.gamma_pre(v_pooled)
    gamma = tf.keras.activations.relu(gamma)
    gamma = self.gamma_output(gamma)

    # comments assume score_range = 800
    n, _ = v_pooled.shape

    # [[-399.5]
    #   ...
    #   399.5]]
    scores = self.score_identity(tf.expand_dims(self.scores, axis=1))
    # [[[-399.5]
    #   ...
    #   399.5]]]
    # scores = tf.expand_dims(scores, axis=0)
    # scores = tf.tile(
    #     scores,
    #     [n, 1, 1])  # duplicate `batch_size` times (1, 800, 1) -> (n, 800, 1)

    v_pools = tf.expand_dims(v_pooled, axis=1)  # (n, k) -> (n, 1, k)
    v_pools = tf.tile(v_pools,
                      [1, self.score_range, 1])  # (n, 1, k) -> (n, 800, k)
    # v_scores = tf.concat([v_pools, scores], axis=2)  # (n, 800, k + 1)
    v_scores = tf.vectorized_map(lambda x: tf.concat([x, scores], axis=1),
                                 v_pools)
    v_scores = self.score_pre(v_scores)
    v_scores = tf.keras.activations.relu(v_scores)
    score_logits = self.score_output(v_scores)  # (n, 800, 1)
    score_logits = tf.squeeze(score_logits, axis=2)  # (n, 800)
    score_logits = tf.nn.softplus(gamma) * score_logits

    return (game_outcome, game_ownership, score_logits, gamma)

  def get_config(self):
    return {
        'channels': self.channels,
        'c_val': self.c_val,
        'board_len': self.board_len,
        'score_range': self.score_range,
        'name': self.name
    }


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

  as well as a (1, ) feature vector consisting of

  (komi / 15.0)

  Output:

  One (19, 19) feature plane of logits, where softmax(logits) = policy
  '''

  def __init__(self,
               board_len,
               num_input_planes,
               num_input_features,
               num_blocks,
               num_channels,
               num_bottleneck_channels,
               num_head_channels,
               c_val,
               bottleneck_length,
               conv_size,
               broadcast_interval,
               name=None):
    assert (num_blocks > 1)

    super(P3achyGoModel, self).__init__(name=name)

    ## Initialize Model Layers ##
    self.init_board_conv = ConvBlock(num_channels,
                                     conv_size + 2,
                                     name='init_board_conv')
    self.init_game_layer = tf.keras.layers.Dense(num_channels,
                                                 kernel_regularizer=L2(C_L2),
                                                 name='init_game_layer')
    self.blocks = []
    for i in range(num_blocks - 2):
      if i > 0 and i % broadcast_interval == 0:
        self.blocks.append(
            BroadcastResidualBlock(num_channels,
                                   board_len,
                                   activation=tf.keras.activations.relu,
                                   name=f'broadcast_res_{i}'))
      else:
        self.blocks.append(
            BottleneckResidualConvBlock(num_channels,
                                        num_bottleneck_channels,
                                        conv_size,
                                        stack_size=bottleneck_length,
                                        name=f'bottleneck_res_{i}'))

    self.policy_head = PolicyHead(channels=num_head_channels,
                                  board_len=board_len,
                                  name='policy_head')
    self.value_head = ValueHead(num_head_channels,
                                c_val,
                                board_len,
                                name='value_head')

    ## Initialize Loss Objects. Defer reduction strategy to loss objects ##
    self.scce_logits = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    self.scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    self.mse = tf.keras.losses.MeanSquaredError()
    self.identity = tf.keras.layers.Activation(
        'linear')  # need for mixed-precision

    # store parameters so we can serialize model correctly
    self.board_len = board_len
    self.num_input_planes = num_input_planes
    self.num_input_features = num_input_features
    self.num_blocks = num_blocks
    self.num_channels = num_channels
    self.num_bottleneck_channels = num_bottleneck_channels
    self.num_head_channels = num_head_channels
    self.c_val = c_val
    self.bottleneck_length = bottleneck_length
    self.conv_size = conv_size
    self.broadcast_interval = broadcast_interval

  def call(self, board_state, game_state, training=False):
    x = self.init_board_conv(board_state, training=training)
    game_state_biases = self.init_game_layer(game_state)

    x = tf.transpose(x, (1, 2, 0, 3))  # NHWC -> HWNC

    x = x + game_state_biases

    x = tf.transpose(x, (2, 0, 1, 3))  # HWNC -> NHWC

    for block in self.blocks:
      x = block(x, training=training)

    pi_logits = self.policy_head(x, training=training)
    pi = tf.keras.activations.softmax(pi_logits)
    game_outcome, game_ownership, score_logits, gamma = self.value_head(x)

    return (tf.cast(pi_logits, tf.float32), tf.cast(pi, tf.float32),
            tf.cast(game_outcome,
                    tf.float32), tf.cast(game_ownership, tf.float32),
            tf.cast(score_logits, tf.float32), tf.cast(gamma, tf.float32))

  def loss(self, pi_logits, game_outcome, score_logits, own_pred, gamma, policy,
           score, score_one_hot, own, w_pi, w_val, w_outcome, w_score, w_own,
           w_gamma, use_kl_policy_loss):
    policy_loss = tf.reduce_mean(
        tf.keras.metrics.kl_divergence(
            tf.cast(policy, tf.float32),
            tf.keras.activations.softmax(tf.cast(
                pi_logits,
                tf.float32)))) if use_kl_policy_loss else self.scce_logits(
                    policy, pi_logits)

    did_win = score >= 0
    outcome_loss = self.scce_logits(did_win, game_outcome)

    score_index = score + SCORE_RANGE_MIDPOINT
    score_distribution = tf.keras.activations.softmax(score_logits)
    score_pdf_loss = self.scce(score_index, score_distribution)
    score_cdf_loss = tf.math.reduce_mean(
        tf.math.reduce_sum(tf.math.square(
            tf.math.cumsum(score_one_hot, axis=1) -
            tf.math.cumsum(score_distribution, axis=1)),
                           axis=1))

    own_pred = tf.squeeze(own_pred, -1)  # tailing 1 dim.
    own_loss = self.mse(own, own_pred)

    gamma = tf.squeeze(gamma, axis=-1)
    gamma_loss = tf.math.reduce_mean(gamma * gamma * w_gamma)

    woutcome_loss = w_outcome * outcome_loss
    wscore_pdf_loss = w_score * score_pdf_loss
    wscore_cdf_loss = w_score * score_cdf_loss
    wown_loss = w_own * own_loss
    val_loss = w_val * (woutcome_loss + wscore_pdf_loss +
                        wown_loss) + wscore_cdf_loss

    loss = w_pi * tf.cast(policy_loss, tf.float32) + tf.cast(
        val_loss, tf.float32) + tf.cast(gamma_loss, tf.float32)

    # yapf: disable
    # tf.print('Loss:', loss,
    #          '\nPolicy Loss:', policy_loss,
    #          '\nWeighted Policy Loss:', w_pi * policy_loss,
    #          '\nOutcome Loss:', outcome_loss,
    #          '\nWeighted Outcome Loss:', woutcome_loss,
    #          '\nScore PDF Loss:', score_pdf_loss,
    #          '\nWeighted Score PDF Loss:', wscore_pdf_loss,
    #          '\nScore CDF Loss:', score_cdf_loss,
    #          '\nWeighted Score CDF Loss:', wscore_cdf_loss,
    #          '\nOwn Loss:', own_loss,
    #          '\nWeighted Own Loss:', wown_loss,
    #          '\nGamma Loss:', gamma_loss)
    # yapf: enable
    return loss, policy_loss, outcome_loss, score_pdf_loss, own_loss

  def get_config(self):
    return {
        'board_len': self.board_len,
        'num_input_planes': self.num_input_planes,
        'num_input_features': self.num_input_features,
        'num_blocks': self.num_blocks,
        'num_channels': self.num_channels,
        'num_bottleneck_channels': self.num_bottleneck_channels,
        'num_head_channels': self.num_head_channels,
        'c_val': self.c_val,
        'bottleneck_length': self.bottleneck_length,
        'conv_size': self.conv_size,
        'broadcast_interval': self.broadcast_interval,
        'name': self.name,
    }

  def summary(self, batch_size=32):
    x0 = tf.keras.layers.Input(shape=(self.board_len, self.board_len,
                                      self.num_input_planes),
                               batch_size=batch_size)
    x1 = tf.keras.layers.Input(shape=(self.num_input_features,),
                               batch_size=batch_size)
    model = tf.keras.Model(inputs=[x0, x1], outputs=self.call(x0, x1))
    return model.summary()

  @staticmethod
  def create(config: ModelConfig, board_len: int, num_input_planes: int,
             num_input_features: int, name: str):
    return P3achyGoModel(board_len,
                         num_input_planes,
                         num_input_features,
                         config.kBlocks,
                         config.kChannels,
                         config.kBottleneckChannels,
                         config.kHeadChannels,
                         config.kCVal,
                         config.kBottleneckLength,
                         config.kConvSize,
                         config.kBroadcastInterval,
                         name=name)

  @staticmethod
  def custom_objects():
    return {
        'ConvBlock': ConvBlock,
        'ResidualBlock': ResidualBlock,
        'BottleneckResidualConvBlock': BottleneckResidualConvBlock,
        'BroadcastResidualBlock': BroadcastResidualBlock,
        'Broadcast': BroadcastResidualBlock.Broadcast,
        'GlobalPoolBias': GlobalPoolBias,
        'GlobalPool': GlobalPool,
        'PolicyHead': PolicyHead,
        'P3achyGoModel': P3achyGoModel,
    }
