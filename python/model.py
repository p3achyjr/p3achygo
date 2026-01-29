"""
Model Definition.

Amalgamation of different layers types from various AlphaZero reproductions.
"""

from __future__ import annotations
from typing import NamedTuple, Optional

import tensorflow as tf
import keras

from constants import *
from model_config import ModelConfig

L2 = keras.regularizers.L2

C_L2 = 1e-4


class ModelPredictions(NamedTuple):
    """Model prediction outputs."""

    pi_logits: tf.Tensor
    pi_logits_aux: tf.Tensor
    game_outcome: tf.Tensor
    score_logits: tf.Tensor
    own_pred: tf.Tensor
    q6_pred: tf.Tensor
    q16_pred: tf.Tensor
    q50_pred: tf.Tensor
    gamma: tf.Tensor
    # v1 predictions (optional)
    q6_err_pred: Optional[tf.Tensor] = None
    q16_err_pred: Optional[tf.Tensor] = None
    q50_err_pred: Optional[tf.Tensor] = None
    q6_score_pred: Optional[tf.Tensor] = None
    q16_score_pred: Optional[tf.Tensor] = None
    q50_score_pred: Optional[tf.Tensor] = None
    q6_score_err_pred: Optional[tf.Tensor] = None
    q16_score_err_pred: Optional[tf.Tensor] = None
    q50_score_err_pred: Optional[tf.Tensor] = None
    pi_logits_soft: Optional[tf.Tensor] = None
    pi_logits_optimistic: Optional[tf.Tensor] = None


class GroundTruth(NamedTuple):
    """Ground truth labels for training."""

    policy: tf.Tensor
    policy_aux: tf.Tensor
    score: tf.Tensor
    score_one_hot: tf.Tensor
    own: tf.Tensor
    q6: tf.Tensor
    q16: tf.Tensor
    q50: tf.Tensor
    # v1 labels (optional)
    q6_score: Optional[tf.Tensor] = None
    q16_score: Optional[tf.Tensor] = None
    q50_score: Optional[tf.Tensor] = None


class LossWeights(NamedTuple):
    """Weights for different loss components."""

    w_pi: float
    w_pi_aux: float
    w_val: float
    w_outcome: float
    w_score: float
    w_own: float
    w_q6: float
    w_q16: float
    w_q50: float
    w_gamma: float
    # v1 weights
    w_q_err: float = 0.0
    w_q_score: float = 0.0
    w_q_score_err: float = 0.0
    w_pi_soft: float = 0.0
    w_pi_optimistic: float = 0.0


def make_conv(
    output_channels: int,
    kernel_size: int,
    init="glorot_uniform",
    use_bias=False,
    name=None,
):
    return keras.layers.Conv2D(
        output_channels,
        kernel_size,
        activation=None,
        kernel_regularizer=L2(C_L2),
        padding="same",
        use_bias=use_bias,
        kernel_initializer=init,
        name=name,
    )


def make_dense(output_dim: int, kern_init="glorot_uniform", name=None):
    return keras.layers.Dense(
        output_dim,
        kernel_initializer=kern_init,
        kernel_regularizer=L2(C_L2),
        name=name,
    )


def make_v1_dense(
    output_dim: int,
    activation=keras.activations.mish,
    name=None,
):
    """Dense layer with He initialization scaled by gamma for the given activation."""
    kern_init = keras.initializers.VarianceScaling(
        scale=gamma(activation) ** 2,
        mode="fan_in",
        distribution="truncated_normal",
    )
    return keras.layers.Dense(
        output_dim,
        kernel_initializer=kern_init,
        kernel_regularizer=L2(C_L2),
        name=name,
    )


def gamma(act):
    if act == keras.activations.relu:
        return 1.712
    elif act == keras.activations.mish:
        return 1.592
    return 2.0**0.5


class ConvBlock(keras.layers.Layer):
    """
    Basic convolutional block.
    """

    def __init__(
        self,
        output_channels: int,
        conv_size: int,
        activation=keras.activations.relu,
        use_var_norm=True,
        variance=1.0,
        version=1,
        name=None,
    ):
        super(ConvBlock, self).__init__(name=name)
        kern_init = (
            keras.initializers.VarianceScaling(
                scale=gamma(activation) ** 2,
                mode="fan_in",
                distribution="truncated_normal",
            )
            if use_var_norm
            else "glorot_uniform"
        )
        self.conv = make_conv(output_channels, kernel_size=conv_size, init=kern_init)
        self.norm_layer = (
            keras.layers.Rescaling(scale=float(1.0 / (variance**0.5)), offset=0.0)
            if use_var_norm
            else keras.layers.BatchNormalization(
                scale=False, momentum=0.999, epsilon=1e-3
            )
        )
        self.activation = activation
        self.variance = variance
        self.version = version

        # save for serialization
        self.output_channels = output_channels
        self.conv_size = conv_size
        self.use_var_norm = use_var_norm

    def call(self, x, training=False):
        raise Exception("Do not call directly")

    def get_config(self):
        return {
            "output_channels": self.output_channels,
            "conv_size": self.conv_size,
            "activation": keras.activations.serialize(self.activation),
            "use_var_norm": self.use_var_norm,
            "variance": self.variance,
            "version": self.version,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            output_channels=config["output_channels"],
            conv_size=config["conv_size"],
            activation=keras.activations.deserialize(config["activation"]),
            use_var_norm=config.get("use_var_norm", True),
            variance=config.get("variance", 1.0),
            version=config.get("version", 1),
            name=config.get("name"),
        )


class ConvPostActivation(ConvBlock):
    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm_layer(x, training=training)
        x = self.activation(x)
        return x


class ConvPreActivation(ConvBlock):
    def call(self, x, training=False):
        x = self.norm_layer(x, training=training)
        x = self.activation(x)
        x = self.conv(x)
        return x


def make_v0_conv_block(output_channels: int, conv_size: int, name=None):
    return ConvPostActivation(
        output_channels=output_channels,
        conv_size=conv_size,
        activation=keras.activations.relu,
        use_var_norm=False,
        variance=1.0,
        version=0,
        name=name,
    )


def make_v1_conv_block(output_channels: int, conv_size: int, variance=1.0, name=None):
    return ConvPreActivation(
        output_channels=output_channels,
        conv_size=conv_size,
        activation=keras.activations.mish,
        use_var_norm=True,
        variance=variance,
        version=1,
        name=name,
    )


class ResidualBlock(keras.layers.Layer):
    """
    Generalized residual block.

    Input:

    1. A series of operations (blocks)
    2. An activation (activation)

    ResidualBlock(x) = activation(blocks(x) + x)

    IMPORTANT: This block is impossible to serialize. Calling this block
    directly in a model definition will cause issues.
    """

    def __init__(
        self,
        inner_blocks: list[keras.layers.Layer],
        activation=keras.activations.linear,
        name=None,
    ):
        super(ResidualBlock, self).__init__(name=name)
        self.blocks = inner_blocks
        self.activation = activation

    def call(self, x, training=False):
        res = x
        for block in self.blocks:
            x = block(x, training=training)

        return self.activation(res + x)


class ClassicResidualBlock(ResidualBlock):
    """
    Residual block found in AlphaGo, KataGo, and Leela.
    """

    def __init__(
        self,
        output_channels: int,
        conv_size: int,
        stack_size=2,
        version=0,
        incoming_var=1.0,
        name=None,
    ):
        blocks = []
        for i in range(stack_size):
            variance = incoming_var if i == 0 else 1.0
            if version == 0:
                block = make_v0_conv_block(
                    output_channels, conv_size, name=f"res_id_inner_{i}"
                )
            else:
                block = make_v1_conv_block(
                    output_channels,
                    conv_size,
                    variance=variance,
                    name=f"res_id_inner_{i}",
                )
            blocks.append(block)
        super(ClassicResidualBlock, self).__init__(blocks, name=name)

        # save for serialization
        self.output_channels = output_channels
        self.conv_size = conv_size
        self.stack_size = stack_size
        self.version = version
        self.incoming_var = incoming_var

    def get_config(self):
        return {
            "output_channels": self.output_channels,
            "conv_size": self.conv_size,
            "stack_size": self.stack_size,
            "version": self.version,
            "incoming_var": self.incoming_var,
            "name": self.name,
        }


class BottleneckResidualConvBlock(ResidualBlock):
    """
    Bottleneck block that reduces dimension, performs inner convolutions, and
    expands dimension.

    Does so via `num_conv_stacks` number of:

    1. A 1x1 convolution outputting `bottleneck_channels` number of channels.
    2. `inner_stack_size` number of `conv_size` convolutions.
    3. A 1x1 convolution expanding to `output_channels` number of channels.

    """

    def __init__(
        self,
        output_channels: int,
        bottleneck_channels: int,
        conv_size: int,
        stack_size=3,
        version=0,
        incoming_var=1.0,
        name=None,
    ):
        blocks = []
        if version == 0:
            blocks.append(
                make_v0_conv_block(
                    bottleneck_channels, 1, name="res_id_reduce_dim_begin"
                )
            )
            for i in range(stack_size - 2):
                blocks.append(
                    make_v0_conv_block(
                        bottleneck_channels, conv_size, name=f"res_id_inner_{i}"
                    )
                )
            blocks.append(
                make_v0_conv_block(output_channels, 1, name="res_id_expand_dim_end")
            )
        else:
            blocks.append(
                make_v1_conv_block(
                    bottleneck_channels,
                    1,
                    variance=incoming_var,
                    name="res_id_reduce_dim_begin",
                )
            )
            for i in range(stack_size - 2):
                blocks.append(
                    make_v1_conv_block(
                        bottleneck_channels, conv_size, name=f"res_id_inner_{i}"
                    )
                )
            blocks.append(
                make_v1_conv_block(output_channels, 1, name="res_id_expand_dim_end")
            )
        super(BottleneckResidualConvBlock, self).__init__(blocks, name=name)

        # save for serialization
        self.output_channels = output_channels
        self.bottleneck_channels = bottleneck_channels
        self.conv_size = conv_size
        self.stack_size = stack_size
        self.version = version
        self.incoming_var = incoming_var

    def get_config(self):
        return {
            "output_channels": self.output_channels,
            "bottleneck_channels": self.bottleneck_channels,
            "conv_size": self.conv_size,
            "stack_size": self.stack_size,
            "version": self.version,
            "incoming_var": self.incoming_var,
            "name": self.name,
        }


class NbtResidualBlock(ResidualBlock):
    """
    Nested Bottleneck Residual Block, a. la. KataGo.
    """

    def __init__(
        self,
        output_channels: int,
        bottleneck_channels: int,
        conv_size: int,
        version=0,
        incoming_var=1.0,
        name=None,
    ):
        blocks = []
        if version == 0:
            blocks.append(
                make_v0_conv_block(bottleneck_channels, 1, name="nbt_reduce_dim")
            )
        else:
            blocks.append(
                make_v1_conv_block(
                    bottleneck_channels, 1, variance=incoming_var, name="nbt_reduce_dim"
                )
            )
        blocks.append(
            ClassicResidualBlock(
                bottleneck_channels,
                conv_size,
                2,
                version=version,
                incoming_var=1.0,
                name="nbt_res0",
            )
        )
        blocks.append(
            ClassicResidualBlock(
                bottleneck_channels,
                conv_size,
                2,
                version=version,
                incoming_var=2.0,
                name="nbt_res1",
            )
        )
        if version == 0:
            blocks.append(make_v0_conv_block(output_channels, 1, name="nbt_expand_dim"))
        else:
            blocks.append(
                make_v1_conv_block(
                    output_channels, 1, variance=3.0, name="nbt_expand_dim"
                )
            )
        super(NbtResidualBlock, self).__init__(blocks, name=name)

        # save for serialization
        self.output_channels = output_channels
        self.bottleneck_channels = bottleneck_channels
        self.conv_size = conv_size
        self.version = version
        self.incoming_var = incoming_var

    def get_config(self):
        return {
            "output_channels": self.output_channels,
            "bottleneck_channels": self.bottleneck_channels,
            "conv_size": self.conv_size,
            "version": self.version,
            "incoming_var": self.incoming_var,
            "name": self.name,
        }


class BroadcastResidualBlock(ResidualBlock):
    """
    Block that mixes data across channels, and globally, within each channel.

    Does so via:

    1. A 1x1 convolution (mix across channels)
    2. A linear layer `BroadcastResidualBlock.Broadcast`
    3. An (additional) 1x1 convolution.

    The input tensor is added to the result of the series of layers.
    """

    class Broadcast(keras.layers.Layer):
        """
        Block that, per channel, mixes global state.

        Does this via a linear layer from the flattened channel to an output
        with the same dimensions.
        """

        def __init__(
            self,
            c: int,
            h: int,
            w: int,
            act=keras.activations.relu,
            version=0,
            name=None,
        ):
            super(BroadcastResidualBlock.Broadcast, self).__init__(name=name)
            self.channel_flatten = keras.layers.Reshape(
                (c, h * w), name="broadcast_flatten"
            )
            if version == 0:
                self.dense = make_dense(h * w, name="broadcast_linear")
            else:
                self.dense = make_v1_dense(
                    h * w, activation=act, name="broadcast_linear"
                )
            self.channel_expand = keras.layers.Reshape(
                (c, h, w), name="broadcast_expand"
            )

            # save for serialization
            self.c, self.h, self.w = c, h, w
            self.act = act
            self.version = version

        def call(self, x, training=False):
            raise Exception("Do not call directly")

        def get_config(self):
            return {
                "c": self.c,
                "h": self.h,
                "w": self.w,
                "act": keras.activations.serialize(self.act),
                "version": self.version,
                "name": self.name,
            }

        @classmethod
        def from_config(cls, config):
            return cls(
                c=config["c"],
                h=config["h"],
                w=config["w"],
                act=keras.activations.deserialize(config["act"]),
                version=config.get("version", 0),
                name=config.get("name"),
            )

    class BroadcastPostAct(Broadcast):
        def call(self, x, training=False):
            assert len(x.shape) == 4

            x = keras.ops.transpose(x, axes=(0, 3, 1, 2))  # NHWC -> NCHW
            x = self.channel_flatten(x)
            x = self.dense(x)  # mix
            x = self.act(x)
            x = self.channel_expand(x)
            x = keras.ops.transpose(x, axes=(0, 2, 3, 1))  # NCHW -> NHWC

            return x

    class BroadcastPreAct(Broadcast):
        def call(self, x, training=False):
            assert len(x.shape) == 4

            x = keras.ops.transpose(x, axes=(0, 3, 1, 2))  # NHWC -> NCHW
            x = self.act(x)
            x = self.channel_flatten(x)
            x = self.dense(x)  # mix
            x = self.channel_expand(x)
            x = keras.ops.transpose(x, axes=(0, 2, 3, 1))  # NCHW -> NHWC

            return x

    def __init__(
        self,
        output_channels: int,
        board_len: int,
        version=0,
        incoming_var=1.0,
        name=None,
    ):
        broadcast_act = (
            keras.activations.relu if version == 0 else keras.activations.mish
        )
        if version == 0:
            conv_first = make_v0_conv_block(
                output_channels, 1, name="broadcast_conv_first"
            )
            conv_last = make_v0_conv_block(
                output_channels, 1, name="broadcast_conv_last"
            )
        else:
            conv_first = make_v1_conv_block(
                output_channels, 1, variance=incoming_var, name="broadcast_conv_first"
            )
            conv_last = make_v1_conv_block(
                output_channels, 1, name="broadcast_conv_last"
            )

        broadcast_fn = (
            BroadcastResidualBlock.BroadcastPreAct
            if version >= 1
            else BroadcastResidualBlock.BroadcastPostAct
        )
        blocks = [
            conv_first,
            broadcast_fn(
                output_channels,
                board_len,
                board_len,
                act=broadcast_act,
                version=version,
                name="broadcast_mix",
            ),
            conv_last,
        ]

        super(BroadcastResidualBlock, self).__init__(blocks, name=name)

        # save for serialization
        self.output_channels = output_channels
        self.board_len = board_len
        self.version = version
        self.incoming_var = incoming_var

    def get_config(self):
        return {
            "output_channels": self.output_channels,
            "board_len": self.board_len,
            "version": self.version,
            "incoming_var": self.incoming_var,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            config["output_channels"],
            config["board_len"],
            config["version"],
            config["incoming_var"],
            config["name"],
        )


class GlobalPool(keras.layers.Layer):
    """
    Computes mean and max of each channel. Given a tensor with shape (n, h, w, c),
    outputs a tensor of shape (n, 2c).
    """

    def __init__(self, name=None):
        super(GlobalPool, self).__init__(name=name)

    def call(self, x):
        # Reduce over spatial dims (h, w)
        x_mean = keras.ops.mean(x, axis=(1, 2))  # (batch, c)
        x_max = keras.ops.max(x, axis=(1, 2))  # (batch, c)
        return keras.ops.concatenate([x_mean, x_max], axis=-1)  # (batch, 2c)

    def get_config(self):
        return {
            "name": self.name,
        }


class GlobalPoolBias(keras.layers.Layer):
    """
    Takes in two vectors (x, y), and returns x + dense(gpool(y)), where gpool(y) is
    a vector of the concatenated mean and max of each channel, and dense is a
    fully connected layer to the number of channels in x (so that channelwise addition
    works).
    """

    def __init__(
        self,
        channels: int,
        act=keras.activations.relu,
        use_var_norm=True,
        incoming_var=1.0,
        name=None,
    ):
        super(GlobalPoolBias, self).__init__(name=name)
        self.g_norm_layer = (
            keras.layers.Identity()  # scaling done already
            if use_var_norm
            else keras.layers.BatchNormalization(
                scale=False, momentum=0.999, epsilon=1e-3, name="batch_norm_gpool"
            )
        )
        self.gpool = GlobalPool(name="gpool")
        self.dense = make_dense(
            channels, kern_init=keras.initializers.VarianceScaling(1.0)
        )

        # save for serialization
        self.channels = channels
        self.use_var_norm = use_var_norm
        self.incoming_var = incoming_var
        self.act = act

    def call(self, x, g, training=False):
        assert x.shape == g.shape
        assert len(x.shape) == 4
        assert x.shape[3] == g.shape[3]

        g = self.g_norm_layer(g, training=training)
        g = self.act(g)
        g_pooled = self.gpool(g)
        g_biases = self.dense(g_pooled)  # shape = (N, C)
        x = x + g_biases[:, None, None, :]

        return (x, g_pooled)

    def get_config(self):
        return {
            "channels": self.channels,
            "act": keras.activations.serialize(self.act),
            "use_var_norm": self.use_var_norm,
            "incoming_var": self.incoming_var,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            channels=config["channels"],
            act=keras.activations.deserialize(config["act"]),
            use_var_norm=config.get("use_var_norm", True),
            incoming_var=config.get("incoming_var", 1.0),
            name=config.get("name"),
        )


class PolicyHead(keras.layers.Layer):
    """
    Implementation of policy head from KataGo.

    Input: b x b x c feature matrix
    Output: b x b + 1 length vector indicating logits for each move (incl. pass)

    Layers:

    1) Broadcast Residual Block
    2) Batch normalization of P
    3) A 1x1x1 Convolution to logits on the board.
    4) An FC layer from gpool to pass logit.
    """

    def __init__(
        self,
        channels=32,
        version=0,
        use_var_norm=True,
        incoming_var=1.0,
        name=None,
    ):
        super(PolicyHead, self).__init__(name=name)
        self.conv_p = make_conv(channels, kernel_size=1, name="policy_conv_p")
        self.conv_g = make_conv(channels, kernel_size=1, name="policy_conv_g")
        self.gpool = GlobalPoolBias(
            channels,
            act=keras.activations.mish if version >= 1 else keras.activations.relu,
            use_var_norm=use_var_norm,
            incoming_var=incoming_var,
            name="policy_gpool",
        )
        self.norm_layer = (
            keras.layers.Rescaling(
                scale=float(1.0 / (incoming_var**0.5)),
                offset=0.0,
                name="policy_rescale",
            )
            if use_var_norm
            else keras.layers.Identity()
        )
        self.flatten = keras.layers.Flatten()
        self.output_moves = make_conv(2, kernel_size=1, name="policy_output_moves")
        self.output_pass = make_dense(2, name="policy_output_pass")
        self.act = keras.activations.mish if version > 0 else keras.activations.relu
        self.use_var_norm = use_var_norm
        self.incoming_var = incoming_var

        if version == 0:
            self.soft_policy_moves = None
            self.soft_policy_pass = None
            self.optimistic_policy_moves = None
            self.optimistic_policy_pass = None
        else:
            self.soft_policy_moves = make_conv(
                1, kernel_size=1, name="policy_soft_moves"
            )
            self.soft_policy_pass = make_dense(1, name="policy_soft_pass")
            self.optimistic_policy_moves = make_conv(
                1, kernel_size=1, name="policy_optimistic_moves"
            )
            self.optimistic_policy_pass = make_dense(1, name="policy_optimistic_pass")

        # save parameters for serialization
        self.channels = channels
        self.version = version

    def call(self, x, training=False):
        x = self.norm_layer(x, training=training)
        p = self.conv_p(x)
        g = self.conv_g(x)

        (p, g_pooled) = self.gpool(p, g)
        p = self.act(p)

        pi = self.output_moves(p)

        # Hacky, but forces model to learn when to pass, rather than to learn when
        # not to.
        pass_logits = self.output_pass(g_pooled) - 3
        pass_logit = keras.ops.expand_dims(pass_logits[:, 0], axis=1)
        pass_logit_aux = keras.ops.expand_dims(pass_logits[:, 1], axis=1)

        pi, pi_aux = pi[:, :, :, 0], pi[:, :, :, 1]
        pi, pi_aux = self.flatten(pi), self.flatten(pi_aux)

        if self.version == 0:
            return (
                keras.ops.concatenate([pi, pass_logit], axis=1),
                keras.ops.concatenate([pi_aux, pass_logit_aux], axis=1),
            )
        else:
            pi_soft = self.flatten(self.soft_policy_moves(p))
            pass_soft = self.soft_policy_pass(g_pooled) - 3
            pi_optimistic = self.flatten(self.optimistic_policy_moves(p))
            pass_optimistic = self.optimistic_policy_pass(g_pooled) - 3

            return (
                keras.ops.concatenate([pi, pass_logit], axis=1),
                keras.ops.concatenate([pi_aux, pass_logit_aux], axis=1),
                keras.ops.concatenate([pi_soft, pass_soft], axis=1),
                keras.ops.concatenate([pi_optimistic, pass_optimistic], axis=1),
            )

    def get_config(self):
        return {
            "channels": self.channels,
            "use_var_norm": self.use_var_norm,
            "incoming_var": self.incoming_var,
            "version": self.version,
            "name": self.name,
        }


class ValueHead(keras.layers.Layer):
    """
    Implementation of KataGo value head.

    Input: b x b x c feature matrix
    Output:

    - (2, ) logits for {win, loss}
    - (b x b) ownership matrix
    - (800, ) logits representing score difference
    - () q30 ~ [-1, 1] representing q at 30 move horizon.
    - () q100 ~ [-1, 1] representing q at 100 move horizon.
    - () q200 ~ [-1, 1] representing q at 200 move horizon.
    """

    def __init__(
        self,
        channels=32,
        c_val=64,
        score_range=SCORE_RANGE,
        version=0,
        incoming_var=1.0,
        name=None,
    ):
        self.version = version
        super(ValueHead, self).__init__(name=name)

        ## Initialize Model Layers ##
        self.act = (
            keras.activations.relu if self.version == 0 else keras.activations.mish
        )
        self.conv = make_conv(channels, kernel_size=1, name="value_conv")
        self.gpool = GlobalPool(name="value_gpool")
        self.norm_layer = (
            keras.layers.Rescaling(
                scale=float(1.0 / (incoming_var**0.5)), offset=0.0, name="value_rescale"
            )
            if version >= 1
            else keras.layers.Identity()
        )

        # Game Outcome/Q Subhead
        self.outcome_q_biases = make_dense(c_val, name="value_outcome_q_biases")
        self.outcome_q_output = (
            make_dense(5, name="value_outcome_q_output")
            if self.version == 0
            else make_dense(14, name="value_outcome_q_output")
        )

        # Ownership Subhead
        self.conv_ownership = make_conv(1, kernel_size=1, name="value_conv_ownership")

        # Score Distribution Subhead
        self.gamma_pre = make_dense(c_val, name="value_gamma_pre")
        self.gamma_output = make_dense(1, name="value_gamma_output")

        self.score_range = score_range
        self.score_min, self.score_max = -score_range // 2, score_range // 2
        # self.scores = .05 * tf.range(self.score_min + .5,
        #                              self.score_max + .5)  # [-399.5 ... 399.5]
        self.score_pre = make_dense(c_val, name="score_distribution_pre")
        self.score_output = make_dense(1, name="score_distribution_output")

        # Save for serialization
        self.channels = channels
        self.c_val = c_val
        self.score_range = score_range
        self.incoming_var = incoming_var

    def call(self, x, scores):
        x = self.norm_layer(x)
        v = self.conv(x)
        v_pooled = self.gpool(v)

        # Compute Game Outcome Values (Outcome Logits + Q-values).
        game_outcome = self.outcome_q_biases(v_pooled)
        game_outcome = self.act(game_outcome)
        game_outcome = self.outcome_q_output(game_outcome)

        outcome_logits = game_outcome[:, 0:2]

        q6 = keras.activations.tanh(game_outcome[:, 2])
        q16 = keras.activations.tanh(game_outcome[:, 3])
        q50 = keras.activations.tanh(game_outcome[:, 4])

        # Compute Game Ownership
        game_ownership = self.conv_ownership(v)
        game_ownership = keras.activations.tanh(game_ownership)

        # Compute Score Distribution
        gamma = self.gamma_pre(v_pooled)
        gamma = self.act(gamma)
        gamma = self.gamma_output(gamma)

        # Compute score logits by combining v_pooled features with each score bin
        # Goal: Create (batch, score_range, pooled_features + 1) tensor
        # where each slice (batch, i, :) = [v_pooled_features..., score_bin_i]

        # Cast scores for mixed-precision without losing shape information
        scores = tf.cast(scores, dtype=v_pooled.dtype)

        # Reshape to broadcastable shapes
        batch_size = keras.ops.shape(x)[0]
        pooled_features = keras.ops.shape(v_pooled)[1]

        v_pooled_exp = v_pooled[:, None, :]  # (batch, 1, features)
        scores_exp = scores[None, :, None]  # (1, score_range, 1)

        # Broadcast and concatenate to create (batch, score_range, features + 1)
        v_scores = keras.ops.concatenate(
            [
                keras.ops.broadcast_to(
                    v_pooled_exp, (batch_size, self.score_range, pooled_features)
                ),
                keras.ops.broadcast_to(scores_exp, (batch_size, self.score_range, 1)),
            ],
            axis=-1,
        )

        v_scores = self.score_pre(v_scores)
        v_scores = self.act(v_scores)
        score_logits = self.score_output(v_scores)  # (n, 800, 1)
        score_logits = keras.ops.squeeze(score_logits, axis=2)  # (n, 800)
        score_logits = keras.ops.softplus(gamma) * score_logits

        if self.version == 0:
            return (
                outcome_logits,
                game_ownership,
                score_logits,
                gamma,
                q6,
                q16,
                q50,
            )
        else:
            # q_err outputs predict squared error of q values (range [0, 4] since q is in [-1, 1])
            # Use 4 * sigmoid to constrain to [0, 4]
            q6_err = 4 * keras.activations.sigmoid(game_outcome[:, 5])
            q16_err = 4 * keras.activations.sigmoid(game_outcome[:, 6])
            q50_err = 4 * keras.activations.sigmoid(game_outcome[:, 7])

            # q_score_err outputs predict squared error of score predictions
            # Use abs to ensure non-negative
            q6_score_err = keras.ops.abs(game_outcome[:, 11])
            q16_score_err = keras.ops.abs(game_outcome[:, 12])
            q50_score_err = keras.ops.abs(game_outcome[:, 13])

            return (
                outcome_logits,
                game_ownership,
                score_logits,
                gamma,
                q6,
                q16,
                q50,
                q6_err,
                q16_err,
                q50_err,
                game_outcome[:, 8],  # q6_score
                game_outcome[:, 9],  # q16_score
                game_outcome[:, 10],  # q50_score
                q6_score_err,
                q16_score_err,
                q50_score_err,
            )

    def get_config(self):
        return {
            "channels": self.channels,
            "c_val": self.c_val,
            "score_range": self.score_range,
            "version": self.version,
            "incoming_var": self.incoming_var,
            "name": self.name,
        }


class P3achyGoModel(keras.Model):
    """
    Input:

    At move k, pass in 15 19 x 19 binary feature planes containing:

    1. Location has own stone
    2. Location has opponent stone
    3. {k - 5}th move (one hot)
    4. {k - 4}th move
    5. {k - 3}rd move
    6. {k - 2}nd move
    7. {k - 1}st move
    8. Own stones in atari
    9. Opp Stones in atari
    10. Own stones with 2 liberties
    11. Opp stones with 2 liberties
    12. Own stones with 3 liberties
    13. Opp stones with 3 liberties
    (v1) 14. Own stones in ladder
    (v1) 15. Opp stones in ladder

    as well as a (7, ) feature vector consisting of

    1. Player is B
    2. Player is W
    3. {k - 5}th move was pass
    4. {k - 4}th move was pass
    5. {k - 3}rd move was pass
    6. {k - 2}nd move was pass
    7. {k - 1}st move was pass
    (v1) komi (current player perspective) / 15.0

    Output:

    0: Move Logits
    1: Move Probabilities
    2: Outcome Logits
    3: Outcome probabilities
    4: Ownership
    5: Score Logits
    6: Score Probabilities
    7: Gamma
    8: Aux Policy

    (v0) 9: Q-value in 6 moves
    (v0) 10: Q-value in 16 moves
    (v0) 11: Q-value in 50 moves

    (v1) 9. Exp-weighted sum of Q-values 0-6 turns into the future
    (v1) 10. Exp-weighted sum of Q-values 0-30 turns into the future
    (v1) 11. Exp-weighted sum of Q-values 0-50 turns into the future
    (v1) 12. Squared Error of (9)
    (v1) 13. Squared Error of (10)
    (v1) 14. Squared Error of (11)
    (v1) 15. Exp-weighted sum of score 0-6 turns into the future
    (v1) 16. Exp-weighted sum of score 0-30 turns into the future
    (v1) 17. Exp-weighted sum of score 0-50 turns into the future
    (v1) 18. Squared Error of (14)
    (v1) 19. Squared Error of (15)
    (v1) 20. Squared Error of (16)
    (v1) 21. Soft Policy Target (pi ^ .25)
    (v1) 22. Optimistic Policy
    """

    def __init__(
        self,
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
        trunk_block_type="btl",
        version=0,
        name=None,
    ):
        assert num_blocks > 1

        super(P3achyGoModel, self).__init__(name=name)
        self.version = version
        self.act = keras.activations.mish if version >= 1 else keras.activations.relu
        self.use_var_norm = version >= 1

        ## Initialize Model Layers ##
        if version == 0:
            self.init_board_conv = make_v0_conv_block(
                num_channels, conv_size + 2, name="init_board_conv"
            )
            self.init_game_layer = make_dense(num_channels, name="init_game_layer")
        else:
            self.init_board_conv = make_conv(
                num_channels,
                conv_size + 2,
                init=keras.initializers.VarianceScaling(
                    scale=2.0,  # inputs are very sparse
                    mode="fan_in",
                    distribution="truncated_normal",
                ),
            )
            self.init_game_layer = make_dense(num_channels, name="init_game_layer")

        self.blocks = []
        for i in range(num_blocks):
            if i % broadcast_interval == broadcast_interval - 1:
                self.blocks.append(
                    BroadcastResidualBlock(
                        num_channels,
                        board_len,
                        version=version,
                        incoming_var=float(i + 1),
                        name=f"broadcast_res_{i}",
                    )
                )
            else:
                if trunk_block_type == "btl":
                    self.blocks.append(
                        BottleneckResidualConvBlock(
                            num_channels,
                            num_bottleneck_channels,
                            conv_size,
                            stack_size=bottleneck_length,
                            version=version,
                            incoming_var=float(i + 1),
                            name=f"bottleneck_res_{i}",
                        )
                    )
                elif trunk_block_type == "classic":
                    self.blocks.append(
                        ClassicResidualBlock(
                            num_channels,
                            conv_size,
                            version=version,
                            incoming_var=float(i + 1),
                            name=f"classic_res_{i}",
                        )
                    )
                elif trunk_block_type == "nbt":
                    self.blocks.append(
                        NbtResidualBlock(
                            num_channels,
                            num_bottleneck_channels,
                            conv_size,
                            version=version,
                            incoming_var=float(i + 1),
                            name=f"nbt_res_{i}",
                        )
                    )

        self.policy_head = PolicyHead(
            channels=num_head_channels,
            version=self.version,
            use_var_norm=self.use_var_norm,
            incoming_var=float(num_blocks + 1),
            name="policy_head",
        )
        self.value_head = ValueHead(
            num_head_channels,
            c_val,
            version=self.version,
            incoming_var=float(num_blocks + 1),
            name="value_head",
        )

        # v1 always uses one-batch-norm: duplicate heads with batch norm on trunk input
        if version >= 1:
            self.trunk_bn = keras.layers.BatchNormalization(
                momentum=0.999, epsilon=1e-3, name="trunk_bn"
            )
            self.policy_head_bn = PolicyHead(
                channels=num_head_channels,
                version=self.version,
                use_var_norm=self.use_var_norm,
                incoming_var=1.0,  # using batchnorm
                name="policy_head_bn",
            )
            self.value_head_bn = ValueHead(
                num_head_channels,
                c_val,
                version=self.version,
                incoming_var=1.0,  # using batchnorm
                name="value_head_bn",
            )

        ## Initialize Loss Objects. Defer reduction strategy to loss objects ##
        self.scce_logits = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.scce = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.mse = keras.losses.MeanSquaredError()
        self.identity = keras.layers.Activation("linear")  # need for mixed-precision

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
        self.trunk_block_type = trunk_block_type

    def call(self, board_state, game_state, training=False, scores=None):
        if scores is None:
            # Use tf.range with explicit dtype to ensure shape inference works
            scores = (
                0.05 * tf.range(-SCORE_RANGE // 2, SCORE_RANGE // 2, dtype=tf.float32)
                + 0.025
            )

        x = self.init_board_conv(board_state, training=training)
        game_state_biases = self.init_game_layer(game_state)

        x = keras.ops.transpose(x, axes=(1, 2, 0, 3))  # NHWC -> HWNC

        x = x + game_state_biases

        x = keras.ops.transpose(x, axes=(2, 0, 1, 3))  # HWNC -> NHWC

        for block in self.blocks:
            x = block(x, training=training)

        # v0: No batch norm heads
        if self.version == 0:
            pi_logits, pi_logits_aux = self.policy_head(x, training=training)
            (
                outcome_logits,
                ownership,
                score_logits,
                gamma,
                q6,
                q16,
                q50,
            ) = self.value_head(x, scores=scores)
            pi = keras.activations.softmax(pi_logits)
            outcome_probs = keras.activations.softmax(outcome_logits)
            score_probs = keras.activations.softmax(score_logits)
            return (
                keras.ops.cast(pi_logits, "float32"),
                keras.ops.cast(pi, "float32"),
                keras.ops.cast(outcome_logits, "float32"),
                keras.ops.cast(outcome_probs, "float32"),
                keras.ops.cast(ownership, "float32"),
                keras.ops.cast(score_logits, "float32"),
                keras.ops.cast(score_probs, "float32"),
                keras.ops.cast(gamma, "float32"),
                keras.ops.cast(pi_logits_aux, "float32"),
                keras.ops.cast(q6, "float32"),
                keras.ops.cast(q16, "float32"),
                keras.ops.cast(q50, "float32"),
            )

        pi_logits, pi_logits_aux, pi_logits_soft, pi_logits_optimistic = (
            self.policy_head(x, training=training)
        )
        (
            outcome_logits,
            ownership,
            score_logits,
            gamma,
            q6,
            q16,
            q50,
            q6_err,
            q16_err,
            q50_err,
            q6_score,
            q16_score,
            q50_score,
            q6_score_err,
            q16_score_err,
            q50_score_err,
        ) = self.value_head(x, scores=scores)
        pi = keras.activations.softmax(pi_logits)
        outcome_probs = keras.activations.softmax(outcome_logits)
        score_probs = keras.activations.softmax(score_logits)

        # v1: Always use one-batch-norm (compute BN heads)
        x_bn = self.trunk_bn(x, training=training)
        pi_logits_bn, pi_logits_aux_bn, pi_logits_soft_bn, pi_logits_optimistic_bn = (
            self.policy_head_bn(x_bn, training=training)
        )
        (
            outcome_logits_bn,
            ownership_bn,
            score_logits_bn,
            gamma_bn,
            q6_bn,
            q16_bn,
            q50_bn,
            q6_err_bn,
            q16_err_bn,
            q50_err_bn,
            q6_score_bn,
            q16_score_bn,
            q50_score_bn,
            q6_score_err_bn,
            q16_score_err_bn,
            q50_score_err_bn,
        ) = self.value_head_bn(x_bn, scores=scores)
        pi_bn = keras.activations.softmax(pi_logits_bn)
        outcome_probs_bn = keras.activations.softmax(outcome_logits_bn)
        score_probs_bn = keras.activations.softmax(score_logits_bn)

        # v1 + one-batch-norm: 23 FVI outputs + 23 BN outputs = 46 total
        return (
            # FVI outputs (23)
            keras.ops.cast(pi_logits, "float32"),
            keras.ops.cast(pi, "float32"),
            keras.ops.cast(outcome_logits, "float32"),
            keras.ops.cast(outcome_probs, "float32"),
            keras.ops.cast(ownership, "float32"),
            keras.ops.cast(score_logits, "float32"),
            keras.ops.cast(score_probs, "float32"),
            keras.ops.cast(gamma, "float32"),
            keras.ops.cast(pi_logits_aux, "float32"),
            keras.ops.cast(q6, "float32"),
            keras.ops.cast(q16, "float32"),
            keras.ops.cast(q50, "float32"),
            keras.ops.cast(q6_err, "float32"),
            keras.ops.cast(q16_err, "float32"),
            keras.ops.cast(q50_err, "float32"),
            keras.ops.cast(q6_score, "float32"),
            keras.ops.cast(q16_score, "float32"),
            keras.ops.cast(q50_score, "float32"),
            keras.ops.cast(q6_score_err, "float32"),
            keras.ops.cast(q16_score_err, "float32"),
            keras.ops.cast(q50_score_err, "float32"),
            keras.ops.cast(pi_logits_soft, "float32"),
            keras.ops.cast(pi_logits_optimistic, "float32"),
            # BN outputs (23)
            keras.ops.cast(pi_logits_bn, "float32"),
            keras.ops.cast(pi_bn, "float32"),
            keras.ops.cast(outcome_logits_bn, "float32"),
            keras.ops.cast(outcome_probs_bn, "float32"),
            keras.ops.cast(ownership_bn, "float32"),
            keras.ops.cast(score_logits_bn, "float32"),
            keras.ops.cast(score_probs_bn, "float32"),
            keras.ops.cast(gamma_bn, "float32"),
            keras.ops.cast(pi_logits_aux_bn, "float32"),
            keras.ops.cast(q6_bn, "float32"),
            keras.ops.cast(q16_bn, "float32"),
            keras.ops.cast(q50_bn, "float32"),
            keras.ops.cast(q6_err_bn, "float32"),
            keras.ops.cast(q16_err_bn, "float32"),
            keras.ops.cast(q50_err_bn, "float32"),
            keras.ops.cast(q6_score_bn, "float32"),
            keras.ops.cast(q16_score_bn, "float32"),
            keras.ops.cast(q50_score_bn, "float32"),
            keras.ops.cast(q6_score_err_bn, "float32"),
            keras.ops.cast(q16_score_err_bn, "float32"),
            keras.ops.cast(q50_score_err_bn, "float32"),
            keras.ops.cast(pi_logits_soft_bn, "float32"),
            keras.ops.cast(pi_logits_optimistic_bn, "float32"),
        )

    def compute_losses(
        self,
        predictions: ModelPredictions,
        targets: GroundTruth,
        weights: LossWeights,
    ):
        # Policy Loss
        pi_probs = keras.activations.softmax(tf.cast(predictions.pi_logits, tf.float32))
        policy_loss = keras.metrics.kl_divergence(
            tf.cast(targets.policy, tf.float32), pi_probs
        )
        policy_loss = tf.reduce_mean(policy_loss)
        policy_aux_loss = self.scce_logits(
            targets.policy_aux, predictions.pi_logits_aux
        )

        # Outcome Loss
        did_win = targets.score >= 0
        outcome_loss = self.scce_logits(did_win, predictions.game_outcome)
        q6_loss = self.mse(targets.q6, predictions.q6_pred)
        q16_loss = self.mse(targets.q16, predictions.q16_pred)
        q50_loss = self.mse(targets.q50, predictions.q50_pred)

        # Score Loss
        score_index = targets.score + SCORE_RANGE_MIDPOINT
        score_distribution = keras.activations.softmax(predictions.score_logits)
        score_pdf_loss = self.scce(score_index, score_distribution)
        score_cdf_loss = tf.math.reduce_mean(
            tf.math.reduce_sum(
                tf.math.square(
                    tf.math.cumsum(targets.score_one_hot, axis=1)
                    - tf.math.cumsum(score_distribution, axis=1)
                ),
                axis=1,
            )
        )

        # Ownership Loss
        own_pred_squeezed = tf.squeeze(predictions.own_pred, -1)  # tailing 1 dim.
        own_loss = self.mse(targets.own, own_pred_squeezed)

        gamma_squeezed = tf.squeeze(predictions.gamma, axis=-1)
        gamma_loss = tf.math.reduce_mean(
            gamma_squeezed * gamma_squeezed * weights.w_gamma
        )

        # Weight everything
        woutcome_loss = weights.w_outcome * outcome_loss
        wq6_loss = weights.w_q6 * q6_loss
        wq16_loss = weights.w_q16 * q16_loss
        wq50_loss = weights.w_q50 * q50_loss
        wscore_pdf_loss = weights.w_score * score_pdf_loss
        wscore_cdf_loss = weights.w_score * score_cdf_loss
        wown_loss = weights.w_own * own_loss
        val_loss = (
            weights.w_val
            * (
                woutcome_loss
                + wq6_loss
                + wq16_loss
                + wq50_loss
                + wscore_pdf_loss
                + wown_loss
            )
            + wscore_cdf_loss  # Outside w_val to prevent score variance
        )

        loss = (
            weights.w_pi * tf.cast(policy_loss, tf.float32)
            + weights.w_pi_aux * tf.cast(policy_aux_loss, tf.float32)
            + tf.cast(val_loss, tf.float32)
            + tf.cast(gamma_loss, tf.float32)
        )

        # v1 losses (only computed if v1 outputs are provided)
        q_err_loss = tf.constant(0.0)
        q_score_loss = tf.constant(0.0)
        q_score_err_loss = tf.constant(0.0)
        pi_soft_loss = tf.constant(0.0)
        pi_optimistic_loss = tf.constant(0.0)

        if self.version >= 1 and predictions.q6_err_pred is not None:
            (
                q_err_loss,
                q_score_loss,
                q_score_err_loss,
                pi_soft_loss,
                pi_optimistic_loss,
            ) = self.v1_loss_terms(predictions, targets)

            # Add v1 losses to total
            loss = loss + (
                weights.w_q_err * q_err_loss
                + weights.w_q_score * q_score_loss
                + weights.w_q_score_err * q_score_err_loss
                + weights.w_pi_soft * pi_soft_loss
                + weights.w_pi_optimistic * pi_optimistic_loss
            )

        return (
            loss,
            policy_loss,
            policy_aux_loss,
            outcome_loss,
            q6_loss,
            q16_loss,
            q50_loss,
            score_pdf_loss,
            score_cdf_loss,
            own_loss,
            # v1 losses (for logging)
            q_err_loss,
            q_score_loss,
            q_score_err_loss,
            pi_soft_loss,
            pi_optimistic_loss,
        )

    def v1_loss_terms(
        self,
        predictions: ModelPredictions,
        targets: GroundTruth,
    ):
        """
        Compute v1-specific losses.

        Returns:
            Tuple of (q_err_loss, q_score_loss, q_score_err_loss, pi_soft_loss, pi_optimistic_loss)
        """
        epsilon = 1e-6
        huber = keras.losses.Huber(reduction="none")
        huber_score = keras.losses.Huber(reduction="none", delta=10.0)

        # Q error losses (outputs 12-14): Huber loss
        # Target is squared diff between NN prediction and ground truth
        # Use stop_gradient on predictions used as targets
        q6_err_target = tf.square(tf.stop_gradient(predictions.q6_pred) - targets.q6)
        q16_err_target = tf.square(tf.stop_gradient(predictions.q16_pred) - targets.q16)
        q50_err_target = tf.square(tf.stop_gradient(predictions.q50_pred) - targets.q50)

        q6_err_loss = tf.reduce_mean(huber(q6_err_target, predictions.q6_err_pred))
        q16_err_loss = tf.reduce_mean(huber(q16_err_target, predictions.q16_err_pred))
        q50_err_loss = tf.reduce_mean(huber(q50_err_target, predictions.q50_err_pred))
        q_err_loss = q6_err_loss + q16_err_loss + q50_err_loss

        # Q score losses (outputs 15-17): Huber loss on score predictions
        q_score_loss = tf.constant(0.0)
        if targets.q6_score is not None:
            q6_score_loss = tf.reduce_mean(
                huber_score(targets.q6_score, predictions.q6_score_pred)
            )
            q16_score_loss = tf.reduce_mean(
                huber_score(targets.q16_score, predictions.q16_score_pred)
            )
            q50_score_loss = tf.reduce_mean(
                huber_score(targets.q50_score, predictions.q50_score_pred)
            )
            q_score_loss = q6_score_loss + q16_score_loss + q50_score_loss

        # Q score error losses (outputs 18-20): Huber loss
        # Use stop_gradient on predictions used as targets
        q_score_err_loss = tf.constant(0.0)
        if targets.q6_score is not None:
            q6_score_err_target = tf.square(
                tf.stop_gradient(predictions.q6_score_pred) - targets.q6_score
            )
            q16_score_err_target = tf.square(
                tf.stop_gradient(predictions.q16_score_pred) - targets.q16_score
            )
            q50_score_err_target = tf.square(
                tf.stop_gradient(predictions.q50_score_pred) - targets.q50_score
            )

            q6_score_err_loss = tf.reduce_mean(
                huber_score(q6_score_err_target, predictions.q6_score_err_pred)
            )
            q16_score_err_loss = tf.reduce_mean(
                huber_score(q16_score_err_target, predictions.q16_score_err_pred)
            )
            q50_score_err_loss = tf.reduce_mean(
                huber_score(q50_score_err_target, predictions.q50_score_err_pred)
            )
            q_score_err_loss = (
                q6_score_err_loss + q16_score_err_loss + q50_score_err_loss
            )

        # Soft policy loss (output 21): KLD on policy^0.25
        policy_f32 = tf.cast(targets.policy, tf.float32)
        policy_soft = tf.pow(policy_f32, 0.25)
        policy_soft = policy_soft / tf.reduce_sum(policy_soft, axis=-1, keepdims=True)

        # Compute KLD loss
        pi_soft_probs = keras.activations.softmax(
            tf.cast(predictions.pi_logits_soft, tf.float32)
        )
        pi_soft_loss = keras.metrics.kl_divergence(policy_soft, pi_soft_probs)
        pi_soft_loss = tf.reduce_mean(pi_soft_loss)

        # Optimistic policy loss (output 22): KLD with weighted policy target
        # Weight = clamp(0, 1, sigmoid((z_value-1.5)*3) + sigmoid((z_score-1.5)*3))
        # z_value = (q6 - stop_grad(q6_pred)) / stop_grad(sqrt(q6_err_pred + eps))
        # Note: q_err_pred is already constrained to be non-negative via sigmoid in ValueHead
        z_value = (targets.q6 - tf.stop_gradient(predictions.q6_pred)) / (
            tf.stop_gradient(tf.sqrt(predictions.q6_err_pred + epsilon))
        )
        if targets.q6_score is not None:
            z_score = (
                targets.q6_score - tf.stop_gradient(predictions.q6_score_pred)
            ) / (tf.stop_gradient(tf.sqrt(predictions.q6_score_err_pred + epsilon)))
        else:
            z_score = tf.zeros_like(z_value)

        optimistic_weight = tf.clip_by_value(
            tf.nn.sigmoid((z_value - 1.5) * 3) + tf.nn.sigmoid((z_score - 1.5) * 3),
            0.0,
            1.0,
        )
        # Expand weight to match policy shape for element-wise multiplication
        optimistic_weight = tf.expand_dims(optimistic_weight, axis=-1)

        # Weighted policy target
        weighted_policy = tf.cast(targets.policy, tf.float32) * optimistic_weight
        # Renormalize
        weighted_policy = weighted_policy / (
            tf.reduce_sum(weighted_policy, axis=-1, keepdims=True) + epsilon
        )

        pi_optimistic_probs = keras.activations.softmax(
            tf.cast(predictions.pi_logits_optimistic, tf.float32)
        )
        pi_optimistic_loss = keras.metrics.kl_divergence(
            weighted_policy, pi_optimistic_probs
        )
        pi_optimistic_loss = tf.reduce_mean(pi_optimistic_loss)

        return (
            q_err_loss,
            q_score_loss,
            q_score_err_loss,
            pi_soft_loss,
            pi_optimistic_loss,
        )

    def get_config(self):
        return {
            "board_len": self.board_len,
            "num_input_planes": self.num_input_planes,
            "num_input_features": self.num_input_features,
            "num_blocks": self.num_blocks,
            "num_channels": self.num_channels,
            "num_bottleneck_channels": self.num_bottleneck_channels,
            "num_head_channels": self.num_head_channels,
            "c_val": self.c_val,
            "bottleneck_length": self.bottleneck_length,
            "conv_size": self.conv_size,
            "broadcast_interval": self.broadcast_interval,
            "trunk_block_type": self.trunk_block_type,
            "version": self.version,
            "name": self.name,
        }

    def get_build_config(self):
        """Return config needed to rebuild the model's state."""
        return {
            "input_shape": [None, self.board_len, self.board_len, self.num_input_planes],
            "game_state_shape": [None, self.num_input_features],
        }

    def build_from_config(self, config):
        """Build the model from the saved build config."""
        input_shape = config.get("input_shape")
        game_shape = config.get("game_state_shape")

        if input_shape is not None:
            # Extract spatial dimensions, ignoring batch
            if isinstance(input_shape, list) and len(input_shape) >= 4:
                board_input = keras.ops.zeros([1, input_shape[-3], input_shape[-2], input_shape[-1]])
            else:
                board_input = keras.ops.zeros([1, self.board_len, self.board_len, self.num_input_planes])

            if game_shape is not None and isinstance(game_shape, list):
                game_input = keras.ops.zeros([1, game_shape[-1]])
            else:
                game_input = keras.ops.zeros([1, self.num_input_features])

            # Call the model to trigger building all layers
            self(board_input, game_input, training=False)

    def summary(self):
        x0 = keras.layers.Input(
            shape=(self.board_len, self.board_len, self.num_input_planes)
        )
        x1 = keras.layers.Input(shape=(self.num_input_features,))
        model = keras.Model(inputs=[x0, x1], outputs=self.call(x0, x1))
        model.summary()

    def input_planes_shape(self):
        return [self.board_len, self.board_len, self.num_input_planes]

    def input_features_shape(self):
        return [self.num_input_features]

    @staticmethod
    def create(
        config: ModelConfig,
        board_len: int,
        num_input_planes: int,
        num_input_features: int,
        name: str,
    ):
        return P3achyGoModel(
            board_len,
            num_input_planes,
            num_input_features,
            config.kBlocks,
            config.kChannels,
            config.kBottleneckChannels,
            config.kHeadChannels,
            config.kCVal,
            config.kInnerBottleneckLayers + 2,
            config.kConvSize,
            config.kBroadcastInterval,
            config.kTrunkBlockType,
            config.kVersion,
            name=name,
        )

    @staticmethod
    def custom_objects():
        return {
            "ConvBlock": ConvBlock,
            "ConvPostActivation": ConvPostActivation,
            "ConvPreActivation": ConvPreActivation,
            "ResidualBlock": ResidualBlock,
            "ClassicResidualBlock": ClassicResidualBlock,
            "BottleneckResidualConvBlock": BottleneckResidualConvBlock,
            "NbtResidualBlock": NbtResidualBlock,
            "BroadcastResidualBlock": BroadcastResidualBlock,
            "Broadcast": BroadcastResidualBlock.Broadcast,
            "BroadcastPostAct": BroadcastResidualBlock.BroadcastPostAct,
            "BroadcastPreAct": BroadcastResidualBlock.BroadcastPreAct,
            "GlobalPoolBias": GlobalPoolBias,
            "GlobalPool": GlobalPool,
            "PolicyHead": PolicyHead,
            "ValueHead": ValueHead,
            "P3achyGoModel": P3achyGoModel,
        }
