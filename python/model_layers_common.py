import tensorflow as tf
import keras
import math
from absl import logging

L2 = keras.regularizers.L2
C_L2 = 1e-4


# mainly used when c_l2=0, for muon/adamw
def set_global_c_l2(c_l2):
    global C_L2
    C_L2 = c_l2
    logging.info(f"Set global C_L2 to {c_l2}")


def gamma(act):
    if act == keras.activations.relu:
        return 1.712
    elif act == keras.activations.mish:
        return 1.592
    return 2.0**0.5


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


# Currently unused as BatchNorm is much better.
@keras.saving.register_keras_serializable(package="p3achygo")
class ConvSWS(keras.layers.Layer):
    """
    Implements scaled weight standardization
    """

    def __init__(self, output_channels: int, conv_size: int, gamma: float, **kwargs):
        super(ConvSWS, self).__init__(**kwargs)
        self._kern_init = keras.initializers.VarianceScaling(scale=1.0)
        self._gamma = gamma
        self._k = output_channels
        self._r = conv_size
        self._kernel = None

    def build(self, input_shape):
        c = input_shape[-1]
        k = self._k
        r = self._r
        self._kernel = self.add_weight(
            name="kernel",
            shape=(r, r, c, k),  # HWIO
            initializer=self._kern_init,
            trainable=True,
            regularizer=L2(C_L2),
        )

        self.fan_in = r * r * c
        self.fan_in_sqrt = math.sqrt(self.fan_in)

    def call(self, x, training=False):
        w = self._kernel
        eps = 1e-6
        mean = tf.reduce_mean(w, axis=(0, 1, 2), keepdims=True)
        var = tf.reduce_mean(tf.square(w - mean), axis=(0, 1, 2), keepdims=True)
        std = tf.sqrt(var + eps)
        w_hat = (w - mean) / (std * self.fan_in_sqrt)
        w_hat = w_hat * self._gamma
        y = tf.nn.conv2d(
            x,
            w_hat,
            strides=(1, 1, 1, 1),
            padding="SAME",
        )
        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_channels": self._k,
                "conv_size": self._r,
                "gamma": self._gamma,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Need to deserialize initializer manually
        config["kernel_initializer"] = keras.initializers.deserialize(
            config["kernel_initializer"]
        )
        return cls(**config)


@keras.saving.register_keras_serializable(package="p3achygo")
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
            else keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)
        )
        self.activation = activation
        self.variance = variance

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
            name=config.get("name"),
        )


@keras.saving.register_keras_serializable(package="p3achygo")
class ConvPostActivation(ConvBlock):
    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm_layer(x, training=training)
        x = self.activation(x)
        return x


@keras.saving.register_keras_serializable(package="p3achygo")
class ConvPreActivation(ConvBlock):
    def call(self, x, training=False):
        x = self.norm_layer(x, training=training)
        x = self.activation(x)
        x = self.conv(x)
        return x


def make_conv_block(output_channels: int, conv_size: int, variance=1.0, name=None):
    return ConvPreActivation(
        output_channels=output_channels,
        conv_size=conv_size,
        activation=keras.activations.mish,
        use_var_norm=False,
        variance=variance,
        name=name,
    )
