"""Is keras BN training-mode forward deterministic across calls?"""

import os, sys

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "python")
sys.path.insert(0, "python/test")

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import keras

np.random.seed(7)
W_conv = np.random.randn(3, 3, 4, 8).astype(np.float32)
W_fc = np.random.randn(8, 2).astype(np.float32)
np.random.seed(42)
x = np.random.randn(8, 5, 5, 4).astype(np.float32)
y = np.arange(8) % 2

inp = keras.Input(shape=(5, 5, 4))
h = keras.layers.Conv2D(8, 3, padding="same", use_bias=False)(inp)
h = keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(h)
h = keras.layers.Activation("relu")(h)
h = keras.layers.GlobalAveragePooling2D()(h)
out = keras.layers.Dense(2)(h)
km = keras.Model(inp, out)
km(np.zeros((1, 5, 5, 4), np.float32))
cl = next(l for l in km.layers if isinstance(l, keras.layers.Conv2D))
bl = next(l for l in km.layers if isinstance(l, keras.layers.BatchNormalization))
dl = next(l for l in km.layers if isinstance(l, keras.layers.Dense))
cl.kernel.assign(W_conv)
dl.kernel.assign(W_fc)
dl.bias.assign(np.zeros(2, np.float32))

# Two consecutive forwards in training=True mode with same weights
o1 = km(x, training=True)
print("forward 1 logits[0]:", o1.numpy()[0])
print("running_mean after fwd1:", bl.moving_mean.numpy())
o2 = km(x, training=True)
print("forward 2 logits[0]:", o2.numpy()[0])
print("running_mean after fwd2:", bl.moving_mean.numpy())
print("logits diff:", float(np.abs(o1.numpy() - o2.numpy()).max()))

# Check training=False
o3 = km(x, training=False)
print("forward training=False logits[0]:", o3.numpy()[0])
