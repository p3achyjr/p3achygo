import tensorflow as tf
import numpy as np
from model import P3achyGoModel
from model_config import ModelConfig

# Create a simple config (you may need to adjust based on your ModelConfig class)
model = P3achyGoModel.create(ModelConfig.b10c128btl3(), 19, 15, 8, "test")

# Create dummy inputs
batch_size = 2
board_state = tf.random.uniform([batch_size, 19, 19, 15], dtype=tf.float32)
game_state = tf.random.uniform([batch_size, 8], dtype=tf.float32)

# Run once to build the model
outputs = model(board_state, game_state, training=False)

# Print shapes
for i, out in enumerate(outputs):
    if out is not None:
        print(f"Output {i}: {out.shape}")
    else:
        print(f"Output {i}: None")

model.summary()
