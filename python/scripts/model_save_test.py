import numpy as np
import tensorflow as tf
from constants import *
from model import P3achyGoModel
from model_config import ModelConfig

model = P3achyGoModel.create(ModelConfig.tiny(),
                             board_len=BOARD_LEN,
                             num_input_planes=NUM_INPUT_PLANES,
                             num_input_features=NUM_INPUT_FEATURES,
                             name='test')
test_input = np.random.random((32, 19, 19, 7))
test_output = model(test_input)

model.save('/tmp/test')

loaded_1 = tf.keras.models.load_model(
    '/tmp/test', custom_objects=P3achyGoModel.custom_objects())

print(model)
print(loaded_1)

np.testing.assert_allclose(loaded_1(test_input), test_output)
