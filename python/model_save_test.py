import numpy as np
import tensorflow as tf
from model import P3achyGoModel, custom_objects_dict_for_serialization
from model_config import ModelConfig

model = P3achyGoModel.create(ModelConfig.tiny(), 'test')
test_input = np.random.random((32, 19, 19, 7))
test_output = model(test_input)

model.save('/tmp/test')

loaded_1 = tf.keras.models.load_model(
    '/tmp/test', custom_objects=custom_objects_dict_for_serialization())

print(model)
print(loaded_1)

np.testing.assert_allclose(loaded_1(test_input), test_output)
