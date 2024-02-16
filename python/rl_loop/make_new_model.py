import rl_loop.model_utils as model_utils
import sys
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from model_config import CONFIG_OPTIONS

FLAGS = flags.FLAGS

flags.DEFINE_enum('model_config', 'small', CONFIG_OPTIONS, 'Model Config/Size.')
flags.DEFINE_string('model_path', '', 'Path to store model')


def main(_):
  if FLAGS.model_config == '':
    logging.error('No --model_config specified.')
    return
  if FLAGS.model_path == '':
    logging.error('No --model_path specified.')
    return

  with tf.device('/cpu:0'):
    batch_size = 32
    model = model_utils.new_model(name=f'p3achygo',
                                  model_config=FLAGS.model_config)
    model(
        tf.convert_to_tensor(np.random.random([batch_size] +
                                              model.input_planes_shape()),
                             dtype=tf.float32),
        tf.convert_to_tensor(np.random.random([batch_size] +
                                              model.input_features_shape()),
                             dtype=tf.float32))
    model.summary()
    model.save(FLAGS.model_path)


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
