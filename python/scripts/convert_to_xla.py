import tensorflow as tf
import numpy as np

from absl import app, flags, logging
from pathlib import Path

from tensorflow.python.framework import convert_to_constants

from model import P3achyGoModel
from constants import *

FLAGS = flags.FLAGS
DUMMY_BATCH_SIZE = 32

flags.DEFINE_string('model_path', '', 'Path to SavedModel.')
flags.DEFINE_string('xla_name', 'model.pb', 'Name of XLA model.')


def random_inputs(planes_shape, features_shape):
  return (np.random.random([DUMMY_BATCH_SIZE] + planes_shape).astype(
      np.float32), np.random.random([DUMMY_BATCH_SIZE] + features_shape).astype(
          np.float32))


def main(_):
  if not FLAGS.model_path:
    logging.warning('No Model Path Specified.')
    return

  tf.keras.mixed_precision.set_global_policy('mixed_float16')

  model_path = FLAGS.model_path
  model_xla_dir = str(Path(model_path, '_xla'))
  logging.info(f'Model Path: {model_path}')
  logging.info(f'Model XLA Path: {model_xla_dir + "/" + FLAGS.xla_name}')

  model = tf.keras.models.load_model(
      model_path, custom_objects=P3achyGoModel.custom_objects())
  planes_shape = model.input_planes_shape()
  features_shape = model.input_features_shape()
  model(*[
      tf.convert_to_tensor(x)
      for x in random_inputs(planes_shape, features_shape)
  ])
  model.summary()

  @tf.function(experimental_compile=True)
  def model_fn(board_state: tf.Tensor, game_state: tf.Tensor):
    (pi_logits, pi, outcome_logits, outcome, own, score_logits, score_probs,
     gamma, pi_logits_aux, q30, q100, q200) = model(board_state,
                                                    game_state,
                                                    training=False)

    return (
        tf.identity(pi_logits, name='pi_logits'),
        tf.identity(q30, name='q30'),
        tf.identity(q100, name='q100'),
        tf.identity(q200, name='q200'),
        tf.identity(pi, name='pi'),
        tf.identity(outcome_logits, name='outcome_logits'),
        tf.identity(outcome, name='outcome'),
        tf.identity(own, name='own'),
        tf.identity(score_logits, name='score_logits'),
        tf.identity(score_probs, name='score_probs'),
        tf.identity(gamma, name='gamma'),
        tf.identity(pi_logits_aux, name='pi_logits_aux'),
    )

  input_signature = [
      tf.TensorSpec(shape=[None] + model.input_planes_shape(),
                    dtype=tf.float32,
                    name='board_state'),
      tf.TensorSpec(shape=[None] + model.input_features_shape(),
                    dtype=tf.float32,
                    name='game_state'),
  ]

  fn = model_fn.get_concrete_function(*input_signature)
  frozen_fn = convert_to_constants.convert_variables_to_constants_v2(fn)
  frozen_fn.graph.as_graph_def()

  tf.io.write_graph(graph_or_graph_def=frozen_fn.graph,
                    logdir=str(model_xla_dir),
                    name=str(FLAGS.xla_name),
                    as_text=False)


if __name__ == '__main__':
  app.run(main)
