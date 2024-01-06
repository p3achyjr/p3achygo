import tensorflow as tf
import tvm
import numpy as np

from absl import app, flags, logging
from pathlib import Path
from tvm import relay

from tensorflow.python.framework import convert_to_constants

from model import P3achyGoModel
from constants import *

FLAGS = flags.FLAGS
DUMMY_BATCH_SIZE = 32

flags.DEFINE_string('model_path', '', 'Path to SavedModel.')
flags.DEFINE_string('tvm_name', 'model', 'Name of TVM files.')


def random_inputs(planes_shape, features_shape):
  return (np.random.random([DUMMY_BATCH_SIZE] + planes_shape).astype(
      np.float32), np.random.random([DUMMY_BATCH_SIZE] + features_shape).astype(
          np.float32))


def main(_):
  if not FLAGS.model_path:
    logging.warning('No Model Path Specified.')
    return

  for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

  model_path = FLAGS.model_path
  model_tvm_dir = str(Path(model_path, '_tvm'))
  logging.info(f'Model Path: {model_path}')
  logging.info(f'Model TVM Path: {model_tvm_dir + "/" + FLAGS.tvm_name}')

  model = tf.keras.models.load_model(
      model_path, custom_objects=P3achyGoModel.custom_objects())
  planes_shape = model.input_planes_shape()
  features_shape = model.input_features_shape()
  model(*[
      tf.convert_to_tensor(x)
      for x in random_inputs(planes_shape, features_shape)
  ])
  model.summary()

  input_signature = [
      tf.TensorSpec(shape=[None] + model.input_planes_shape(),
                    dtype=tf.float32,
                    name='board_state'),
      tf.TensorSpec(shape=[None] + model.input_features_shape(),
                    dtype=tf.float32,
                    name='game_state'),
  ]

  @tf.function(input_signature=input_signature)
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

  fn = model_fn.get_concrete_function(*input_signature)
  frozen_fn = convert_to_constants.convert_variables_to_constants_v2(fn)
  frozen_fn.graph.as_graph_def()
  relay_mod, relay_params = relay.frontend.from_tensorflow(
      frozen_fn.graph.as_graph_def(), layout='NHWC')
  # desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
  # shape_dict = {inp.op.name: inp.shape for inp in model.inputs}
  # mod, params = relay.frontend.from_keras(model, shape_dict)
  seq = tvm.transform.Sequential([
      relay.transform.RemoveUnusedFunctions(),
      relay.transform.ConvertLayout(desired_layouts),
      relay.transform.ToMixedPrecision(dtype="float16", skip_conv_layers=False)
  ])
  with tvm.transform.PassContext(opt_level=3):
    mod = seq(mod)

  # Ensure inputs and outputs are in FP32
  mod["main"] = relay.Function(mod["main"].params,
                               relay.cast(mod["main"].body, "float32"))
  target = "cuda"
  with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(relay_mod, target, params=relay_params)

  # Save the compiled model
  tvm_model_path = str(Path(model_tvm_dir, FLAGS.tvm_name + '.tar'))
  tvm_params_path = str(Path(model_tvm_dir, FLAGS.tvm_name + '.params'))
  lib.export_library(tvm_model_path)
  with open(tvm_params_path, "wb") as f:
    f.write(relay.save_param_dict(relay_params))


if __name__ == '__main__':
  app.run(main)
