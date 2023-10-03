import tensorflow as tf
import transforms
import tf2onnx, onnx

from pathlib import Path
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from model import P3achyGoModel
from constants import *

NUM_CALIB_BATCHES = 10


def get_converter(model_path: str, chunk_path: str,
                  batch_size: int) -> trt.TrtGraphConverterV2:
  calib_ds = tf.data.TFRecordDataset(chunk_path, compression_type='ZLIB')
  calib_ds = calib_ds.map(transforms.expand)
  calib_ds = calib_ds.batch(batch_size)
  calib_ds = calib_ds.take(NUM_CALIB_BATCHES)

  def calibration_input_fn():
    for input, input_global_state, _, _, _, _, _, _, _, _, _, _ in calib_ds:
      yield input, input_global_state

  def input_fn():
    input, input_global_state, _, _, _, _, _, _, _, _, _, _ = next(
        iter(calib_ds))
    yield input, input_global_state

  # Instantiate the TF-TRT converter
  converter = trt.TrtGraphConverterV2(
      input_saved_model_dir=model_path,
      precision_mode=trt.TrtPrecisionMode.INT8,
      use_calibration=True,
      use_dynamic_shape=False,
  )
  converter.convert(calibration_input_fn=calibration_input_fn)
  converter.build(input_fn=input_fn)

  return converter


def convert_onnx(model_path: str) -> str:
  with tf.device("/cpu:0"):
    # Load model and resave without mixed precision.
    tf.keras.mixed_precision.set_global_policy('float32')
    model = tf.keras.models.load_model(
        model_path, custom_objects=P3achyGoModel.custom_objects())
    onnx_path = str(Path(model_path, '_onnx', 'model.onnx'))

    @tf.function
    def model_fn(board_state: tf.Tensor, game_state: tf.Tensor,
                 scores: tf.Tensor):
      (pi_logits, pi, outcome_logits, outcome, own, score_logits, score_probs,
       gamma, pi_logits_aux, q30, q100, q200) = model(board_state,
                                                      game_state,
                                                      training=False,
                                                      scores=scores)

      return {
          '00:pi_logits': pi_logits,
          '01:pi': pi,
          '02:outcome_logits': outcome_logits,
          '03:outcome': outcome,
          '04:own': own,
          '05:score_logits': score_logits,
          '06:score_probs': score_probs,
          '07:gamma': gamma,
          '08:pi_logits_aux': pi_logits_aux,
          '09:q30': q30,
          '10:q100': q100,
          '11:q200': q200,
      }

    input_signature = [
        tf.TensorSpec(shape=[None] + model.input_planes_shape(),
                      dtype=tf.float32,
                      name='board_state'),
        tf.TensorSpec(shape=[None] + model.input_features_shape(),
                      dtype=tf.float32,
                      name='game_state'),
        # need this b/c otherwise ONNX loses an edge to the score tensor.
        tf.TensorSpec(shape=(SCORE_RANGE,), dtype=tf.float32, name='scores')
    ]
    onnx_model, _ = tf2onnx.convert.from_function(
        model_fn, input_signature=input_signature)

    Path(model_path, '_onnx').mkdir(exist_ok=True)
    onnx.save(onnx_model, onnx_path)
    return onnx_path
