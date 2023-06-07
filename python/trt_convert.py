import tensorflow as tf
import transforms
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from pathlib import Path

# Use the same batch size when running self-play.
BATCH_SIZE = 48
NUM_CALIB_BATCHES = 10


def get_converter(model_path: str, chunk_path: str) -> trt.TrtGraphConverterV2:
  calib_ds = tf.data.TFRecordDataset(chunk_path, compression_type='ZLIB')
  calib_ds = calib_ds.map(transforms.expand_rl)
  calib_ds = calib_ds.batch(BATCH_SIZE)
  calib_ds = calib_ds.take(NUM_CALIB_BATCHES)

  def calibration_input_fn():
    for input, komi, _, _, _, _ in calib_ds:
      yield input, komi

  def input_fn():
    input, komi, _, _, _, _ = next(iter(calib_ds))
    yield input, komi

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
