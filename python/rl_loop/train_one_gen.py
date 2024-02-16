'''
Trains a model for one generation and writes it back to disk.
'''
from __future__ import annotations

import sys
import gcs_utils as gcs
import tensorflow as tf
import transforms
import rl_loop.model_utils as model_utils
import rl_loop.train
import rl_loop.config

from absl import app, flags, logging
from model import P3achyGoModel
from pathlib import Path
from rl_loop.trt_batch_size import trt_batch_size
from typing import Tuple

BATCH_SIZE = 256

FLAGS = flags.FLAGS

flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('models_dir', '', 'Directory to find model.')
flags.DEFINE_integer('gen', -1, 'Generation, or -1 for most recent')
flags.DEFINE_integer('next_gen', -1, 'Next generation.')
flags.DEFINE_string('chunk_path', '', 'Path to training chunk')
flags.DEFINE_integer(
    'chunk_size', None,
    'Size of chunk. Used for cyclic LR. Specify `None` to use linear LR.')
flags.DEFINE_string('val_ds_path', '', 'Path to val ds')
flags.DEFINE_string('batch_num_path', '', 'File storing batch counter.')
flags.DEFINE_boolean('save_trt', True, 'Whether to save TRT converted model.')
flags.DEFINE_string('trt_convert_path', '', 'Path to TRT convert binary.')


def get_model_path(models_dir: str, gen: int) -> Tuple[str, int]:
  if gen != -1:
    model_path = gcs.MODEL_FORMAT.format(gen)
    return str(Path(models_dir, model_path)), gen

  model_paths = Path(models_dir).glob('**/*')
  model_paths = [f for f in model_paths if gcs.MODEL_RE.fullmatch(f.name)]
  model_paths = sorted(model_paths,
                       key=lambda f: gcs.MODEL_RE.fullmatch(f.name).group(1))
  model_path = model_paths[-1]
  return str(model_path), gcs.MODEL_RE.fullmatch(model_path).group(1)


def main(_):
  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return
  if FLAGS.models_dir == '':
    logging.error('No --models_dir specified.')
    return
  if FLAGS.next_gen == -1:
    logging.error('No --next_gen specified.')
    return
  if FLAGS.chunk_path == '':
    logging.error('No --chunk_path specified.')
    return
  if FLAGS.val_ds_path == '':
    logging.error('No --val_ds_path specified.')
    return
  if FLAGS.batch_num_path == '':
    logging.error('No --batch_num_path specified.')
    return
  if FLAGS.save_trt and FLAGS.trt_convert_path == '':
    logging.error('No --trt_convert_path specified.')
    return

  is_gpu = False
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    is_gpu = True
  else:
    logging.warning('No GPU detected.')

  config = rl_loop.config.parse(FLAGS.run_id)
  val_ds = tf.data.TFRecordDataset(FLAGS.val_ds_path, compression_type='ZLIB')
  val_ds = val_ds.map(transforms.expand, num_parallel_calls=tf.data.AUTOTUNE)
  val_ds = val_ds.batch(config.batch_size)
  val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

  model_path, _ = get_model_path(FLAGS.models_dir, FLAGS.gen)
  model = tf.keras.models.load_model(
      model_path, custom_objects=P3achyGoModel.custom_objects())

  with open(FLAGS.batch_num_path, 'r') as f:  # assumes file is already created.
    batch_num = int(f.read())

  logging.info(
      f'Model Path: {model_path}, Training on Chunk: {FLAGS.chunk_path}')
  batch_num = rl_loop.train.train_one_gen(model,
                                          FLAGS.gen,
                                          FLAGS.chunk_path,
                                          val_ds,
                                          config=config,
                                          is_gpu=is_gpu,
                                          batch_num=batch_num,
                                          chunk_size=FLAGS.chunk_size)
  if FLAGS.save_trt:
    model_utils.save_onnx_trt(model,
                              FLAGS.val_ds_path,
                              FLAGS.models_dir,
                              FLAGS.next_gen,
                              batch_size=trt_batch_size(),
                              trt_convert_path=FLAGS.trt_convert_path)
  else:
    model_utils.save(model, FLAGS.models_dir, FLAGS.next_gen)

  with open(FLAGS.batch_num_path, 'w') as f:
    f.write(str(batch_num))


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
