'''
RL Training Task.

Repeatedly polls for a new chunk from GCS, and trains a model on it.
'''

from __future__ import annotations

import gcs_utils as gcs
import sys, time
import tensorflow as tf
import transforms
import train
import rl_loop.model_utils as model_utils
import rl_loop.train

from absl import app, flags, logging
from constants import *
from pathlib import Path
from rl_loop.trt_batch_size import trt_batch_size
from model import P3achyGoModel

sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error

FLAGS = flags.FLAGS

POLL_INTERVAL_S = 10
BATCH_SIZE = 256
LR = 3e-3
EPOCHS_PER_GEN = 1
MOMENTUM = .9

flags.DEFINE_integer('num_generations', 10, 'Number of generations to produce.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '',
                    'Local path to store models and training chunks.')
flags.DEFINE_string('model_checkpoint_dir', '/tmp/p3achygo_checkpoints',
                    'Local path to store intermediate models.')
flags.DEFINE_string('val_ds_path', '',
                    'Local path to validation DS. Also used for calibration')
flags.DEFINE_integer(
    'log_interval', 100,
    'Interval at which to log training information (in mini-batches)')


def main(_):
  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return

  if FLAGS.local_run_dir == '':
    logging.error('No --local_run_dir specified.')
    return

  if FLAGS.val_ds_path == '':
    logging.error('No --val_ds_path specified.')
    return

  is_gpu = False
  if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    is_gpu = True
  else:
    logging.warning('No GPU detected.')

  chunk_dir = Path(FLAGS.local_run_dir, 'chunks')
  model_dir = Path(FLAGS.local_run_dir, 'models')
  ckpt_dir = Path(FLAGS.model_checkpoint_dir)

  chunk_dir.mkdir(exist_ok=True)
  model_dir.mkdir(exist_ok=True)
  ckpt_dir.mkdir(exist_ok=True)

  model_gen, model = 0, None
  if gcs.get_most_recent_model(FLAGS.run_id) < 0:
    model = model_utils.new_model(name=f'p3achygo_{FLAGS.run_id}')
    # upload for self-play to pick up.
    model_utils.save_trt_and_upload(model,
                                    FLAGS.val_ds_path,
                                    str(model_dir),
                                    0,
                                    run_id=FLAGS.run_id,
                                    batch_size=trt_batch_size())
  else:
    model_path = gcs.download_model(FLAGS.run_id, str(model_dir), model_gen)
    model = tf.keras.models.load_model(
        model_path, custom_objects=P3achyGoModel.custom_objects())

  model.summary(batch_size=BATCH_SIZE)
  val_ds = tf.data.TFRecordDataset(FLAGS.val_ds_path, compression_type='ZLIB')
  val_ds = val_ds.map(transforms.expand, num_parallel_calls=tf.data.AUTOTUNE)
  val_ds = val_ds.batch(BATCH_SIZE)
  val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

  logging.info(f'Collecting initial validation stats...')
  train.val(model, mode=train.Mode.RL, val_ds=val_ds)
  while model_gen < FLAGS.num_generations:
    next_model_gen = model_gen + 1
    most_recent_chunk = gcs.get_most_recent_chunk(FLAGS.run_id)
    if most_recent_chunk < next_model_gen:
      # chunk is not ready yet.
      time.sleep(POLL_INTERVAL_S)
      continue

    logging.info(f'Found chunk: {most_recent_chunk}')
    chunk_filename = gcs.download_golden_chunk(FLAGS.run_id, str(chunk_dir),
                                               next_model_gen)
    logging.info(f'Training model gen {model_gen}...')
    rl_loop.train.train_one_gen(model, chunk_filename, val_ds,
                                FLAGS.log_interval, is_gpu)

    model_utils.save_trt_and_upload(model,
                                    FLAGS.val_ds_path,
                                    str(model_dir),
                                    next_model_gen,
                                    run_id=FLAGS.run_id,
                                    batch_size=trt_batch_size())
    model_gen += 1

  logging.info(f'Model gen: {model_gen}. Shutting down.')


if __name__ == '__main__':
  app.run(main)
