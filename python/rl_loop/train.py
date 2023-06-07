'''
RL Training Task.

Repeatedly polls for a new chunk from GCS, and trains a model on it.
'''

from __future__ import annotations

import gcs_utils as gcs
import sys, time
import tensorflow as tf
import transforms

from absl import app, flags, logging
from constants import *
from loss_coeffs import LossCoeffs
from pathlib import Path
from training_manager import TrainingManager
from model import P3achyGoModel
from model_config import ModelConfig

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
    model = P3achyGoModel.create(config=ModelConfig.small(),
                                 board_len=BOARD_LEN,
                                 num_input_planes=NUM_INPUT_PLANES,
                                 num_input_features=NUM_INPUT_FEATURES,
                                 name=f'p3achygo_{FLAGS.run_id}')
  else:
    model_path = gcs.download_model(FLAGS.run_id, str(model_dir), model_gen)
    model = tf.keras.models.load_model(
        model_path, custom_objects=P3achyGoModel.custom_objects())

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
    ds = tf.data.TFRecordDataset(chunk_filename, compression_type='ZLIB')
    ds = ds.map(transforms.expand_rl, num_parallel_calls=8)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    training_manager = TrainingManager(model,
                                       ds,
                                       tf.data.Dataset.from_tensor_slices([]),
                                       save_interval=1000,
                                       save_path=ckpt_dir)

    logging.info(f'Training model {next_model_gen}...')
    logging.info(f'{model.summary(batch_size=BATCH_SIZE)}')
    training_manager.train(EPOCHS_PER_GEN,
                           MOMENTUM,
                           lr=LR,
                           log_interval=FLAGS.log_interval,
                           coeffs=LossCoeffs.RLCoeffs(),
                           is_gpu=is_gpu)
    model_filename = Path(model_dir, f'model_{next_model_gen}')
    model.save(str(model_filename),
               signatures={'infer_mixed': model.infer_mixed})
    gcs.upload_model(FLAGS.run_id, str(model_dir), next_model_gen)
    model_gen += 1


if __name__ == '__main__':
  app.run(main)