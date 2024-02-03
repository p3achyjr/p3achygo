from __future__ import annotations

import tensorflow as tf
import transforms
import train
import rl_loop.model_utils as model_utils

from absl import logging
from constants import *
from lr_schedule import ConstantLRSchedule, CyclicLRSchedule
from model import P3achyGoModel
from rl_loop.config import RunConfig
from weight_snapshot import WeightSnapshotManager

EPOCHS_PER_GEN = 1
MOMENTUM = .9
SWA_MOMENTUM = .75


def train_one_gen(model: P3achyGoModel,
                  model_gen: int,
                  chunk_path: str,
                  val_ds: tf.data.TFRecordDataset,
                  config: RunConfig,
                  log_interval=100,
                  is_gpu=True,
                  batch_num=0,
                  chunk_size=None):
  '''
  Trains through dataset held at `chunk_path`.
  '''

  def find_num_batches(ds: tf.data.TFRecordDataset) -> int:
    n = 0
    for _ in ds.batch(config.batch_size):
      n += 1

    return n

  batch_size = config.batch_size
  lr_scale = 0.1 + 0.9 * min(1.0, model_gen / config.lr_growth_window)
  if not chunk_size:
    lr_schedule = ConstantLRSchedule(config.lr * lr_scale)
  else:
    num_batches = chunk_size // batch_size
    lr_schedule = CyclicLRSchedule(config.min_lr * lr_scale,
                                   config.max_lr * lr_scale, num_batches)

  logging.info(f'Batch Size: {batch_size}')
  logging.info(f'Learning Rate Schedule: {lr_schedule.info()}')
  logging.info(f'Running initial validation...')
  train.val(model, mode=train.Mode.RL, val_ds=val_ds, val_batch_num=-1)

  ds = tf.data.TFRecordDataset(chunk_path, compression_type='ZLIB')
  num_batches = find_num_batches(ds)

  ds = ds.map(transforms.expand, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  ss_interval = num_batches / 3
  ss_manager = WeightSnapshotManager([
      int(ss_interval),
      int(ss_interval * 2),
  ])
  prev_weights = model.get_weights()

  old_batch_num = batch_num
  batch_num = train.train(model,
                          ds,
                          EPOCHS_PER_GEN,
                          MOMENTUM,
                          lr_schedule=lr_schedule,
                          log_interval=log_interval,
                          mode=train.Mode.RL,
                          save_interval=None,
                          save_path=None,
                          is_gpu=is_gpu,
                          batch_num=batch_num,
                          ss_manager=ss_manager)

  print(f'SWA Momentum: {SWA_MOMENTUM}, ' +
        f'Num Batches in Chunk: {batch_num - old_batch_num}')
  new_weights = model_utils.swa_avg_weights(
      [prev_weights] + ss_manager.weights + [model.get_weights()],
      swa_momentum=SWA_MOMENTUM)
  model.set_weights(new_weights)

  logging.info(f'Running validation for new model...')
  train.val(model,
            mode=train.Mode.RL,
            val_ds=val_ds,
            val_batch_num=model_gen + 1)

  return batch_num
