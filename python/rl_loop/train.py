from __future__ import annotations

import tensorflow as tf
import transforms
import train
import rl_loop.model_utils as model_utils

from absl import logging
from constants import *
from lr_schedule import ConstantLRSchedule
from model import P3achyGoModel
from rl_loop.config import RunConfig

EPOCHS_PER_GEN = 1
MOMENTUM = .9


def train_one_gen(model: P3achyGoModel,
                  model_gen: int,
                  chunk_path: str,
                  val_ds: tf.data.TFRecordDataset,
                  config: RunConfig,
                  log_interval=100,
                  is_gpu=True,
                  batch_num=0):
  '''
  Trains through dataset held at `chunk_path`.
  '''
  batch_size = config.batch_size
  lr_schedule = ConstantLRSchedule(config.lr)

  logging.info(f'Batch Size: {batch_size}')
  logging.info(f'Learning Rate Schedule: {lr_schedule.info()}')
  logging.info(f'Running initial validation...')
  train.val(model, mode=train.Mode.RL, val_ds=val_ds, val_batch_num=-1)

  ds = tf.data.TFRecordDataset(chunk_path, compression_type='ZLIB')
  ds = ds.map(transforms.expand, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)

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
                          batch_num=batch_num)

  num_batches_in_chunk = batch_num - old_batch_num
  new_weights = model_utils.avg_weights(prev_weights, model.get_weights(),
                                        num_batches_in_chunk)
  model.set_weights(new_weights)

  logging.info(f'Running validation for new model...')
  train.val(model,
            mode=train.Mode.RL,
            val_ds=val_ds,
            val_batch_num=model_gen + 1)

  return batch_num
