from __future__ import annotations

import gcs_utils as gcs
import tensorflow as tf
import transforms
import train
import rl_loop.model_utils as model_utils

from absl import logging
from constants import *
from model import P3achyGoModel

BATCH_SIZE = 256
LR = 3e-3
EPOCHS_PER_GEN = 1
MOMENTUM = .9


def train_one_gen(model: P3achyGoModel,
                  model_gen: int,
                  chunk_path: str,
                  val_ds: tf.data.TFRecordDataset,
                  log_interval=100,
                  batch_size=BATCH_SIZE,
                  lr=LR,
                  is_gpu=True,
                  batch_num=0):
  '''
  Trains through dataset held at `chunk_path`.
  '''
  logging.info(f'Batch Size: {batch_size}, Learning Rate: {lr}')
  logging.info(f'Running initial validation...')
  train.val(model, mode=train.Mode.RL, val_ds=val_ds, val_batch_num=-1)

  ds = tf.data.TFRecordDataset(chunk_path, compression_type='ZLIB')
  ds = ds.map(transforms.expand, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  prev_weights = model.get_weights()

  batch_num = train.train(model,
                          ds,
                          EPOCHS_PER_GEN,
                          MOMENTUM,
                          lr=lr,
                          log_interval=log_interval,
                          mode=train.Mode.RL,
                          save_interval=None,
                          save_path=None,
                          is_gpu=is_gpu,
                          batch_num=batch_num)

  new_weights = model_utils.avg_weights(prev_weights, model.get_weights())
  model.set_weights(new_weights)

  logging.info(f'Running validation for new model...')
  train.val(model,
            mode=train.Mode.RL,
            val_ds=val_ds,
            val_batch_num=model_gen + 1)

  return batch_num
