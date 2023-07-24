'''
Routines for supervised learning.

We will train our model on samples generated from professional games.
'''

from __future__ import annotations

import tensorflow as tf

import sys
import transforms
import train
import trt_convert

from absl import app, flags, logging
from constants import *
from lr_schedule import CyclicLRDecaySchedule
from model import P3achyGoModel
from model_config import ModelConfig
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error

FLAGS = flags.FLAGS

# Flags for GCS
flags.DEFINE_boolean('upload_to_gcs', False, 'Whether to upload models to GCS.')

# Flags for local storage
flags.DEFINE_string('model_save_path', '', 'Folder under which to save models.')

# Flags for training configuration
flags.DEFINE_integer('batch_size', 32, 'Mini-batch size')
flags.DEFINE_integer('epochs', 1, 'Number of Epochs')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial Learning Rate')
flags.DEFINE_float('momentum', .9, 'SGD Momentum')
flags.DEFINE_integer('shuf_buf_size', 100000, 'Shuffle Buffer Size')
flags.DEFINE_integer(
    'log_interval', 100,
    'Interval at which to log training information (in mini-batches)')
flags.DEFINE_integer('model_save_interval', 5000,
                     'Interval at which to save a new model/model checkpoint')
flags.DEFINE_string('dataset_dir', '', 'Directory to datasets.')
flags.DEFINE_string('tensorboard_logdir', '/tmp/logs',
                    'Tensorboard log directory.')


def main(_):
  if FLAGS.dataset_dir == '':
    logging.warning('Please provide --dataset_dir where dataset lives.')
    return

  if FLAGS.model_save_path == '':
    logging.warning('Please provide --model_save_path.')
    return

  batch_size = FLAGS.batch_size
  train_shards = [
      str(path) for path in Path(FLAGS.dataset_dir).glob("shard*.tfrecord.zz")
  ]
  val_shard = str(Path(FLAGS.dataset_dir, "val.tfrecord.zz"))
  with open(Path(FLAGS.dataset_dir, 'LENGTH.txt')) as f:
    ds_len = int(f.read()) // batch_size

  tensorboard_log_dir = FLAGS.tensorboard_logdir

  # setup train ds.
  train_ds = tf.data.Dataset.from_tensor_slices(train_shards)
  train_ds = train_ds.interleave(lambda x: tf.data.TFRecordDataset(
      x, compression_type='ZLIB').map(transforms.expand),
                                 cycle_length=64,
                                 block_length=16,
                                 num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = train_ds.shuffle(FLAGS.shuf_buf_size)
  train_ds = train_ds.batch(batch_size)
  train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

  # setup validation dataset
  val_ds = tf.data.TFRecordDataset(val_shard, compression_type='ZLIB')
  val_ds = val_ds.map(transforms.expand)
  val_ds = val_ds.batch(batch_size)

  lr, momentum, epochs = FLAGS.learning_rate, FLAGS.momentum, FLAGS.epochs
  model = P3achyGoModel.create(config=ModelConfig.small(),
                               board_len=BOARD_LEN,
                               num_input_planes=NUM_INPUT_PLANES,
                               num_input_features=NUM_INPUT_FEATURES,
                               name='p3achygo_sl')
  lr_schedule = CyclicLRDecaySchedule(lr, lr * 10, ds_len * epochs)
  print(lr_schedule.info())
  model.summary()

  is_gpu = False
  if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logging.info('Compute Policy dtype: %s' %
                 tf.keras.mixed_precision.global_policy().compute_dtype)
    logging.info('Variable Policy dtype: %s' %
                 tf.keras.mixed_precision.global_policy().variable_dtype)
    is_gpu = True

  logging.info(f'Running initial validation...')
  train.val(model, mode=train.Mode.SL, val_ds=val_ds, val_batch_num=0)

  logging.info(f'Starting Training...')
  train.train(model,
              train_ds,
              epochs,
              momentum,
              log_interval=FLAGS.log_interval,
              mode=train.Mode.SL,
              save_interval=FLAGS.model_save_interval,
              save_path=FLAGS.model_save_path,
              tensorboard_log_dir=tensorboard_log_dir,
              lr_schedule=lr_schedule,
              is_gpu=is_gpu)

  logging.info(f'Running final validation...')
  train.val(model, mode=train.Mode.SL, val_ds=val_ds, val_batch_num=1)

  model_path = str(Path(FLAGS.model_save_path, 'p3achygo_sl'))
  model.save(model_path)


if __name__ == '__main__':
  app.run(main)
