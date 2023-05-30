'''
Routines for supervised learning.

We will train our model on samples generated from professional games.
'''

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds

import sys
import transforms

from absl import app, flags, logging
from constants import *
from loss_coeffs import LossCoeffs
from lr_schedule import CyclicLRDecaySchedule
from training_manager import TrainingManager
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
flags.DEFINE_integer(
    'learning_rate_interval', 200000,
    'Interval at which to anneal learning rate (in mini-batches)')
flags.DEFINE_integer(
    'learning_rate_cutoff', 800000,
    'Point after which to stop annealing learning rate (in mini-batches)')
flags.DEFINE_integer(
    'log_interval', 100,
    'Interval at which to log training information (in mini-batches)')
flags.DEFINE_integer('model_save_interval', 5000,
                     'Interval at which to save a new model/model checkpoint')
flags.DEFINE_string('dataset', '', 'Which dataset to use.')


def main(_):
  if FLAGS.dataset == '':
    logging.warning('Please provide --dataset from ~/tensorflow_datasets')
    return

  if FLAGS.model_save_path == '':
    logging.warning('Please provide --model_save_path.')
    return

  train_ds, val_ds = tfds.load(FLAGS.dataset,
                               split=['train[:95%]', 'train[95%:]'],
                               shuffle_files=True)

  # setup training dataset
  batch_size = FLAGS.batch_size
  train_ds = train_ds.map(transforms.expand_sl,
                          num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = train_ds.batch(batch_size)
  train_ds = train_ds.shuffle(1000)
  train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

  # setup test dataset
  val_ds = val_ds.map(transforms.expand_sl, num_parallel_calls=tf.data.AUTOTUNE)
  val_ds = val_ds.batch(batch_size)
  val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

  lr, momentum, epochs = FLAGS.learning_rate, FLAGS.momentum, FLAGS.epochs
  model = P3achyGoModel.create(config=ModelConfig.small(),
                               board_len=BOARD_LEN,
                               num_input_planes=NUM_INPUT_PLANES,
                               num_input_features=NUM_INPUT_FEATURES,
                               name='p3achygo_sl')
  training_manager = TrainingManager(model,
                                     train_ds,
                                     val_ds,
                                     save_interval=FLAGS.model_save_interval,
                                     save_path=FLAGS.model_save_path)
  lr_schedule = CyclicLRDecaySchedule(lr, lr * 10, len(train_ds) * epochs)
  print(lr_schedule.info())
  print(model.summary(batch_size=batch_size))

  is_gpu = False
  if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logging.info('Compute Policy dtype: %s' %
                 tf.keras.mixed_precision.global_policy().compute_dtype)
    logging.info('Variable Policy dtype: %s' %
                 tf.keras.mixed_precision.global_policy().variable_dtype)
    is_gpu = True

  logging.info(f'Starting Training...')
  training_manager.train(epochs,
                         momentum,
                         lr_schedule=lr_schedule,
                         log_interval=FLAGS.log_interval,
                         coeffs=LossCoeffs.SLCoeffs(),
                         is_gpu=is_gpu)
  model_path = Path(FLAGS.model_save_path, 'p3achygo_sl')
  model.save(str(model_path), signatures={'infer_mixed': model.infer_mixed})


if __name__ == '__main__':
  app.run(main)
