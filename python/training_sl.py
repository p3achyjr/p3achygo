'''
Routines for supervised learning.

We will train our model on samples generated from professional games.
'''

import tensorflow as tf
import tensorflow_datasets as tfds

import glob
import os
import time

from absl import app, flags, logging
from board import GoBoard, GoBoardTrainingUtils
from training_config import TrainingConfig
from model import P3achyGoModel
from model_config import ModelConfig
from google.cloud import storage

import matplotlib.pyplot as plt

LEARNING_RATE_INTERVAL = 200000
LEARNING_RATE_CUTOFF = 800000
LOG_INTERVAL = 5
SAVE_INTERVAL = 5000

LOCAL_SAVED_MODEL_PATH = '/tmp/model_{}'
LOCAL_CHECKPOINT_DIR = '/tmp/model_checkpoint_{}'
LOCAL_CHECKPOINT_PATH = '/tmp/model_checkpoint_{}/checkpoint_{}'
GCP_BUCKET = 'p3achygo_models'

FLAGS = flags.FLAGS

flags.DEFINE_string('gcp_credentials_path', '', 'DO NOT HARDCODE')
flags.DEFINE_string('model_path', '',
                    'Folder under which to save model configs')


def upload_dir_to_gcs(gcs_client, directory_path: str, dest_blob_name: str):
  if not gcs_client:
    logging.warning(
        f'No GCS client passed in. Not uploading {directory_path} to gcs.')

  rel_paths = glob.glob(directory_path + '/**', recursive=True)
  bucket = gcs_client.get_bucket(GCP_BUCKET)
  for local_file in rel_paths:
    remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
    if os.path.isfile(local_file):
      logging.info(f'Uploading file {local_file} to dest {remote_path}')
      blob = bucket.blob(remote_path)
      blob.upload_from_filename(local_file)


@tf.function
def train_step(input, policy, model, optimizer, loss_fn):
  with tf.GradientTape() as g:
    preds = model(input, training=True)
    training_loss = loss_fn(policy, preds)
    regularization_loss = tf.math.add_n(model.losses)

    loss = training_loss + regularization_loss

  gradients = g.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return preds, loss


class LossHistory:

  def __init__(self, smoothing_factor=0.95):
    self.alpha = smoothing_factor
    self.loss = []

  def append(self, value):
    self.loss.append(self.alpha * self.loss[-1] +
                     (1 - self.alpha) * value if len(self.loss) > 0 else value)

  def get(self):
    return self.loss


class PeriodicPlotter:

  def __init__(self, sec, xlabel='', ylabel='', scale=None):

    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale

    self.tic = time.time()

  def plot(self, data):
    plt.ion()
    plt.show()
    if time.time() - self.tic > self.sec:
      plt.cla()

      if self.scale is None:
        plt.plot(data)
      elif self.scale == 'semilogx':
        plt.semilogx(data)
      elif self.scale == 'semilogy':
        plt.semilogy(data)
      elif self.scale == 'loglog':
        plt.loglog(data)
      else:
        raise ValueError("unrecognized parameter scale {}".format(self.scale))

      plt.xlabel(self.xlabel)
      plt.ylabel(self.ylabel)
      plt.draw()
      plt.pause(0.001)

      self.tic = time.time()


class SupervisedTrainingManager:
  '''
  Training routine for supervised learning.

  Initializes dataset and holds method for training loop.
  '''

  def __init__(self, training_config=TrainingConfig()):
    self.training_config = training_config
    self.train_ds, self.test_ds = tfds.load(
        'badukmovies', split=['train[:80%]', 'train[80%:]'], shuffle_files=True)

    # setup training dataset
    self.train_ds = self.train_ds.map(SupervisedTrainingManager.expand,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    self.train_ds = self.train_ds.shuffle(
        self.training_config.kDatasetShuffleSize)
    self.train_ds = self.train_ds.batch(self.training_config.kBatchSize)
    self.train_ds = self.train_ds.prefetch(tf.data.AUTOTUNE)

    # setup test dataset
    self.test_ds = self.test_ds.map(SupervisedTrainingManager.expand,
                                    num_parallel_calls=tf.data.AUTOTUNE)
    self.test_ds = self.test_ds.batch(self.training_config.kBatchSize)
    self.test_ds = self.test_ds.prefetch(tf.data.AUTOTUNE)

  # @tf.function
  def train(self):
    model = P3achyGoModel.create(config=ModelConfig.small(), name='p3achy_test')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=self.training_config.kInitLearningRate)

    print(model.summary())

    batch_num = 0
    for _ in range(self.training_config.kEpochs):
      for (input, policy) in self.train_ds:
        preds, current_loss = train_step(input, policy, model, optimizer,
                                         loss_fn)
        if batch_num % LOG_INTERVAL == 0:
          top_preds = tf.math.top_k(preds[0], k=5).indices
          top_vals = tf.math.top_k(preds[0], k=5).values
          board = tf.transpose(input, (0, 3, 1, 2))  # NHWC -> NCHW
          board = tf.cast(board[0][0] + (2 * board[0][1]), dtype=tf.int32)

          print(f'---------- Example {batch_num} -----------')
          print(f'Learning Rate: {optimizer.learning_rate}')
          print(f'Loss: {current_loss}')
          print(f'Top 5 Moves: {top_preds}')
          print(f'Top 5 Move Values: {top_vals}')
          print(f'Policy: {policy[0]}')
          print(f'Board:')
          print(GoBoard.to_string(board.numpy()))

        if 0 < batch_num < LEARNING_RATE_CUTOFF and batch_num % LEARNING_RATE_INTERVAL == 0:
          tf.keras.backend.set_value(optimizer.learning_rate,
                                     optimizer.learning_rate / 10)

        if batch_num % SAVE_INTERVAL == 0:
          local_path = LOCAL_SAVED_MODEL_PATH.format(batch_num)
          remote_path = self.training_config.kGcsCheckpointPath.format(
              batch_num)
          model.save(local_path)
          upload_dir_to_gcs(self.training_config.kGcsClient, local_path,
                            remote_path)

          local_path_ckpt = LOCAL_CHECKPOINT_PATH.format(batch_num, batch_num)
          local_path_ckpt_dir = LOCAL_CHECKPOINT_DIR.format(batch_num)
          remote_path_ckpt = remote_path + '_checkpoint'
          model.save_weights(local_path_ckpt)
          upload_dir_to_gcs(self.training_config.kGcsClient,
                            local_path_ckpt_dir, remote_path_ckpt)

        batch_num += 1

  @staticmethod
  def expand(ex):
    board, last_moves, policy = ex['board'], ex['last_moves'], ex['policy']

    assert (last_moves.shape == (5, 2))

    black_stones = GoBoardTrainingUtils.get_black(board)
    white_stones = GoBoardTrainingUtils.get_white(board)
    fifth_move_before = GoBoardTrainingUtils.as_one_hot(
        tf.cast(last_moves[0], dtype=tf.int32))
    fourth_move_before = GoBoardTrainingUtils.as_one_hot(
        tf.cast(last_moves[1], dtype=tf.int32))
    third_move_before = GoBoardTrainingUtils.as_one_hot(
        tf.cast(last_moves[2], dtype=tf.int32))
    second_move_before = GoBoardTrainingUtils.as_one_hot(
        tf.cast(last_moves[3], dtype=tf.int32))
    first_move_before = GoBoardTrainingUtils.as_one_hot(
        tf.cast(last_moves[4], dtype=tf.int32))

    policy = GoBoardTrainingUtils.as_index(tf.cast(policy, dtype=tf.int32))

    input = tf.convert_to_tensor([
        black_stones, white_stones, fifth_move_before, fourth_move_before,
        third_move_before, second_move_before, first_move_before
    ],
                                 dtype=tf.float32)

    input = tf.transpose(input, perm=(1, 2, 0))  # CHW -> HWC

    return input, policy


def main(argv):
  if FLAGS.gcp_credentials_path == '':
    logging.warning('Please provide a path to GCP credentials.')
    return

  if FLAGS.model_path == '':
    logging.warning('Please provide a path under which to store your model')
    return

  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = FLAGS.gcp_credentials_path
  checkpoint_path = os.path.join(FLAGS.model_path, "batch_{}")

  if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

  logging.info('Compute Policy dtype: %s' %
               tf.keras.mixed_precision.global_policy().compute_dtype)
  logging.info('Variable Policy dtype: %s' %
               tf.keras.mixed_precision.global_policy().variable_dtype)

  training_manager = SupervisedTrainingManager(
      training_config=TrainingConfig(init_learning_rate=1e-2,
                                     epochs=1,
                                     gcs_checkpoint_path=checkpoint_path,
                                     gcs_client=storage.Client()))
  training_manager.train()


if __name__ == '__main__':
  app.run(main)
