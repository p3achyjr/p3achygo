'''
Routines for supervised learning.

We will train our model on samples generated from professional games.
'''

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds

import glob
import os
import sys
import time

from absl import app, flags, logging
from board import GoBoard, GoBoardTrainingUtils
from constants import *
from training_config import TrainingConfig
from model import P3achyGoModel
from model_config import ModelConfig
from google.cloud import storage

import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

GCP_BUCKET = 'p3achygo_models'

FLAGS = flags.FLAGS

# Flags for GCS
flags.DEFINE_boolean('upload_to_gcs', False,
                     'Whether to upload model checkpoints to GCS.')
flags.DEFINE_string('gcp_credentials_path', '',
                    'Path to GCS credentials. DO NOT HARDCODE.')

# Flags for local storage
flags.DEFINE_string('local_path', '/tmp',
                    'Folder under which to save models and checkpoints')
flags.DEFINE_string('gcs_path', '',
                    'Remote folder under which to save models and checkpoints')

# Flags for training configuration
flags.DEFINE_integer('batch_size', 32, 'Mini-batch size')
flags.DEFINE_integer('epochs', 10, 'Number of Epochs')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial Learning Rate')
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
def train_step(input, komi, score, score_one_hot, policy, model, optimizer):
  with tf.GradientTape() as g:
    pi_logits, game_outcome, game_ownership, score_logits, gamma = model(
        input, tf.expand_dims(komi, axis=1), training=True)
    training_loss, policy_loss, outcome_loss, score_pdf_loss = model.loss(
        pi_logits, game_outcome, score_logits, gamma, policy, score,
        score_one_hot)

    regularization_loss = tf.math.add_n(model.losses)

    loss = training_loss + regularization_loss

  gradients = g.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return (pi_logits, game_outcome, score_logits, loss, policy_loss,
          outcome_loss, score_pdf_loss)


@tf.function()
def train_step_gpu(input, komi, score, score_one_hot, policy, model, optimizer):
  with tf.GradientTape() as g:
    pi_logits, game_outcome, game_ownership, score_logits, gamma = model(
        input, tf.expand_dims(komi, axis=1), training=True)
    training_loss, policy_loss, outcome_loss, score_pdf_loss = model.loss(
        pi_logits, game_outcome, score_logits, gamma, policy, score,
        score_one_hot)

    regularization_loss = tf.cast(tf.math.add_n(model.losses), dtype=tf.float16)

    loss = training_loss + regularization_loss
    scaled_loss = optimizer.get_scaled_loss(loss)

  scaled_gradients = g.gradient(scaled_loss, model.trainable_variables)
  gradients = optimizer.get_unscaled_gradients(scaled_gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return (pi_logits, game_outcome, score_logits, loss, policy_loss,
          outcome_loss, score_pdf_loss)


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


class CyclicLearningRate:
  '''
  Class implementing cyclic learning rate.

  https://arxiv.org/pdf/1803.09820.pdf
  '''

  def __init__(self, learning_rate, max_learning_rate, total_iterations):
    self.iteration = 0
    self.learning_rate_min = learning_rate
    self.learning_rate = learning_rate
    self.learning_rate_max = max_learning_rate
    self.total_iterations = total_iterations

    # decay learning rate for end of training cycle
    self.learning_rate_final = learning_rate * .25

    self.cycle_len = int(self.total_iterations * .45)
    self.lr_delta = (max_learning_rate - learning_rate) / self.cycle_len
    self.lr_decay_delta = (learning_rate - self.learning_rate_final) / (
        self.total_iterations * .1)

    print("Cyclic Learning Rate, Learning Rate Min: ", learning_rate,
          ", Learning Rate Max: ", self.learning_rate_max,
          ", Total Iterations: ", self.total_iterations, "Cycle Len, ",
          self.cycle_len, ", LR_Delta: ", self.lr_delta, ", LR_Delta_Decay: ",
          self.lr_decay_delta)

  def update_lr(self):
    if self.iteration < self.cycle_len:
      # first phase, increase
      self.learning_rate += self.lr_delta
    elif self.iteration < self.cycle_len * 2:
      # second phase, decrease
      self.learning_rate -= self.lr_delta
    else:
      # final decay phase. linear decay to `learning_rate_final`
      self.learning_rate -= self.lr_decay_delta

    self.iteration += 1


class SupervisedTrainingManager:
  '''
  Training routine for supervised learning.

  Initializes dataset and holds method for training loop.
  '''

  class MinLossTracker:

    def __init__(self):
      self.min_train_loss = 10000.
      self.min_train_policy_loss = 10000.
      self.min_train_outcome_loss = 10000.
      self.min_train_score_pdf_loss = 10000.

      self.min_test_loss = 10000.
      self.min_test_policy_loss = 10000.
      self.min_test_outcome_loss = 10000.
      self.min_test_score_pdf_loss = 10000.

    def update_train_losses(self, loss, policy_loss, outcome_loss,
                            score_pdf_loss):
      if loss < self.min_train_loss:
        self.min_train_loss = loss
      if policy_loss < self.min_train_policy_loss:
        self.min_train_policy_loss = policy_loss
      if outcome_loss < self.min_train_outcome_loss:
        self.min_train_outcome_loss = outcome_loss
      if score_pdf_loss < self.min_train_score_pdf_loss:
        self.min_train_score_pdf_loss = score_pdf_loss

    def update_test_losses(self, loss, policy_loss, outcome_loss,
                           score_pdf_loss):
      if loss < self.min_test_loss:
        self.min_test_loss = loss
      if policy_loss < self.min_test_policy_loss:
        self.min_test_policy_loss = policy_loss
      if outcome_loss < self.min_test_outcome_loss:
        self.min_test_outcome_loss = outcome_loss
      if score_pdf_loss < self.min_test_score_pdf_loss:
        self.min_test_score_pdf_loss = score_pdf_loss

  def __init__(self, training_config=TrainingConfig()):
    self.training_config = training_config
    self.train_ds, self.test_ds = tfds.load(
        self.training_config.kDatasetName,
        split=['train[:95%]', 'train[95%:]'],
        shuffle_files=True)

    logging.info(f"Dataset: {self.training_config.kDatasetName}, \
          Length: {len(self.train_ds) + len(self.test_ds)}")

    # setup training dataset
    self.train_ds = self.train_ds.map(SupervisedTrainingManager.expand,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    self.train_ds = self.train_ds.batch(self.training_config.kBatchSize)
    self.train_ds = self.train_ds.shuffle(
        self.training_config.kDatasetShuffleSize)
    self.train_ds = self.train_ds.prefetch(tf.data.AUTOTUNE)

    # setup test dataset
    self.test_ds = self.test_ds.map(SupervisedTrainingManager.expand,
                                    num_parallel_calls=tf.data.AUTOTUNE)
    self.test_ds = self.test_ds.batch(self.training_config.kBatchSize)
    self.test_ds = self.test_ds.prefetch(tf.data.AUTOTUNE)

  def train(self, train_fn, is_gpu=False):
    model = P3achyGoModel.create(config=ModelConfig.small(),
                                 board_len=BOARD_LEN,
                                 num_input_planes=NUM_INPUT_PLANES,
                                 num_input_features=NUM_INPUT_FEATURES,
                                 name='p3achy_test')
    cyclic_learning_rate = CyclicLearningRate(
        self.training_config.kInitLearningRate,
        self.training_config.kInitLearningRate * 10,
        len(self.train_ds) * self.training_config.kEpochs)
    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=cyclic_learning_rate.learning_rate,
        momentum=self.training_config.kLearningRateMomentum)
    if is_gpu:
      optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    print(model.summary(batch_size=self.training_config.kBatchSize))

    batch_num = 0
    loss_tracker = self.MinLossTracker()
    for _ in range(self.training_config.kEpochs):
      # train
      for (input, komi, score, score_one_hot, policy) in self.train_ds:
        (pi_logits, game_outcome, score_logits, current_loss, policy_loss,
         outcome_loss, score_pdf_loss) = train_fn(input, komi, score,
                                                  score_one_hot, policy, model,
                                                  optimizer)
        loss_tracker.update_train_losses(current_loss, policy_loss,
                                         outcome_loss, score_pdf_loss)

        if batch_num % self.training_config.kLogInterval == 0:
          top_policy_indices = tf.math.top_k(pi_logits[0], k=5).indices
          top_policy_values = tf.math.top_k(pi_logits[0], k=5).values
          board = tf.transpose(input, (0, 3, 1, 2))  # NHWC -> NCHW
          board = tf.cast(board[0][0] + (2 * board[0][1]), dtype=tf.int32)
          top_score_indices = tf.math.top_k(score_logits[0],
                                            k=5).indices - SCORE_RANGE_MIDPOINT
          top_score_values = tf.math.top_k(score_logits[0], k=5).values

          print(f'---------- Batch {batch_num} -----------')
          print(f'Learning Rate: {optimizer.learning_rate.numpy()}')
          print(f'Loss: {current_loss}')
          print(f'Min Loss: {loss_tracker.min_train_loss}')
          print(f'Min Policy Loss: {loss_tracker.min_train_policy_loss}')
          print(f'Min Outcome Loss: {loss_tracker.min_train_outcome_loss}')
          print(f'Min Score PDF Loss: {loss_tracker.min_train_score_pdf_loss}')
          print(f'Predicted Outcome: {tf.nn.softmax(game_outcome[0])}')
          print(f'Predicted Scores: {top_score_indices}')
          print(f'Predicted Score Values: {top_score_values}')
          print(f'Actual Score: {score[0]}')
          print(f'Predicted Top 5 Moves: {top_policy_indices}')
          print(f'Predicted Top 5 Move Values: {top_policy_values}')
          print(f'Actual Policy: {policy[0]}')
          print(f'Board:')
          print(GoBoard.to_string(board.numpy()))

        # update learning rate
        cyclic_learning_rate.update_lr()
        tf.keras.backend.set_value(optimizer.learning_rate,
                                   cyclic_learning_rate.learning_rate)

        if batch_num % self.training_config.kModelSaveInterval == 0:
          self.save_model(model, batch_num)

        batch_num += 1

      # validation
      correct_moves = 0
      correct_outcomes = 0
      total_moves = 0
      total_outcomes = 0
      test_batch_num = 0
      for (input, komi, score, score_one_hot, policy) in self.test_ds:
        pi_logits, game_outcome, game_ownership, score_logits, gamma = model(
            input, tf.expand_dims(komi, axis=1), training=False)
        current_loss, policy_loss, outcome_loss, score_pdf_loss = model.loss(
            pi_logits, game_outcome, score_logits, gamma, policy, score,
            score_one_hot)
        loss_tracker.update_test_losses(current_loss, policy_loss, outcome_loss,
                                        score_pdf_loss)

        predicted_move = tf.math.argmax(pi_logits, axis=1, output_type=tf.int32)
        predicted_outcome = tf.math.argmax(game_outcome,
                                           axis=1,
                                           output_type=tf.int32)

        correct_move = tf.cast(tf.equal(policy, predicted_move), dtype=tf.int32)
        correct_outcome = tf.cast(tf.equal(
            tf.where(score >= 0, tf.ones_like(predicted_outcome,
                                              dtype=tf.int32),
                     tf.zeros_like(predicted_outcome, dtype=tf.int32)),
            predicted_outcome),
                                  dtype=tf.int32)

        correct_moves += tf.reduce_sum(correct_move).numpy()
        correct_outcomes += tf.reduce_sum(correct_outcome).numpy()
        total_moves += tf.size(predicted_move).numpy()
        total_outcomes += tf.size(predicted_outcome).numpy()

        if test_batch_num % self.training_config.kLogInterval == 0:
          print(f"---------- Batch {test_batch_num} ----------")
          print(f'Loss: {current_loss}')
          print(f'Min Loss: {loss_tracker.min_test_loss}')
          print(f'Min Test Policy Loss: {loss_tracker.min_test_policy_loss}')
          print(f'Min Test Outcome Loss: {loss_tracker.min_test_outcome_loss}')
          print(
              f'Min Test Score PDF Loss: {loss_tracker.min_test_score_pdf_loss}'
          )
          print("Correct Moves: ", correct_moves, ", Total Moves: ",
                total_moves)
          print("Correct Outcomes: ", correct_outcomes, ", Total Outcomes: ",
                total_outcomes)
          print("Prediction Moves Percentage: ",
                float(correct_moves) / total_moves)
          print("Prediction Outcome Percentage: ",
                float(correct_outcomes) / total_outcomes)

        test_batch_num += 1

    # save final model
    print("Saving Final Model...")
    self.save_model(model, batch_num)

    # log final stats
    print(f'---------- Final Stats ----------')
    print(f'Min Train Loss: {loss_tracker.min_train_loss}')
    print(f'Min Test Loss: {loss_tracker.min_test_loss}')
    print(f'Min Train Policy Loss: {loss_tracker.min_train_policy_loss}')
    print(f'Min Test Policy Loss: {loss_tracker.min_test_policy_loss}')
    print(f'Min Train Outcome Loss: {loss_tracker.min_train_outcome_loss}')
    print(f'Min Test Outcome Loss: {loss_tracker.min_test_outcome_loss}')
    print(f'Min Train Score PDF Loss: {loss_tracker.min_train_score_pdf_loss}')
    print(f'Min Test Score PDF Loss: {loss_tracker.min_test_score_pdf_loss}')

  def save_model(self, model: P3achyGoModel, batch_num: int):
    local_path = self.training_config.kLocalModelPath.format(batch_num)
    model.save(local_path, signatures={'infer_mixed': model.infer_mixed})
    if self.training_config.kUploadingToGcs:
      remote_path = self.training_config.kGcsModelPath.format(batch_num)
      upload_dir_to_gcs(self.training_config.kGcsClient, local_path,
                        remote_path)

    local_path_ckpt = self.training_config.kLocalCheckpointPath.format(
        batch_num)
    local_path_ckpt_dir = self.training_config.kLocalCheckpointDir.format(
        batch_num)

    model.save_weights(local_path_ckpt)
    if self.training_config.kUploadingToGcs:
      remote_path_ckpt = self.training_config.kGcsCheckpointPath.format(
          batch_num)
      upload_dir_to_gcs(self.training_config.kGcsClient, local_path_ckpt_dir,
                        remote_path_ckpt)

  @staticmethod
  def expand(ex):
    board, komi, color, score, last_moves, policy = (ex['board'], ex['komi'],
                                                     ex['color'], ex['result'],
                                                     ex['last_moves'],
                                                     ex['policy'])
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

    score_index = tf.cast([[score + SCORE_RANGE_MIDPOINT]], dtype=tf.int32)
    score_one_hot = tf.cast(tf.scatter_nd(score_index, [1.0],
                                          shape=(SCORE_RANGE,)),
                            dtype=tf.float32)
    policy = GoBoardTrainingUtils.as_index(tf.cast(policy, dtype=tf.int32))

    input = tf.convert_to_tensor([
        black_stones, white_stones, fifth_move_before, fourth_move_before,
        third_move_before, second_move_before, first_move_before
    ],
                                 dtype=tf.float32)

    input = tf.transpose(input, perm=(1, 2, 0))  # CHW -> HWC

    return input, komi, score, score_one_hot, policy


def main(argv):
  if FLAGS.dataset == '':
    logging.warning('Please provide a dataset from ~/tensorflow_datasets')
    return

  if FLAGS.local_path == '':
    logging.warning('Please provide a path under which to store your model')
    return

  gcs_client = None
  if FLAGS.upload_to_gcs:
    if FLAGS.gcp_credentials_path == '':
      logging.warning(
          'Requesting gcs upload, but no path to GCS credentials provided. \
           Please provide a path to GCP credentials via --gcp_credentials_path')
      return

    if FLAGS.gcs_path == '':
      logging.warning('Requesting gcs upload, but no remote path provided. \
           Please provide a remote path via --gcs_path')

      return

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = FLAGS.gcp_credentials_path
    gcs_client = storage.Client()

  local_model_path = os.path.join(FLAGS.local_path, "model_{}")

  # need two of these b/c TF does not create a new folder for each checkpoint.
  local_checkpoint_dir = os.path.join(FLAGS.local_path, 'model_checkpoint_{}')
  local_checkpoint_path = os.path.join(FLAGS.local_path, 'model_checkpoint_{}',
                                       'checkpoint')
  gcs_model_path = os.path.join(FLAGS.gcs_path,
                                'model_{}') if FLAGS.upload_to_gcs else None
  gcs_checkpoint_path = os.path.join(
      FLAGS.gcs_path, 'model_checkpoint_{}') if FLAGS.upload_to_gcs else None

  training_manager = SupervisedTrainingManager(training_config=TrainingConfig(
      dataset_name=FLAGS.dataset,
      batch_size=FLAGS.batch_size,
      epochs=FLAGS.epochs,
      init_learning_rate=FLAGS.learning_rate,
      learning_rate_interval=FLAGS.learning_rate_interval,
      learning_rate_cutoff=FLAGS.learning_rate_cutoff,
      log_interval=FLAGS.log_interval,
      local_model_path=local_model_path,
      local_checkpoint_dir=local_checkpoint_dir,
      local_checkpoint_path=local_checkpoint_path,
      uploading_to_gcs=FLAGS.upload_to_gcs,
      gcs_model_path=gcs_model_path,
      gcs_checkpoint_path=gcs_checkpoint_path,
      gcs_client=gcs_client))

  if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logging.info('Compute Policy dtype: %s' %
                 tf.keras.mixed_precision.global_policy().compute_dtype)
    logging.info('Variable Policy dtype: %s' %
                 tf.keras.mixed_precision.global_policy().variable_dtype)
    training_manager.train(train_step_gpu, is_gpu=True)
  else:
    training_manager.train(train_step, is_gpu=False)


if __name__ == '__main__':
  app.run(main)
