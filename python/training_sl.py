'''
Routines for supervised learning.

We will train our model on samples generated from professional games.
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import time

from board import GoBoard, GoBoardTrainingUtils
from training_config import TrainingConfig
from model import P3achyGoModel
from model_config import ModelConfig

import matplotlib.pyplot as plt

LEARNING_RATE_INTERVAL = 200000
LEARNING_RATE_CUTOFF = 800000
LOG_INTERVAL = 2
SAVE_INTERVAL = 100000


@tf.function
def train_step(input, policy, model, optimizer, loss_fn):
  with tf.GradientTape() as g:
    preds = model(input)
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
    model = P3achyGoModel(config=ModelConfig.small(), name='p3achy_test')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=self.training_config.kInitLearningRate)

    print(model.summary())

    # plotter = PeriodicPlotter(sec=2,
    #                           xlabel='Iterations',
    #                           ylabel='Loss',
    #                           scale='semilogy')
    # loss_history = LossHistory()
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
          model.save(f'/Users/axlui/p3achyGo/models/small_step_{batch_num}')
        # loss_history.append(current_loss.numpy().mean())
        # plotter.plot(loss_history.get())

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


if __name__ == '__main__':
  training_manager = SupervisedTrainingManager(
      training_config=TrainingConfig(init_learning_rate=1e-2, epochs=1))
  training_manager.train()
