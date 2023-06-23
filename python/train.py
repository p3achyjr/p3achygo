from __future__ import annotations

import functools
import tensorflow as tf

from absl import logging
from board import GoBoard
from collections import defaultdict
from constants import *
from model import P3achyGoModel
from pathlib import Path
from typing import Optional
from loss_coeffs import LossCoeffs
from enum import Enum


class Mode(Enum):
  SL = 1
  RL = 2


@tf.function
def train_step(w_pi, w_val, w_outcome, w_score, w_own, w_gamma, use_kl_loss,
               input, komi, score, score_one_hot, policy, own, model,
               optimizer):
  with tf.GradientTape() as g:
    pi_logits, _, game_outcome, _, own_pred, score_logits, _, gamma = model(
        input, tf.cast(komi, tf.float32), training=True)
    (loss, policy_loss, outcome_loss, score_pdf_loss,
     own_loss) = model.loss(pi_logits, game_outcome, score_logits, own_pred,
                            gamma, policy, score, score_one_hot, own, w_pi,
                            w_val, w_outcome, w_score, w_own, w_gamma,
                            use_kl_loss)

    reg_loss = tf.math.add_n(model.losses)

    loss = loss + reg_loss

  gradients = g.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return (pi_logits, game_outcome, score_logits, loss, policy_loss,
          outcome_loss, score_pdf_loss, own_loss)


@tf.function
def train_step_gpu(w_pi, w_val, w_outcome, w_score, w_own, w_gamma, use_kl_loss,
                   input, komi, score, score_one_hot, policy, own, model,
                   optimizer):
  with tf.GradientTape() as g:
    pi_logits, _, game_outcome, _, own_pred, score_logits, _, gamma = model(
        input, tf.cast(komi, tf.float32), training=True)
    (loss, policy_loss, outcome_loss, score_pdf_loss,
     own_loss) = model.loss(pi_logits, game_outcome, score_logits, own_pred,
                            gamma, policy, score, score_one_hot, own, w_pi,
                            w_val, w_outcome, w_score, w_own, w_gamma,
                            use_kl_loss)

    reg_loss = tf.math.add_n(model.losses)

    loss = loss + reg_loss
    scaled_loss = optimizer.get_scaled_loss(loss)

  scaled_gradients = g.gradient(scaled_loss, model.trainable_variables)
  gradients = optimizer.get_unscaled_gradients(scaled_gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return (pi_logits, game_outcome, score_logits, loss, policy_loss,
          outcome_loss, score_pdf_loss, own_loss)


@tf.function
def val_step(w_pi, w_val, w_outcome, w_score, w_own, w_gamma, use_kl_loss,
             input, komi, score, score_one_hot, policy, own, model):
  pi_logits, _, game_outcome, _, own_pred, score_logits, _, gamma = model(
      input, tf.cast(komi, tf.float32), training=False)
  (loss, policy_loss, outcome_loss, score_pdf_loss,
   own_loss) = model.loss(pi_logits, game_outcome, score_logits, own_pred,
                          gamma, policy, score, score_one_hot, own, w_pi, w_val,
                          w_outcome, w_score, w_own, w_gamma, use_kl_loss)

  return (pi_logits, game_outcome, score_logits, loss, policy_loss,
          outcome_loss, score_pdf_loss, own_loss)


class LossTracker:
  MAX_LOSS = float('inf')

  def __init__(self):
    self.losses = []
    self.min_losses = defaultdict(lambda: self.MAX_LOSS)

  def update_losses(self, loss, policy_loss, outcome_loss, score_pdf_loss,
                    own_loss):
    self.losses.append({
        'loss': loss,
        'policy': policy_loss,
        'outcome': outcome_loss,
        'score_pdf': score_pdf_loss,
        'own': own_loss,
    })

    self.min_losses['loss'] = min(self.min_losses['loss'], loss)
    self.min_losses['policy'] = min(self.min_losses['policy'], policy_loss)
    self.min_losses['outcome'] = min(self.min_losses['outcome'], outcome_loss)
    self.min_losses['score_pdf'] = min(self.min_losses['score_pdf'],
                                       score_pdf_loss)
    self.min_losses['own'] = min(self.min_losses['own'], own_loss)


class ValMetrics:

  def __init__(self):
    self.num_moves = 0
    self.num_outcomes = 0
    self.correct_moves = 0
    self.correct_outcomes = 0

  def increment(self, num_moves, num_outcomes, correct_moves, correct_outcomes):
    self.num_moves += num_moves
    self.num_outcomes += num_outcomes
    self.correct_moves += correct_moves
    self.correct_outcomes += correct_outcomes


def train(model: P3achyGoModel,
          train_ds: tf.data.Dataset,
          epochs: int,
          momentum: float,
          log_interval: int,
          mode: Mode,
          save_interval=1000,
          save_path='/tmp',
          lr: Optional[float] = None,
          lr_schedule: Optional[
              tf.keras.optimizers.schedules.LearningRateSchedule] = None,
          is_gpu=False):
  """
  Training through single dataset.
  """
  if not lr and not lr_schedule:
    logging.error('Exactly one of `lr` and `lr_schedule` must be set.')

  if lr and lr_schedule:
    logging.error('Exactly one of `lr` and `lr_schedule` must be set.')

  learning_rate = lr if lr else lr_schedule
  coeffs = (LossCoeffs.SLCoeffs() if mode == Mode.SL else LossCoeffs.RLCoeffs())
  use_kl_loss = mode == Mode.RL

  # yapf: disable
  train_fn = functools.partial(train_step_gpu if is_gpu else train_step,
                               coeffs.w_pi,
                               coeffs.w_val,
                               coeffs.w_outcome,
                               coeffs.w_score,
                               coeffs.w_own,
                               coeffs.w_gamma,
                               use_kl_loss)
  # yapf: enable

  optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate,
                                                   momentum=momentum)
  if is_gpu:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

  batch_num = 0
  losses_train = LossTracker()
  for _ in range(epochs):
    # train
    for (input, komi, score, score_one_hot, policy, own) in train_ds:
      (pi_logits, outcome_logits, score_logits, loss, policy_loss, outcome_loss,
       score_pdf_loss, own_loss) = train_fn(input, komi, score, score_one_hot,
                                            policy, own, model, optimizer)
      losses_train.update_losses(loss, policy_loss, outcome_loss,
                                 score_pdf_loss, own_loss)

      if batch_num % log_interval == 0:
        log_train(batch_num, input, pi_logits,
                  score_logits, outcome_logits, policy, score,
                  optimizer.learning_rate.numpy(), losses_train, own)

      if batch_num % save_interval == 0:
        save_model(model, batch_num, save_path)

      batch_num += 1

  # log final stats
  print(f'---------- Final Stats ----------')
  print(f'Min Train Loss: {losses_train.min_losses["loss"]}')
  print(f'Min Train Policy Loss: {losses_train.min_losses["policy"]}')
  print(f'Min Train Outcome Loss: {losses_train.min_losses["outcome"]}')
  print(f'Min Train Score PDF Loss: {losses_train.min_losses["score_pdf"]}')


def log_train(batch_num: int, model_input: tf.Tensor, pi_logits: tf.Tensor,
              score_logits: tf.Tensor, outcome_logits: tf.Tensor, pi: tf.Tensor,
              score: tf.Tensor, lr: tf.Tensor, losses: LossTracker,
              own: tf.Tensor):
  top_policy_indices = tf.math.top_k(pi_logits[0], k=5).indices
  top_policy_values = tf.math.top_k(pi_logits[0], k=5).values
  board = tf.transpose(model_input, (0, 3, 1, 2))  # NHWC -> NCHW
  board = tf.cast(board[0][0] + (2 * board[0][1]), dtype=tf.int32)
  own = tf.cast(own[0], dtype=tf.int32)
  own_black = tf.where(own == BLACK_RL, own, tf.zeros_like(own)) / BLACK_RL
  own_white = tf.where(own == WHITE_RL, own, tf.zeros_like(own)) / WHITE_RL
  own = own_black * BLACK + own_white * WHITE
  top_score_indices = tf.math.top_k(score_logits[0],
                                    k=5).indices - SCORE_RANGE_MIDPOINT
  top_score_values = tf.math.top_k(score_logits[0], k=5).values
  actual_policy = pi[0] if len(pi.shape) == 1 else tf.math.argmax(pi[0])

  print(f'---------- Batch {batch_num} -----------')
  print(f'Learning Rate: {lr}')
  print(f'Loss: {losses.losses[-1]["loss"]}')
  print(f'Min Loss: {losses.min_losses["loss"]}')
  print(f'Policy Loss: {losses.losses[-1]["policy"]}')
  print(f'Min Policy Loss: {losses.min_losses["policy"]}')
  print(f'Outcome Loss: {losses.losses[-1]["outcome"]}')
  print(f'Min Outcome Loss: {losses.min_losses["outcome"]}')
  print(f'Score PDF Loss: {losses.losses[-1]["score_pdf"]}')
  print(f'Min Score PDF Loss: {losses.min_losses["score_pdf"]}')
  print(f'Own Loss: {losses.losses[-1]["own"]}')
  print(f'Min Own Loss: {losses.min_losses["own"]}')
  print(f'Predicted Outcome: {tf.nn.softmax(outcome_logits[0])}')
  print(f'Predicted Scores: {top_score_indices}')
  print(f'Predicted Score Values: {top_score_values}')
  print(f'Actual Score: {score[0]}')
  print(f'Predicted Top 5 Moves: {top_policy_indices}')
  print(f'Predicted Top 5 Move Logits: {top_policy_values}')
  print(f'Actual Policy: {actual_policy}')
  print(f'Own:')
  print(GoBoard.to_string(own.numpy()))
  print(f'Board:')
  print(GoBoard.to_string(board.numpy()))


def save_model(model: P3achyGoModel, batch_num: int, save_path: str):
  local_path = Path(save_path, f'model_b{batch_num}')
  model.save(str(local_path))


def val(model: P3achyGoModel, mode: Mode, val_ds: tf.data.Dataset):
  coeffs = (LossCoeffs.SLCoeffs() if mode == Mode.SL else LossCoeffs.RLCoeffs())
  use_kl_loss = mode == Mode.RL
  # yapf: disable
  val_fn = functools.partial(val_step,
                             coeffs.w_pi,
                             coeffs.w_val,
                             coeffs.w_outcome,
                             coeffs.w_score,
                             coeffs.w_own,
                             coeffs.w_gamma,
                             use_kl_loss)
  # yapf: enable

  losses_val, val_metrics, val_batch_num = (LossTracker(), ValMetrics(), 0)
  for (input, komi, score, score_one_hot, policy, own) in val_ds:
    (pi_logits, outcome_logits, score_logits, loss, policy_loss, outcome_loss,
     score_pdf_loss, own_loss) = val_fn(input, komi, score, score_one_hot,
                                        policy, own, model)
    losses_val.update_losses(loss, policy_loss, outcome_loss, score_pdf_loss,
                             own_loss)

    true_move = policy if len(policy.shape) == 1 else tf.math.argmax(
        policy, axis=1, output_type=tf.int32)
    predicted_move = tf.math.argmax(pi_logits, axis=1, output_type=tf.int32)
    predicted_outcome = tf.math.argmax(outcome_logits,
                                       axis=1,
                                       output_type=tf.int32)

    correct_move = tf.cast(tf.equal(true_move, predicted_move), dtype=tf.int32)
    correct_outcome = tf.cast(tf.equal(
        tf.where(score >= 0, tf.ones_like(predicted_outcome, dtype=tf.int32),
                 tf.zeros_like(predicted_outcome, dtype=tf.int32)),
        predicted_outcome),
                              dtype=tf.int32)

    val_metrics.increment(
        tf.size(predicted_move).numpy(),
        tf.size(predicted_outcome).numpy(),
        tf.reduce_sum(correct_move).numpy(),
        tf.reduce_sum(correct_outcome).numpy())

    val_batch_num += 1

  log_val(val_batch_num, losses_val, val_metrics)


def log_val(batch_num: int, losses: LossTracker, metrics: ValMetrics):
  print(f"---------- Val Batch {batch_num} ----------")
  print(f'Loss: {losses.losses[-1]["loss"]}')
  print(f'Min Loss: {losses.min_losses["loss"]}')
  print(f'Min Val Policy Loss: {losses.min_losses["policy"]}')
  print(f'Min Val Outcome Loss: {losses.min_losses["outcome"]}')
  print(f'Min Val Score PDF Loss: {losses.min_losses["score_pdf"]}')
  print(f'Min Val Own Loss: {losses.min_losses["own"]}')
  print("Correct Moves: ", metrics.correct_moves, ", Total Moves: ",
        metrics.num_moves)
  print("Correct Outcomes: ", metrics.correct_outcomes, ", Total Outcomes: ",
        metrics.num_outcomes)
  print("Prediction Moves Percentage: ",
        float(metrics.correct_moves) / metrics.num_moves)
  print("Prediction Outcome Percentage: ",
        float(metrics.correct_outcomes) / metrics.num_outcomes)
