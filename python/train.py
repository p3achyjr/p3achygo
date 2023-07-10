from __future__ import annotations

import functools
import numpy as np
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
def train_step(w_pi, w_pi_aux, w_val, w_outcome, w_score, w_own, w_q30, w_q100,
               w_q200, w_gamma, input, input_global_state, score, score_one_hot,
               policy, policy_aux, own, q30, q100, q200, model, optimizer):
  with tf.GradientTape() as g:
    (pi_logits, _, outcome_logits, _, own_pred, score_logits, _, gamma,
     pi_logits_aux, q30_pred, q100_pred, q200_pred) = model(input,
                                                            input_global_state,
                                                            training=True)
    (loss, policy_loss, policy_aux_loss, outcome_loss, q30_loss,
     q100_loss, q200_loss, score_pdf_loss, own_loss) = model.loss(
         pi_logits, pi_logits_aux, outcome_logits, score_logits, own_pred,
         q30_pred, q100_pred, q200_pred, gamma, policy, policy_aux, score,
         score_one_hot, own, q30, q100, q200, w_pi, w_pi_aux, w_val, w_outcome,
         w_score, w_own, w_q30, w_q100, w_q200, w_gamma)

    reg_loss = tf.math.add_n(model.losses)

    loss = loss + reg_loss

  gradients = g.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return (pi_logits, pi_logits_aux, outcome_logits, own_pred, q30_pred,
          q100_pred, q200_pred, score_logits, loss, policy_loss,
          policy_aux_loss, outcome_loss, q30_loss, q100_loss, q200_loss,
          score_pdf_loss, own_loss)


@tf.function
def train_step_gpu(w_pi, w_pi_aux, w_val, w_outcome, w_score, w_own, w_q30,
                   w_q100, w_q200, w_gamma, input, input_global_state, score,
                   score_one_hot, policy, policy_aux, own, q30, q100, q200,
                   model, optimizer):
  with tf.GradientTape() as g:
    (pi_logits, _, outcome_logits, _, own_pred, score_logits, _, gamma,
     pi_logits_aux, q30_pred, q100_pred, q200_pred) = model(input,
                                                            input_global_state,
                                                            training=True)
    (loss, policy_loss, policy_aux_loss, outcome_loss, q30_loss,
     q100_loss, q200_loss, score_pdf_loss, own_loss) = model.loss(
         pi_logits, pi_logits_aux, outcome_logits, score_logits, own_pred,
         q30_pred, q100_pred, q200_pred, gamma, policy, policy_aux, score,
         score_one_hot, own, q30, q100, q200, w_pi, w_pi_aux, w_val, w_outcome,
         w_score, w_own, w_q30, w_q100, w_q200, w_gamma)

    reg_loss = tf.math.add_n(model.losses)

    loss = loss + reg_loss
    scaled_loss = optimizer.get_scaled_loss(loss)

  scaled_gradients = g.gradient(scaled_loss, model.trainable_variables)
  gradients = optimizer.get_unscaled_gradients(scaled_gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return (pi_logits, pi_logits_aux, outcome_logits, own_pred, q30_pred,
          q100_pred, q200_pred, score_logits, loss, policy_loss,
          policy_aux_loss, outcome_loss, q30_loss, q100_loss, q200_loss,
          score_pdf_loss, own_loss)


@tf.function
def val_step(w_pi, w_pi_aux, w_val, w_outcome, w_score, w_own, w_q30, w_q100,
             w_q200, w_gamma, input, input_global_state, score, score_one_hot,
             policy, policy_aux, own, q30, q100, q200, model, _):
  (pi_logits, _, outcome_logits, _, own_pred, score_logits, _, gamma,
   pi_logits_aux, q30_pred, q100_pred, q200_pred) = model(input,
                                                          input_global_state,
                                                          training=False)
  (loss, policy_loss, policy_aux_loss, outcome_loss, q30_loss,
   q100_loss, q200_loss, score_pdf_loss, own_loss) = model.loss(
       pi_logits, pi_logits_aux, outcome_logits, score_logits, own_pred,
       q30_pred, q100_pred, q200_pred, gamma, policy, policy_aux, score,
       score_one_hot, own, q30, q100, q200, w_pi, w_pi_aux, w_val, w_outcome,
       w_score, w_own, w_q30, w_q100, w_q200, w_gamma)

  return (pi_logits, pi_logits_aux, outcome_logits, own_pred, q30_pred,
          q100_pred, q200_pred, score_logits, loss, policy_loss,
          policy_aux_loss, outcome_loss, q30_loss, q100_loss, q200_loss,
          score_pdf_loss, own_loss)


class LossTracker:
  MAX_LOSS = float('inf')

  def __init__(self):
    self.losses = []
    self.min_losses = defaultdict(lambda: self.MAX_LOSS)

  def update_losses(self, loss, policy_loss, policy_aux_loss, outcome_loss,
                    score_pdf_loss, own_loss, q30_loss, q100_loss, q200_loss):
    self.losses.append({
        'loss': loss,
        'policy': policy_loss,
        'policy_aux': policy_aux_loss,
        'outcome': outcome_loss,
        'score_pdf': score_pdf_loss,
        'own': own_loss,
        'q30': q30_loss,
        'q100': q100_loss,
        'q200': q200_loss,
    })

    self.min_losses['loss'] = min(self.min_losses['loss'], loss)
    self.min_losses['policy'] = min(self.min_losses['policy'], policy_loss)
    self.min_losses['policy_aux'] = min(self.min_losses['policy_aux'],
                                        policy_aux_loss)
    self.min_losses['outcome'] = min(self.min_losses['outcome'], outcome_loss)
    self.min_losses['score_pdf'] = min(self.min_losses['score_pdf'],
                                       score_pdf_loss)
    self.min_losses['own'] = min(self.min_losses['own'], own_loss)
    self.min_losses['q30'] = min(self.min_losses['q30'], q30_loss)
    self.min_losses['q100'] = min(self.min_losses['q100'], q100_loss)
    self.min_losses['q200'] = min(self.min_losses['q200'], q200_loss)


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
  coeffs = LossCoeffs.SLCoeffs() if mode == Mode.SL else LossCoeffs.RLCoeffs()

  # yapf: disable
  train_fn = functools.partial(train_step_gpu if is_gpu else train_step,
                               coeffs.w_pi,
                               coeffs.w_pi_aux,
                               coeffs.w_val,
                               coeffs.w_outcome,
                               coeffs.w_score,
                               coeffs.w_own,
                               coeffs.w_q30,
                               coeffs.w_q100,
                               coeffs.w_q200,
                               coeffs.w_gamma)
  # yapf: enable

  optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate,
                                                   momentum=momentum)
  if is_gpu:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

  batch_num = 0
  losses_train = LossTracker()
  for _ in range(epochs):
    # train
    for (input, input_global_state, color, komi, score, score_one_hot, policy,
         policy_aux, own, q30, q100, q200) in train_ds:
      (pi_logits, pi_logits_aux, outcome_logits, own_pred, q30_pred, q100_pred,
       q200_pred, score_logits, loss, policy_loss, policy_aux_loss,
       outcome_loss, q30_loss, q100_loss, q200_loss, score_pdf_loss,
       own_loss) = train_fn(input, input_global_state, score, score_one_hot,
                            policy, policy_aux, own, q30, q100, q200, model,
                            optimizer)
      losses_train.update_losses(loss, policy_loss, policy_aux_loss,
                                 outcome_loss, score_pdf_loss, own_loss,
                                 q30_loss, q100_loss, q200_loss)

      if batch_num % log_interval == 0:
        log_train(batch_num, input, pi_logits, pi_logits_aux, score_logits,
                  outcome_logits, own_pred, q30_pred, q100_pred, q200_pred,
                  policy, policy_aux, score, q30, q100, q200,
                  optimizer.learning_rate.numpy(), losses_train, own)

      if save_path and save_interval and batch_num % save_interval == 0:
        save_model(model, batch_num, save_path)

      batch_num += 1

  # log final stats
  print(f'---------- Final Stats ----------')
  print(f'Min Train Loss: {losses_train.min_losses["loss"]}')
  print(f'Min Train Policy Loss: {losses_train.min_losses["policy"]}')
  print(f'Min Train Outcome Loss: {losses_train.min_losses["outcome"]}')
  print(f'Min Train Ownership Loss: {losses_train.min_losses["own"]}')
  print(f'Min Train Score PDF Loss: {losses_train.min_losses["score_pdf"]}')


def log_train(batch_num: int, model_input: tf.Tensor, pi_logits: tf.Tensor,
              pi_logits_aux: tf.Tensor, score_logits: tf.Tensor,
              outcome_logits: tf.Tensor, own_pred: tf.Tensor,
              q30_pred: tf.Tensor, q100_pred: tf.Tensor, q200_pred: tf.Tensor,
              pi: tf.Tensor, pi_aux: tf.Tensor, score: tf.Tensor,
              q30: tf.Tensor, q100: tf.Tensor, q200: tf.Tensor, lr: tf.Tensor,
              losses: LossTracker, own: tf.Tensor):

  def move(x):
    return 'ABCDEFGHIJKLMNOPQRS'[x % 19], x // 19

  def char_at(own_pred, i, j):
    x = own_pred[i, j]
    bounds = [-1.0, -0.5, 0, 0.5, 1.0]
    chars = ['●', '◆', '⋅', '◇', '○']

    deltas = [abs(x - bounds)]
    return chars[np.argmin(deltas)]

  def own_pred_to_string(own_pred: tf.Tensor):
    '''`own_pred` is a bsize * bsize grid of reals [-1, 1]'''
    s = []
    for i in range(len(own_pred)):
      s.append((
          '{:2d}'.format(i) + ' ' +
          ' '.join([char_at(own_pred, i, j) for j in range(len(own_pred[0]))])))

    s.append('   ' + ' '.join(list('ABCDEFGHIJKLMNOPQRS')))
    return '\n'.join(s)

  top_policy_indices = tf.math.top_k(pi_logits[0], k=5).indices
  top_policy_values = tf.math.top_k(pi_logits[0], k=5).values
  top_policy_aux_indices = tf.math.top_k(pi_logits_aux[0], k=5).indices
  top_policy_aux_values = tf.math.top_k(pi_logits_aux[0], k=5).values
  board = tf.transpose(model_input, (0, 3, 1, 2))  # NHWC -> NCHW
  board = tf.cast(board[0][0] + (2 * board[0][1]), dtype=tf.int32)
  own = tf.cast(own[0], dtype=tf.int32)
  own_black = tf.where(own == BLACK, own, tf.zeros_like(own)) / BLACK
  own_white = tf.where(own == WHITE, own, tf.zeros_like(own)) / WHITE
  own = own_black * BLACK + own_white * WHITE
  top_score_indices = tf.math.top_k(score_logits[0],
                                    k=5).indices - SCORE_RANGE_MIDPOINT
  top_score_values = tf.math.top_k(score_logits[0], k=5).values
  actual_policy = pi[0] if len(pi.shape) == 1 else tf.math.argmax(pi[0])
  actual_policy_aux = pi_aux[0] if len(pi_aux.shape) == 1 else tf.math.argmax(
      pi_aux[0])

  print(f'---------- Batch {batch_num} -----------')
  print(f'Learning Rate: {lr}')
  print(f'Loss: {losses.losses[-1]["loss"]}')
  print(f'Min Loss: {losses.min_losses["loss"]}')

  print(f'\n===== POLICY LOSSES =====')
  print(f'Policy Loss: {losses.losses[-1]["policy"]}')
  print(f'Min Policy Loss: {losses.min_losses["policy"]}')
  print(f'Policy Aux Loss: {losses.losses[-1]["policy_aux"]}')
  print(f'Min Policy Aux Loss: {losses.min_losses["policy_aux"]}')

  print(f'\n===== Q/Z LOSSES =====')
  print(f'Outcome Loss: {losses.losses[-1]["outcome"]}')
  print(f'Min Outcome Loss: {losses.min_losses["outcome"]}')
  print(f'q30 Loss: {losses.losses[-1]["q30"]}')
  print(f'Min q30 Loss: {losses.min_losses["q30"]}')
  print(f'q100 Loss: {losses.losses[-1]["q100"]}')
  print(f'Min q100 Loss: {losses.min_losses["q100"]}')
  print(f'q200 Loss: {losses.losses[-1]["q200"]}')
  print(f'Min q200 Loss: {losses.min_losses["q200"]}')

  print(f'\n===== SCORE LOSSES =====')
  print(f'Score PDF Loss: {losses.losses[-1]["score_pdf"]}')
  print(f'Min Score PDF Loss: {losses.min_losses["score_pdf"]}')

  print(f'\n===== OWNERSHIP LOSSES =====')
  print(f'Own Loss: {losses.losses[-1]["own"]}')
  print(f'Min Own Loss: {losses.min_losses["own"]}')

  print(f'\n===== POLICY INFO =====')
  print(f'Predicted Top 5 Moves: {top_policy_indices}')
  print(f'Predicted Top 5 Move Logits: {top_policy_values}')
  print(f'Actual Policy: {actual_policy}')
  print(f'Predicted Top 5 Aux Moves: {top_policy_aux_indices}')
  print(f'Predicted Top 5 Aux Move Logits: {top_policy_aux_values}')
  print(f'Actual Aux Policy: {actual_policy_aux}')

  print(f'\n===== Q/Z INFO =====')
  print(f'Predicted Outcome: {tf.nn.softmax(outcome_logits[0])}, Actual Outcome: {1.0 if score[0] >= 0 else 0.0}')
  print(f'q30 Pred: {q30_pred[0]}, Actual: {q30[0]}')
  print(f'q100 Pred: {q100_pred[0]}, Actual: {q100[0]}')
  print(f'q200 Pred: {q200_pred[0]}, Actual: {q200[0]}')

  print(f'\n===== SCORE INFO =====')
  print(f'Predicted Scores: {top_score_indices}')
  print(f'Predicted Score Values: {top_score_values}')
  print(f'Actual Score: {score[0]}')

  print(f'\n===== GRIDS =====')
  print(f'Own Pred:')
  print(own_pred_to_string(own_pred[0].numpy()))
  print(f'Own:')
  print(GoBoard.to_string(own.numpy()))
  print(f'Board:')
  print(GoBoard.to_string(board.numpy()))


def save_model(model: P3achyGoModel, batch_num: int, save_path: str):
  local_path = Path(save_path, f'model_b{batch_num}')
  model.save(str(local_path))


def val(model: P3achyGoModel, mode: Mode, val_ds: tf.data.Dataset):
  coeffs = LossCoeffs.SLCoeffs() if mode == Mode.SL else LossCoeffs.RLCoeffs()
  # yapf: disable
  val_fn = functools.partial(val_step,
                             coeffs.w_pi,
                             coeffs.w_pi_aux,
                             coeffs.w_val,
                             coeffs.w_outcome,
                             coeffs.w_score,
                             coeffs.w_own,
                             coeffs.w_q30,
                             coeffs.w_q100,
                             coeffs.w_q200,
                             coeffs.w_gamma)
  # yapf: enable

  losses_val, val_metrics, val_batch_num = (LossTracker(), ValMetrics(), 0)
  for (input, komi, score, score_one_hot, policy, own) in val_ds:
    (pi_logits, pi_logits_aux, outcome_logits, own_pred, q30_pred, q100_pred,
     q200_pred, score_logits, loss, policy_loss, policy_aux_loss, outcome_loss,
     q30_loss, q100_loss, q200_loss, score_pdf_loss,
     own_loss) = val_fn(input, komi, score, score_one_hot, policy, own, model)
    losses_val.update_losses(loss, policy_loss, policy_aux_loss, outcome_loss,
                             score_pdf_loss, own_loss, q30_loss, q100_loss,
                             q200_loss)

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
  print(f'Min Val Policy Aux Loss: {losses.min_losses["policy_aux"]}')
  print(f'Min Val Outcome Loss: {losses.min_losses["outcome"]}')
  print(f'Min Val Score PDF Loss: {losses.min_losses["score_pdf"]}')
  print(f'Min Val Own Loss: {losses.min_losses["own"]}')
  print(f'Min Val q30 Loss: {losses.min_losses["q30"]}')
  print(f'Min Val q100 Loss: {losses.min_losses["q100"]}')
  print(f'Min Val q200 Loss: {losses.min_losses["q200"]}')
  print("Correct Moves: ", metrics.correct_moves, ", Total Moves: ",
        metrics.num_moves)
  print("Correct Outcomes: ", metrics.correct_outcomes, ", Total Outcomes: ",
        metrics.num_outcomes)
  print("Prediction Moves Percentage: ",
        float(metrics.correct_moves) / metrics.num_moves)
  print("Prediction Outcome Percentage: ",
        float(metrics.correct_outcomes) / metrics.num_outcomes)
