from __future__ import annotations

import functools
import numpy as np
import tensorflow as tf

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
    scaled_loss = tf.clip_by_value(loss, -100.0, 100.0)
    scaled_loss = optimizer.get_scaled_loss(scaled_loss)

  gradients = g.gradient(scaled_loss, model.trainable_variables)
  gradients = optimizer.get_unscaled_gradients(gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return (pi_logits, pi_logits_aux, outcome_logits, own_pred, q30_pred,
          q100_pred, q200_pred, score_logits, loss, policy_loss,
          policy_aux_loss, outcome_loss, q30_loss, q100_loss, q200_loss,
          score_pdf_loss, own_loss, gradients)


@tf.function
def val_step(w_pi, w_pi_aux, w_val, w_outcome, w_score, w_own, w_q30, w_q100,
             w_q200, w_gamma, input, input_global_state, score, score_one_hot,
             policy, policy_aux, own, q30, q100, q200, model):
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
          tensorboard_log_dir='/tmp/logs',
          lr_schedule: Optional[
              tf.keras.optimizers.schedules.LearningRateSchedule] = None,
          is_gpu=False,
          batch_num=0):
  """
  Training through single dataset.
  """
  summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)
  # tf.summary.trace_on(graph=True, profiler=True)
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

  optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=lr_schedule,
                                                   momentum=momentum)
  if is_gpu:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

  losses_train = LossTracker()
  did_log_graph = False
  for _ in range(epochs):
    # train
    for (input, input_global_state, color, komi, score, score_one_hot, policy,
         policy_aux, own, q30, q100, q200) in train_ds:
      if save_path and save_interval and batch_num % save_interval == 0:
        save_model(model, batch_num, save_path)

      (pi_logits, pi_logits_aux, outcome_logits, own_pred, q30_pred, q100_pred,
       q200_pred, score_logits, loss, policy_loss, policy_aux_loss,
       outcome_loss, q30_loss, q100_loss, q200_loss, score_pdf_loss, own_loss,
       gradients) = train_fn(input, input_global_state, score, score_one_hot,
                             policy, policy_aux, own, q30, q100, q200, model,
                             optimizer)
      losses_train.update_losses(loss, policy_loss, policy_aux_loss,
                                 outcome_loss, score_pdf_loss, own_loss,
                                 q30_loss, q100_loss, q200_loss)

      if batch_num % log_interval == 0:
        with summary_writer.as_default():
          # if not did_log_graph:
          #   tf.summary.trace_export(name="model_trace",
          #                           step=0,
          #                           profiler_outdir=tensorboard_log_dir)
          #   did_log_graph = True
          # Log weights
          for var in model.trainable_variables:
            tf.summary.histogram(var.name, var, step=batch_num)

          # Log gradients
          for grad, var in zip(gradients, model.trainable_variables):
            tf.summary.histogram(var.name + '/gradient', grad, step=batch_num)

          log_train(batch_num, input, pi_logits, pi_logits_aux, score_logits,
                    outcome_logits, own_pred, q30_pred, q100_pred, q200_pred,
                    policy, policy_aux, score, q30, q100, q200,
                    optimizer.learning_rate.numpy(), losses_train, own, mode)

      batch_num += 1

  # log final stats
  print(f'---------- Final Stats ----------')
  print(f'Min Train Loss: {losses_train.min_losses["loss"]}')
  print(f'Min Train Policy Loss: {losses_train.min_losses["policy"]}')
  print(f'Min Train Outcome Loss: {losses_train.min_losses["outcome"]}')
  print(f'Min Train Ownership Loss: {losses_train.min_losses["own"]}')
  print(f'Min Train Score PDF Loss: {losses_train.min_losses["score_pdf"]}')

  return batch_num


def log_train(batch_num: int, model_input: tf.Tensor, pi_logits: tf.Tensor,
              pi_logits_aux: tf.Tensor, score_logits: tf.Tensor,
              outcome_logits: tf.Tensor, own_pred: tf.Tensor,
              q30_pred: tf.Tensor, q100_pred: tf.Tensor, q200_pred: tf.Tensor,
              pi: tf.Tensor, pi_aux: tf.Tensor, score: tf.Tensor,
              q30: tf.Tensor, q100: tf.Tensor, q200: tf.Tensor, lr: tf.Tensor,
              losses: LossTracker, own: tf.Tensor, mode: Mode):

  def move(x):
    return 'ABCDEFGHIJKLMNOPQRS'[x % 19], (x // 19).numpy()

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

  def log(metric_name: str, value):
    print(f'{metric_name}: {value}')
    tf.summary.scalar(metric_name, value, step=batch_num)

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
  log('Learning Rate', lr)
  log('Loss', losses.losses[-1]["loss"])
  log('Min Loss', losses.min_losses["loss"])

  print(f'===== POLICY LOSSES =====')
  log('Policy Loss', losses.losses[-1]["policy"])
  log('Min Policy Loss', losses.min_losses["policy"])
  log('Policy Aux Loss', losses.losses[-1]["policy_aux"])
  log('Min Policy Aux Loss', losses.min_losses["policy_aux"])

  print(f'===== Q/Z LOSSES =====')
  log('Outcome Loss', losses.losses[-1]["outcome"])
  log('Min Outcome Loss', losses.min_losses["outcome"])
  log('q30 Loss', losses.losses[-1]["q30"])
  log('Min q30 Loss', losses.min_losses["q30"])
  log('q100 Loss', losses.losses[-1]["q100"])
  log('Min q100 Loss', losses.min_losses["q100"])
  log('q200 Loss', losses.losses[-1]["q200"])
  log('Min q200 Loss', losses.min_losses["q200"])

  print(f'===== SCORE LOSSES =====')
  log('Score PDF Loss', losses.losses[-1]["score_pdf"])
  log('Min Score PDF Loss', losses.min_losses["score_pdf"])

  print(f'===== OWNERSHIP LOSSES =====')
  log('Own Loss', losses.losses[-1]["own"])
  log('Min Own Loss', losses.min_losses["own"])

  print(f'===== POLICY INFO =====')
  print(f'Predicted Top 5 Moves:', f'{[move(mv) for mv in top_policy_indices]}',
        f'{top_policy_indices}')
  print(f'Predicted Top 5 Move Logits: {top_policy_values}')
  print(f'Actual Policy: {actual_policy}, {move(actual_policy)}')
  print(f'Predicted Top 5 Aux Moves:',
        f'{[move(mv) for mv in top_policy_aux_indices]}',
        f'{top_policy_aux_indices}')
  print(f'Predicted Top 5 Aux Move Logits: {top_policy_aux_values}')
  print(f'Actual Aux Policy: {actual_policy_aux}, {move(actual_policy_aux)}')

  print(f'===== Q/Z INFO =====')
  print(f'Predicted Outcome: {tf.nn.softmax(outcome_logits[0])},',
        f'Actual Outcome: {1.0 if score[0] >= 0 else 0.0}')
  print(f'q30 Pred: {q30_pred[0]}, Actual: {q30[0]}')
  print(f'q100 Pred: {q100_pred[0]}, Actual: {q100[0]}')
  print(f'q200 Pred: {q200_pred[0]}, Actual: {q200[0]}')

  print(f'===== SCORE INFO =====')
  print(f'Predicted Scores: {top_score_indices}')
  print(f'Predicted Score Values: {top_score_values}')
  print(f'Actual Score: {score[0]}')

  print(f'===== GRIDS =====')
  if mode == Mode.RL:
    print(f'Own Pred:')
    print(own_pred_to_string(own_pred[0].numpy()))
    print(f'Own:')
    print(GoBoard.to_string(own.numpy()))
  print(f'Board:')
  print(GoBoard.to_string(board.numpy()))


def save_model(model: P3achyGoModel, batch_num: int, save_path: str):
  local_path = Path(save_path, f'model_b{batch_num}')
  model.save(str(local_path))


def val(model: P3achyGoModel,
        mode: Mode,
        val_ds: tf.data.Dataset,
        val_batch_num=0,
        tensorboard_log_dir='/tmp/logs'):
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

  losses_val, val_metrics = LossTracker(), ValMetrics()
  for (input, input_global_state, color, komi, score, score_one_hot, policy,
       policy_aux, own, q30, q100, q200) in val_ds:
    (pi_logits, pi_logits_aux, outcome_logits, own_pred, q30_pred, q100_pred,
     q200_pred, score_logits, loss, policy_loss, policy_aux_loss, outcome_loss,
     q30_loss, q100_loss, q200_loss, score_pdf_loss,
     own_loss) = val_fn(input, input_global_state, score, score_one_hot, policy,
                        policy_aux, own, q30, q100, q200, model)
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

  summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)
  with summary_writer.as_default():
    log_val(val_batch_num, losses_val, val_metrics)


def log_val(batch_num: int, losses: LossTracker, metrics: ValMetrics):

  def log(metric_name: str, value):
    print(f'{metric_name}: {value}')
    if batch_num >= 0:
      tf.summary.scalar(metric_name, value, step=batch_num)

  print(f"---------- Val Batch {batch_num} ----------")
  log('Val Loss', losses.losses[-1]["loss"])
  log('Min Val Loss', losses.min_losses["loss"])
  log('Min Val Policy Loss', losses.min_losses["policy"])
  log('Min Val Policy Aux Loss', losses.min_losses["policy_aux"])
  log('Min Val Outcome Loss', losses.min_losses["outcome"])
  log('Min Val Score PDF Loss', losses.min_losses["score_pdf"])
  print("Correct Moves: ", metrics.correct_moves, ", Total Moves: ",
        metrics.num_moves)
  print("Correct Outcomes: ", metrics.correct_outcomes, ", Total Outcomes: ",
        metrics.num_outcomes)
  log("Prediction Moves Percentage",
      float(metrics.correct_moves) / metrics.num_moves)
  log("Prediction Outcome Percentage",
      float(metrics.correct_outcomes) / metrics.num_outcomes)
