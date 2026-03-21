from __future__ import annotations

import functools
import numpy as np
import tensorflow as tf
import keras
from typing import NamedTuple, Optional

from board import GoBoard
from collections import defaultdict
from constants import *
from model import P3achyGoModel, ModelPredictions, GroundTruth, LossWeights
from pathlib import Path
from loss_coeffs import LossCoeffs
from enum import Enum
from weight_snapshot import WeightSnapshotManager


class Mode(Enum):
    SL = 1
    RL = 2


class TrainStepResult(NamedTuple):
    """Result from a training step."""

    predictions: ModelPredictions
    total_loss: tf.Tensor
    policy_loss: tf.Tensor
    policy_aux_loss: tf.Tensor
    outcome_loss: tf.Tensor
    q6_loss: tf.Tensor
    q16_loss: tf.Tensor
    q50_loss: tf.Tensor
    score_pdf_loss: tf.Tensor
    score_cdf_loss: tf.Tensor
    own_loss: tf.Tensor
    # v1 only
    q_err_loss: Optional[tf.Tensor] = None
    q_score_loss: Optional[tf.Tensor] = None
    q_score_err_loss: Optional[tf.Tensor] = None
    pi_soft_loss: Optional[tf.Tensor] = None
    pi_optimistic_loss: Optional[tf.Tensor] = None
    grad_norm: float = 0.0


@tf.function
def train_step(
    input: tf.Tensor,
    input_global_state: tf.Tensor,
    targets: GroundTruth,
    weights: LossWeights,
    model: P3achyGoModel,
    optimizer,
) -> TrainStepResult:
    """
    Training step for v1 models (with one-batch-norm).

    Args:
        input: Board state tensor
        input_global_state: Global state tensor
        targets: GroundTruth with labels
        weights: LossWeights with loss weights
        model: The model instance
        optimizer: The optimizer

    Returns:
        TrainStepResult with predictions and losses
    """
    with tf.GradientTape() as g:
        # Get model outputs (v1: 46 outputs = 23 FVI + 23 BN)
        model_outputs = model(input, input_global_state, training=True)

        # Extract FVI outputs (first 23)
        (
            pi_logits,
            pi,
            outcome_logits,
            outcome_probs,
            ownership,
            score_logits,
            score_probs,
            gamma,
            pi_logits_aux,
            q6_pred,
            q16_pred,
            q50_pred,
            q6_err_pred,
            q16_err_pred,
            q50_err_pred,
            q6_score_pred,
            q16_score_pred,
            q50_score_pred,
            q6_score_err_pred,
            q16_score_err_pred,
            q50_score_err_pred,
            pi_logits_soft,
            pi_logits_optimistic,
        ) = model_outputs

        # Create FVI predictions
        predictions = ModelPredictions(
            pi_logits=pi_logits,
            pi_logits_aux=pi_logits_aux,
            game_outcome=outcome_logits,
            score_logits=score_logits,
            own_pred=ownership,
            q6_pred=q6_pred,
            q16_pred=q16_pred,
            q50_pred=q50_pred,
            gamma=gamma,
            q6_err_pred=q6_err_pred,
            q16_err_pred=q16_err_pred,
            q50_err_pred=q50_err_pred,
            q6_score_pred=q6_score_pred,
            q16_score_pred=q16_score_pred,
            q50_score_pred=q50_score_pred,
            q6_score_err_pred=q6_score_err_pred,
            q16_score_err_pred=q16_score_err_pred,
            q50_score_err_pred=q50_score_err_pred,
            pi_logits_soft=pi_logits_soft,
            pi_logits_optimistic=pi_logits_optimistic,
        )

        # Compute losses for both heads
        loss_outputs = model.compute_losses(predictions, targets, weights)

        # Unpack loss outputs
        (
            loss,
            policy_loss,
            policy_aux_loss,
            outcome_loss,
            q6_loss,
            q16_loss,
            q50_loss,
            score_pdf_loss,
            score_cdf_loss,
            own_loss,
            q_err_loss,
            q_score_loss,
            q_score_err_loss,
            pi_soft_loss,
            pi_optimistic_loss,
        ) = loss_outputs

        reg_loss = tf.math.add_n(model.losses)
        total_loss = loss + reg_loss
        scaled_loss = optimizer.scale_loss(total_loss)

    # Compute gradients (LossScaleOptimizer handles scaling automatically in Keras 3.x)
    gradients = g.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    scale = tf.cast(
        optimizer.dynamic_scale if optimizer.built else optimizer.initial_scale,
        tf.float32,
    )
    unscaled_gradients = [g / scale for g in gradients]

    # Return FVI predictions for logging
    return TrainStepResult(
        predictions=predictions,
        total_loss=total_loss,
        policy_loss=policy_loss,
        policy_aux_loss=policy_aux_loss,
        outcome_loss=outcome_loss,
        q6_loss=q6_loss,
        q16_loss=q16_loss,
        q50_loss=q50_loss,
        score_pdf_loss=score_pdf_loss,
        score_cdf_loss=score_cdf_loss,
        own_loss=own_loss,
        q_err_loss=q_err_loss,
        q_score_loss=q_score_loss,
        q_score_err_loss=q_score_err_loss,
        pi_soft_loss=pi_soft_loss,
        pi_optimistic_loss=pi_optimistic_loss,
        grad_norm=tf.linalg.global_norm(unscaled_gradients),
    )


@tf.function
def val_step(
    input: tf.Tensor,
    input_global_state: tf.Tensor,
    targets: GroundTruth,
    weights: LossWeights,
    model: P3achyGoModel,
) -> TrainStepResult:
    """Validation step for v1 models (with one-batch-norm)."""
    # Get model outputs (v1: 46 outputs = 23 FVI + 23 BN)
    model_outputs = model(input, input_global_state, training=False)

    # Extract FVI outputs (first 23)
    (
        pi_logits,
        pi,
        outcome_logits,
        outcome_probs,
        ownership,
        score_logits,
        score_probs,
        gamma,
        pi_logits_aux,
        q6_pred,
        q16_pred,
        q50_pred,
        q6_err_pred,
        q16_err_pred,
        q50_err_pred,
        q6_score_pred,
        q16_score_pred,
        q50_score_pred,
        q6_score_err_pred,
        q16_score_err_pred,
        q50_score_err_pred,
        pi_logits_soft,
        pi_logits_optimistic,
    ) = model_outputs[:23]

    # Create FVI predictions
    predictions = ModelPredictions(
        pi_logits=pi_logits,
        pi_logits_aux=pi_logits_aux,
        game_outcome=outcome_logits,
        score_logits=score_logits,
        own_pred=ownership,
        q6_pred=q6_pred,
        q16_pred=q16_pred,
        q50_pred=q50_pred,
        gamma=gamma,
        q6_err_pred=q6_err_pred,
        q16_err_pred=q16_err_pred,
        q50_err_pred=q50_err_pred,
        q6_score_pred=q6_score_pred,
        q16_score_pred=q16_score_pred,
        q50_score_pred=q50_score_pred,
        q6_score_err_pred=q6_score_err_pred,
        q16_score_err_pred=q16_score_err_pred,
        q50_score_err_pred=q50_score_err_pred,
        pi_logits_soft=pi_logits_soft,
        pi_logits_optimistic=pi_logits_optimistic,
    )

    # Compute losses for both heads
    loss_outputs = model.compute_losses(predictions, targets, weights)

    # Unpack loss outputs
    (
        loss,
        policy_loss,
        policy_aux_loss,
        outcome_loss,
        q6_loss,
        q16_loss,
        q50_loss,
        score_pdf_loss,
        score_cdf_loss,
        own_loss,
        q_err_loss,
        q_score_loss,
        q_score_err_loss,
        pi_soft_loss,
        pi_optimistic_loss,
    ) = loss_outputs

    reg_loss = tf.math.add_n(model.losses)
    total_loss = loss + reg_loss

    return TrainStepResult(
        predictions=predictions,
        total_loss=total_loss,
        policy_loss=policy_loss,
        policy_aux_loss=policy_aux_loss,
        outcome_loss=outcome_loss,
        q6_loss=q6_loss,
        q16_loss=q16_loss,
        q50_loss=q50_loss,
        score_pdf_loss=score_pdf_loss,
        score_cdf_loss=score_cdf_loss,
        own_loss=own_loss,
        q_err_loss=q_err_loss,
        q_score_loss=q_score_loss,
        q_score_err_loss=q_score_err_loss,
        pi_soft_loss=pi_soft_loss,
        pi_optimistic_loss=pi_optimistic_loss,
    )


class LossTracker:
    MAX_LOSS = float("inf")

    def __init__(self):
        self.n = 0
        self.min_losses = defaultdict(lambda: self.MAX_LOSS)
        self.avg_losses = defaultdict(lambda: 0)

    def update_losses(
        self,
        result: TrainStepResult,
    ):
        loss = result.total_loss.numpy()
        policy_loss = result.policy_loss.numpy()
        policy_aux_loss = result.policy_aux_loss.numpy()
        outcome_loss = result.outcome_loss.numpy()
        score_pdf_loss = result.score_pdf_loss.numpy()
        score_cdf_loss = result.score_cdf_loss.numpy()
        own_loss = result.own_loss.numpy()
        q6_loss = result.q6_loss.numpy()
        q16_loss = result.q16_loss.numpy()
        q50_loss = result.q50_loss.numpy()
        q_err_loss = result.q_err_loss.numpy() if result.q_err_loss is not None else 0.0
        q_score_loss = (
            result.q_score_loss.numpy() if result.q_score_loss is not None else 0.0
        )
        q_score_err_loss = (
            result.q_score_err_loss.numpy()
            if result.q_score_err_loss is not None
            else 0.0
        )
        pi_soft_loss = (
            result.pi_soft_loss.numpy() if result.pi_soft_loss is not None else 0.0
        )
        pi_optimistic_loss = (
            result.pi_optimistic_loss.numpy()
            if result.pi_optimistic_loss is not None
            else 0.0
        )

        self.min_losses["loss"] = min(self.min_losses["loss"], loss)
        self.min_losses["policy"] = min(self.min_losses["policy"], policy_loss)
        self.min_losses["policy_aux"] = min(
            self.min_losses["policy_aux"], policy_aux_loss
        )
        self.min_losses["outcome"] = min(self.min_losses["outcome"], outcome_loss)
        self.min_losses["score_pdf"] = min(self.min_losses["score_pdf"], score_pdf_loss)
        self.min_losses["score_cdf"] = min(self.min_losses["score_cdf"], score_cdf_loss)
        self.min_losses["own"] = min(self.min_losses["own"], own_loss)
        self.min_losses["q6"] = min(self.min_losses["q6"], q6_loss)
        self.min_losses["q16"] = min(self.min_losses["q16"], q16_loss)
        self.min_losses["q50"] = min(self.min_losses["q50"], q50_loss)
        self.min_losses["q_err"] = min(self.min_losses["q_err"], q_err_loss)
        self.min_losses["q_score"] = min(self.min_losses["q_score"], q_score_loss)
        self.min_losses["q_score_err"] = min(
            self.min_losses["q_score_err"], q_score_err_loss
        )
        self.min_losses["pi_soft"] = min(self.min_losses["pi_soft"], pi_soft_loss)
        self.min_losses["pi_optimistic"] = min(
            self.min_losses["pi_optimistic"], pi_optimistic_loss
        )

        r_m = 0.99 if self.n > 0 else 0.0
        r_c = 0.01 if self.n > 0 else 1.0
        self.n += 1
        self.avg_losses["loss"] = self.avg_losses["loss"] * r_m + loss * r_c
        self.avg_losses["policy"] = self.avg_losses["policy"] * r_m + policy_loss * r_c
        self.avg_losses["policy_aux"] = (
            self.avg_losses["policy_aux"] * r_m + policy_aux_loss * r_c
        )
        self.avg_losses["outcome"] = (
            self.avg_losses["outcome"] * r_m + outcome_loss * r_c
        )
        self.avg_losses["score_pdf"] = (
            self.avg_losses["score_pdf"] * r_m + score_pdf_loss * r_c
        )
        self.avg_losses["score_cdf"] = (
            self.avg_losses["score_cdf"] * r_m + score_cdf_loss * r_c
        )
        self.avg_losses["own"] = self.avg_losses["own"] * r_m + own_loss * r_c
        self.avg_losses["q6"] = self.avg_losses["q6"] * r_m + q6_loss * r_c
        self.avg_losses["q16"] = self.avg_losses["q16"] * r_m + q16_loss * r_c
        self.avg_losses["q50"] = self.avg_losses["q50"] * r_m + q50_loss * r_c
        self.avg_losses["q_err"] = self.avg_losses["q_err"] * r_m + q_err_loss * r_c
        self.avg_losses["q_score"] = (
            self.avg_losses["q_score"] * r_m + q_score_loss * r_c
        )
        self.avg_losses["q_score_err"] = (
            self.avg_losses["q_score_err"] * r_m + q_score_err_loss * r_c
        )
        self.avg_losses["pi_soft"] = (
            self.avg_losses["pi_soft"] * r_m + pi_soft_loss * r_c
        )
        self.avg_losses["pi_optimistic"] = (
            self.avg_losses["pi_optimistic"] * r_m + pi_optimistic_loss * r_c
        )


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


def train(
    model: P3achyGoModel,
    train_ds: tf.data.Dataset,
    epochs: int,
    momentum: float,
    log_interval: int,
    mode: Mode,
    coeffs: LossCoeffs,
    optimizer: Optional[keras.optimizers.Optimizer] = None,
    save_interval=1000,
    save_path="/tmp",
    tensorboard_log_dir="/tmp/logs",
    lr_schedule: Optional[keras.optimizers.schedules.LearningRateSchedule] = None,
    is_gpu=True,
    batch_num=0,
    ss_manager: Optional[WeightSnapshotManager] = None,
    val_ds: Optional[tf.data.Dataset] = None,
    num_val_batches: int = 10,
):
    """
    Training through single dataset.
    """
    assert is_gpu
    summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)

    # Create LossWeights NamedTuple from coefficients
    weights = LossWeights(
        w_pi=coeffs.w_pi,
        w_pi_aux=coeffs.w_pi_aux,
        w_val=coeffs.w_val,
        w_outcome=coeffs.w_outcome,
        w_score=coeffs.w_score,
        w_own=coeffs.w_own,
        w_q6=coeffs.w_q6,
        w_q16=coeffs.w_q16,
        w_q50=coeffs.w_q50,
        w_gamma=coeffs.w_gamma,
        w_q_err=coeffs.w_q_err,
        w_q_score=coeffs.w_q_score,
        w_q_score_err=coeffs.w_q_score_err,
        w_pi_soft=coeffs.w_pi_soft,
        w_pi_optimistic=coeffs.w_pi_optimistic,
    )

    if not optimizer:
        optimizer = keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=momentum,
            global_clipnorm=20.0,  # purely for numerical stability
            nesterov=True,
        )
        if is_gpu:
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    losses_train = LossTracker()
    local_batch_num = 0
    for _ in range(epochs):
        # train
        for batch_data in train_ds:
            if save_path and save_interval and batch_num % save_interval == 0:
                save_model(model, optimizer, batch_num, save_path)
                # Run validation on checkpoint save
                if val_ds is not None:
                    val(
                        model,
                        val_ds,
                        batch_num,
                        num_batches=num_val_batches,
                        mode=mode,
                        tensorboard_log_dir=tensorboard_log_dir,
                    )

            (
                input,
                input_global_state,
                color,
                komi,
                score,
                score_one_hot,
                policy,
                policy_aux,
                own,
                q6,
                q16,
                q50,
                q6_score,
                q16_score,
                q50_score,
                game_outcome,
            ) = batch_data

            targets = GroundTruth(
                policy=policy,
                policy_aux=policy_aux,
                score=score,
                score_one_hot=score_one_hot,
                game_outcome=game_outcome,
                own=own,
                q6=q6,
                q16=q16,
                q50=q50,
                q6_score=q6_score,
                q16_score=q16_score,
                q50_score=q50_score,
            )

            result = train_step(
                input,
                input_global_state,
                targets,
                weights,
                model,
                optimizer,
            )

            losses_train.update_losses(result)

            local_batch_num += 1
            batch_num += 1

            # Query snapshot manager after every training step
            if ss_manager and ss_manager.should_take_snapshot(local_batch_num):
                ss_manager.take_snapshot(model)

            if local_batch_num % log_interval == 0:
                log_train(
                    batch_num,
                    losses_train,
                    result.grad_norm,
                    summary_writer,
                    mode,
                )

                # Log board position with predictions every 5th log interval
                if local_batch_num % (log_interval * 5) == 0:
                    log_board_position(
                        batch_num,
                        input,
                        input_global_state,
                        result.predictions,
                        targets,
                        model,
                    )

                    # Log policy entropy to detect flat distributions
                    pi_probs = keras.activations.softmax(
                        result.predictions.pi_logits[0]
                    )
                    policy_entropy = -tf.reduce_sum(
                        pi_probs * tf.math.log(pi_probs + 1e-10)
                    )
                    target_entropy = -tf.reduce_sum(
                        targets.policy[0] * tf.math.log(targets.policy[0] + 1e-10)
                    )
                    print(
                        f"Policy entropy - Predicted: {policy_entropy.numpy():.3f}, "
                        f"Target: {target_entropy.numpy():.3f} "
                        f"(max={tf.math.log(362.0).numpy():.3f})"
                    )

    if save_path:
        save_model(model, optimizer, batch_num, save_path)
        # Run validation on final checkpoint save
        if val_ds is not None:
            val(
                model,
                val_ds,
                batch_num,
                num_batches=num_val_batches,
                mode=mode,
                tensorboard_log_dir=tensorboard_log_dir,
            )
    return batch_num, optimizer


def log_train(
    batch_num: int,
    losses: LossTracker,
    grad_norm: float,
    summary_writer: tf.summary.SummaryWriter,
    mode: Mode,
):
    mode_str = "sl" if mode == Mode.SL else "rl"

    loss_avg = losses.avg_losses["loss"]
    policy_avg = losses.avg_losses["policy"]
    policy_aux_avg = losses.avg_losses["policy_aux"]
    outcome_avg = losses.avg_losses["outcome"]
    score_pdf_avg = losses.avg_losses["score_pdf"]
    score_cdf_avg = losses.avg_losses["score_cdf"]
    own_avg = losses.avg_losses["own"]
    q6_avg = losses.avg_losses["q6"]
    q16_avg = losses.avg_losses["q16"]
    q50_avg = losses.avg_losses["q50"]
    q_err_avg = losses.avg_losses["q_err"]
    q_score_avg = losses.avg_losses["q_score"]
    q_score_err_avg = losses.avg_losses["q_score_err"]
    pi_soft_avg = losses.avg_losses["pi_soft"]
    pi_optimistic_avg = losses.avg_losses["pi_optimistic"]

    loss_min = losses.min_losses["loss"]
    policy_min = losses.min_losses["policy"]
    policy_aux_min = losses.min_losses["policy_aux"]
    outcome_min = losses.min_losses["outcome"]
    score_pdf_min = losses.min_losses["score_pdf"]
    score_cdf_min = losses.min_losses["score_cdf"]
    own_min = losses.min_losses["own"]
    q6_min = losses.min_losses["q6"]
    q16_min = losses.min_losses["q16"]
    q50_min = losses.min_losses["q50"]
    q_err_min = losses.min_losses["q_err"]
    q_score_min = losses.min_losses["q_score"]
    q_score_err_min = losses.min_losses["q_score_err"]
    pi_soft_min = losses.min_losses["pi_soft"]
    pi_optimistic_min = losses.min_losses["pi_optimistic"]

    print(
        f"[batch {batch_num}] {mode_str}: "
        f"loss = {loss_avg:.4f} ({loss_min:.4f}), "
        f"policy = {policy_avg:.4f} ({policy_min:.4f}), "
        f"policy_aux = {policy_aux_avg:.4f} ({policy_aux_min:.4f}), "
        f"outcome = {outcome_avg:.4f} ({outcome_min:.4f}), "
        f"score_pdf = {score_pdf_avg:.4f} ({score_pdf_min:.4f}), "
        f"score_cdf = {score_cdf_avg:.4f} ({score_cdf_min:.4f}), "
        f"own = {own_avg:.4f} ({own_min:.4f}), "
        f"q6 = {q6_avg:.4f} ({q6_min:.4f}), "
        f"q16 = {q16_avg:.4f} ({q16_min:.4f}), "
        f"q50 = {q50_avg:.4f} ({q50_min:.4f}), "
        f"q_err = {q_err_avg:.4f} ({q_err_min:.4f}), "
        f"q_score = {q_score_avg:.4f} ({q_score_min:.4f}), "
        f"q_score_err = {q_score_err_avg:.4f} ({q_score_err_min:.4f}), "
        f"pi_soft = {pi_soft_avg:.4f} ({pi_soft_min:.4f}), "
        f"pi_optimistic = {pi_optimistic_avg:.4f} ({pi_optimistic_min:.4f}), "
        f"grad_norm = {grad_norm:.4f}"
    )

    with summary_writer.as_default():
        tf.summary.scalar(f"{mode_str}/loss", loss_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/policy", policy_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/policy_aux", policy_aux_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/outcome", outcome_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/score_pdf", score_pdf_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/score_cdf", score_cdf_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/own", own_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/q6", q6_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/q16", q16_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/q50", q50_avg, step=batch_num)


def log_board_position(
    batch_num: int,
    input_planes: tf.Tensor,
    input_global_state: tf.Tensor,
    predictions: ModelPredictions,
    targets: GroundTruth,
    model: P3achyGoModel,
):
    """Log a sample board position with model predictions."""
    # Take first example from batch
    planes = input_planes[0].numpy()  # (19, 19, num_planes)
    global_state = input_global_state[0].numpy()

    # Reconstruct board from planes
    # Planes 0-2: current position (our stones, opponent stones, empty)
    # For v1: planes 0 = our color, plane 1 = opponent color
    board = np.zeros((19, 19), dtype=np.int8)
    our_stones = planes[:, :, 0]
    opp_stones = planes[:, :, 1]

    # Determine current player from global state
    # global_state format: [is_black, is_white, last_move_pass_0, ..., komi_normalized]
    to_play = BLACK if global_state[0] > 0.5 else WHITE
    komi = global_state[-1]  # Last element is komi_normalized (komi / 15.0)
    komi_actual = abs(komi) * 15.0  # Denormalize to get actual komi value

    # Set stones on board
    if to_play == BLACK:
        board[our_stones > 0.5] = BLACK
        board[opp_stones > 0.5] = WHITE
    else:
        board[our_stones > 0.5] = WHITE
        board[opp_stones > 0.5] = BLACK

    # Get predictions and ground truth
    policy_pred = tf.nn.softmax(predictions.pi_logits[0]).numpy()
    outcome_pred = tf.nn.softmax(predictions.game_outcome[0]).numpy()
    score_pred = tf.nn.softmax(predictions.score_logits[0]).numpy()

    policy_target = targets.policy[0].numpy()
    score_target = targets.score[0].numpy()

    # Get top 5 policy moves
    top_indices = np.argsort(policy_pred)[-5:][::-1]
    top_indices_target = np.argsort(policy_target)[-5:][::-1]

    # short-term
    q6_pred, q6 = predictions.q6_pred[0].numpy(), targets.q6[0].numpy()
    q16_pred, q16 = predictions.q16_pred[0].numpy(), targets.q16[0].numpy()
    q50_pred, q50 = predictions.q50_pred[0].numpy(), targets.q50[0].numpy()
    q6_err_pred, q6_err = predictions.q6_err_pred[0].numpy(), np.square(q6 - q6_pred)
    q16_err_pred, q16_err = predictions.q16_err_pred[0].numpy(), np.square(
        q16 - q16_pred
    )
    q50_err_pred, q50_err = predictions.q50_err_pred[0].numpy(), np.square(
        q50 - q50_pred
    )

    # short-term score
    q6_score_pred, q6_score = (
        predictions.q6_score_pred[0].numpy() * 10,
        targets.q6_score[0].numpy(),
    )
    q16_score_pred, q16_score = (
        predictions.q16_score_pred[0].numpy() * 10,
        targets.q16_score[0].numpy(),
    )
    q50_score_pred, q50_score = (
        predictions.q50_score_pred[0].numpy() * 10,
        targets.q50_score[0].numpy(),
    )
    q6_score_err_pred, q6_score_err = (
        predictions.q6_score_err_pred[0].numpy() * 100,
        np.square(q6_score - q6_score_pred),
    )
    q16_score_err_pred, q16_score_err = (
        predictions.q16_score_err_pred[0].numpy() * 100,
        np.square(q16_score - q16_score_pred),
    )
    q50_score_err_pred, q50_score_err = (
        predictions.q50_score_err_pred[0].numpy() * 100,
        np.square(q50_score - q50_score_pred),
    )

    # Convert move indices to coordinates
    def move_to_coords(move_idx):
        if move_idx == 361:
            return "PASS"
        row = move_idx // 19
        col = move_idx % 19
        return f"{chr(ord('A') + (col if col < 8 else col + 1))}{19 - row}"

    # Print board and predictions
    print(f"\n{'='*60}")
    print(f"BOARD POSITION - Batch {batch_num}")
    print(f"{'='*60}")
    print(f"To play: {'BLACK (○)' if to_play == BLACK else 'WHITE (●)'}")
    print(f"Komi: {komi_actual:.1f} (normalized: {komi:+.3f})")
    print()
    # Ownership (own_pred is from current player perspective; convert to absolute black=positive)
    own_pred = predictions.own_pred[0].numpy().squeeze()  # (19, 19)
    own_pred_abs = own_pred if to_play == BLACK else -own_pred

    def ownership_char(x):
        bounds = [-1.0, -0.5, 0.0, 0.5, 1.0]
        chars = ["●", "◯", "⋅", "◆", "○"]  # positive=black=○, negative=white=●
        return chars[int(np.argmin([abs(x - b) for b in bounds]))]

    board_lines = GoBoard.to_string(board).split("\n")
    own_lines = []
    for i in range(19):
        own_lines.append(
            f"{19 - i:2d} "
            + " ".join([ownership_char(own_pred_abs[i, j]) for j in range(19)])
        )
    own_lines.append("   " + " ".join(list("ABCDEFGHJKLMNOPQRST")))
    col_gap = "    "
    print(f"  {'Board':<42}{col_gap}Ownership (○=black ●=white)")
    for bl, ol in zip(board_lines, own_lines):
        print(f"  {bl:<42}{col_gap}{ol}")
    print()

    print(f"{'='*60}")
    print("PREDICTIONS vs GROUND TRUTH")
    print(f"{'='*60}")

    # Win probability
    print(f"\nWin Probability:")
    print(f"  Predicted: {outcome_pred[1]:.1%} (win) / {outcome_pred[0]:.1%} (loss)")
    print(
        f"  Actual:    {'WIN' if score_target >= 0 else 'LOSS'} (score: {score_target:+.1f})"
    )

    # Score distribution
    score_mean_pred = np.sum(score_pred * np.arange(-400, 400))
    score_std_pred = np.sqrt(
        np.sum(score_pred * ((np.arange(-400, 400) - score_mean_pred) ** 2))
    )
    print(f"\nScore Prediction:")
    print(f"  Predicted: {score_mean_pred:+.1f} ± {score_std_pred:.1f}")
    print(f"  Actual:    {score_target:+.1f}")

    # Short-term
    print(f"\nShort-Term Value:")
    print(
        f"  Q6 Predicted: {q6_pred:.4f}, Actual: {q6:.4f}, Err Predicted: {q6_err_pred:.4f}, Actual: {q6_err:.4f}"
    )
    print(
        f"  Q16 Predicted: {q16_pred:.4f}, Actual: {q16:.4f}, Err Predicted: {q16_err_pred:.4f}, Actual: {q16_err:.4f}"
    )
    print(
        f"  Q50 Predicted: {q50_pred:.4f}, Actual: {q50:.4f}, Err Predicted: {q50_err_pred:.4f}, Actual: {q50_err:.4f}"
    )
    print(f"\n\nShort-Term Score:")
    print(
        f"  Q6 Score Predicted: {q6_score_pred:.4f}, Actual: {q6_score:.4f}"
        f", Err Predicted: {q6_score_err_pred:.4f}, Actual: {q6_score_err:.4f}"
    )
    print(
        f"  Q16 Score Predicted: {q16_score_pred:.4f}, Actual: {q16_score:.4f}"
        f", Err Predicted: {q16_score_err_pred:.4f}, Actual: {q16_score_err:.4f}"
    )
    print(
        f"  Q50 Score Predicted: {q50_score_pred:.4f}, Actual: {q50_score:.4f}"
        f", Err Predicted: {q50_score_err_pred:.4f}, Actual: {q50_score_err:.4f}"
    )

    # Soft policy target (policy^0.25 normalized, mirrors v1_loss_terms)
    policy_soft_target = np.power(np.maximum(policy_target, 0.0), 0.25)
    _soft_sum = policy_soft_target.sum()
    if _soft_sum > 0:
        policy_soft_target /= _soft_sum
    top_indices_soft_target = np.argsort(policy_soft_target)[-5:][::-1]

    # Soft and optimistic predicted policies
    pi_soft_probs = keras.activations.softmax(predictions.pi_logits_soft[0]).numpy()
    pi_optimistic_probs = keras.activations.softmax(
        predictions.pi_logits_optimistic[0]
    ).numpy()
    top_soft = np.argsort(pi_soft_probs)[-5:][::-1]
    top_optimistic = np.argsort(pi_optimistic_probs)[-5:][::-1]

    # Optimistic weight (mirrors v1_loss_terms computation)
    epsilon = 1e-6
    q6_p = predictions.q6_pred[0].numpy()
    q16_p = predictions.q16_pred[0].numpy()
    q50_p = predictions.q50_pred[0].numpy()
    q6_err_p = predictions.q6_err_pred[0].numpy()
    q16_err_p = predictions.q16_err_pred[0].numpy()
    q50_err_p = predictions.q50_err_pred[0].numpy()
    z6 = (targets.q6[0].numpy() - q6_p) / np.sqrt(q6_err_p + epsilon)
    z16 = (targets.q16[0].numpy() - q16_p) / np.sqrt(q16_err_p + epsilon)
    z50 = (targets.q50[0].numpy() - q50_p) / np.sqrt(q50_err_p + epsilon)
    z_wd = 4.0 / 7.0
    z_val = (z_wd * 3 * z6 + z_wd * 1.5 * z16 + z_wd * 0.75 * z50) / 3.0
    opt_weight = float(np.clip(1 / (1 + np.exp(-(z_val - 1.0) * 3)), 0.0, 1.0))

    # Top 5 moves table
    col_w = 20
    print(f"\nTop 5 Policy Moves:")
    print(
        f"  {'Predicted':<{col_w}}{'Target':<{col_w}}{'Soft Predicted':<{col_w}}{'Soft Target':<{col_w}}"
        f"Optimistic (w={opt_weight:.2f}, z={z_val:.2f})"
    )
    for i in range(5):
        pred_str = (
            f"{move_to_coords(top_indices[i]):>6} {policy_pred[top_indices[i]]:>6.1%}"
        )
        tgt_str = f"{move_to_coords(top_indices_target[i]):>6} {policy_target[top_indices_target[i]]:>6.1%}"
        soft_str = (
            f"{move_to_coords(top_soft[i]):>6} {pi_soft_probs[top_soft[i]]:>6.1%}"
        )
        soft_tgt_str = f"{move_to_coords(top_indices_soft_target[i]):>6} {policy_soft_target[top_indices_soft_target[i]]:>6.1%}"
        opt_str = f"{move_to_coords(top_optimistic[i]):>6} {pi_optimistic_probs[top_optimistic[i]]:>6.1%}"
        print(
            f"  {pred_str:<{col_w}}{tgt_str:<{col_w}}{soft_str:<{col_w}}{soft_tgt_str:<{col_w}}{opt_str}"
        )

    print(f"{'='*60}\n")


def save_model(
    model: P3achyGoModel,
    opt: keras.optimizers.Optimizer,
    batch_num: int,
    save_path: str,
):
    filename = f"model_{batch_num}.keras"
    filepath = Path(save_path) / filename
    model.compile(optimizer=opt)
    model.save(filepath)


def val(
    model: P3achyGoModel,
    val_ds: tf.data.Dataset,
    batch_num: int,
    num_batches=10,
    mode=Mode.SL,
    tensorboard_log_dir="/tmp/logs",
):
    """Validation on dataset."""
    summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)

    if mode == Mode.SL:
        coeffs = LossCoeffs.SLCoeffs()
    else:
        coeffs = LossCoeffs.RLCoeffs()

    # Create LossWeights NamedTuple from coefficients
    weights = LossWeights(
        w_pi=coeffs.w_pi,
        w_pi_aux=coeffs.w_pi_aux,
        w_val=coeffs.w_val,
        w_outcome=coeffs.w_outcome,
        w_score=coeffs.w_score,
        w_own=coeffs.w_own,
        w_q6=coeffs.w_q6,
        w_q16=coeffs.w_q16,
        w_q50=coeffs.w_q50,
        w_gamma=coeffs.w_gamma,
        w_q_err=coeffs.w_q_err,
        w_q_score=coeffs.w_q_score,
        w_q_score_err=coeffs.w_q_score_err,
        w_pi_soft=coeffs.w_pi_soft,
        w_pi_optimistic=coeffs.w_pi_optimistic,
    )

    val_fn = val_step

    losses_val = LossTracker()
    metrics_val = ValMetrics()

    for i, batch_data in enumerate(val_ds):
        if i >= num_batches:
            break

        (
            input,
            input_global_state,
            color,
            komi,
            score,
            score_one_hot,
            policy,
            policy_aux,
            own,
            q6,
            q16,
            q50,
            q6_score,
            q16_score,
            q50_score,
            game_outcome,
        ) = batch_data

        targets = GroundTruth(
            policy=policy,
            policy_aux=policy_aux,
            score=score,
            score_one_hot=score_one_hot,
            game_outcome=game_outcome,
            own=own,
            q6=q6,
            q16=q16,
            q50=q50,
            q6_score=q6_score,
            q16_score=q16_score,
            q50_score=q50_score,
        )

        result = val_fn(
            input,
            input_global_state,
            targets,
            weights,
            model,
        )

        losses_val.update_losses(result)

        # Compute accuracy metrics
        num_moves = policy.shape[0]
        num_outcomes = score.shape[0]

        def compute_accuracy(predictions: ModelPredictions):
            predicted_moves = keras.ops.argmax(predictions.pi_logits, axis=1)
            actual_moves = keras.ops.argmax(policy, axis=1)
            correct_moves = keras.ops.sum(
                keras.ops.cast(predicted_moves == actual_moves, "int32")
            ).numpy()

            predicted_outcomes = keras.ops.argmax(predictions.game_outcome, axis=1) == 1
            actual_outcomes = score >= 0  # Actual win
            correct_outcomes = keras.ops.sum(
                keras.ops.cast(predicted_outcomes == actual_outcomes, "int32")
            ).numpy()
            return correct_moves, correct_outcomes

        correct_moves, correct_outcomes = compute_accuracy(result.predictions)
        metrics_val.increment(num_moves, num_outcomes, correct_moves, correct_outcomes)

    log_val(
        batch_num,
        losses_val,
        metrics_val,
        input,
        input_global_state,
        result.predictions,
        targets,
        model,
    )


def log_val(
    batch_num: int,
    losses: LossTracker,
    metrics: ValMetrics,
    input_planes: tf.Tensor,
    input_global_state: tf.Tensor,
    predictions: ModelPredictions,
    targets: GroundTruth,
    model: P3achyGoModel,
):
    loss_avg = losses.avg_losses["loss"]
    policy_avg = losses.avg_losses["policy"]
    policy_aux_avg = losses.avg_losses["policy_aux"]
    outcome_avg = losses.avg_losses["outcome"]
    score_pdf_avg = losses.avg_losses["score_pdf"]
    own_avg = losses.avg_losses["own"]
    q6_avg = losses.avg_losses["q6"]
    q16_avg = losses.avg_losses["q16"]
    q50_avg = losses.avg_losses["q50"]

    move_acc = metrics.correct_moves / metrics.num_moves if metrics.num_moves > 0 else 0
    outcome_acc = (
        metrics.correct_outcomes / metrics.num_outcomes
        if metrics.num_outcomes > 0
        else 0
    )

    print(
        f"[batch {batch_num}] val: "
        f"loss = {loss_avg:.4f}, "
        f"policy = {policy_avg:.4f}, "
        f"policy_aux = {policy_aux_avg:.4f}, "
        f"outcome = {outcome_avg:.4f}, "
        f"score_pdf = {score_pdf_avg:.4f}, "
        f"own = {own_avg:.4f}, "
        f"q6 = {q6_avg:.4f}, "
        f"q16 = {q16_avg:.4f}, "
        f"q50 = {q50_avg:.4f}, "
        f"move_acc = {move_acc:.4f}, "
        f"outcome_acc = {outcome_acc:.4f}"
    )

    # Log a sample board position
    log_board_position(
        batch_num, input_planes, input_global_state, predictions, targets, model
    )
