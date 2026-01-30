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
    grad_norm: float
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


class ValStepResult(NamedTuple):
    """Result from a validation step."""

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


@tf.function
def train_step_v0(
    input: tf.Tensor,
    input_global_state: tf.Tensor,
    targets: GroundTruth,
    weights: LossWeights,
    model: P3achyGoModel,
    optimizer,
) -> TrainStepResult:
    """
    Training step for v0 models (no one-batch-norm).

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
        # Get model outputs (v0: 12 outputs)
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
        ) = model(input, input_global_state, training=True)

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
        )

        # Compute losses
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
            _,  # q_err_loss (v0 doesn't have these)
            _,  # q_score_loss
            _,  # q_score_err_loss
            _,  # pi_soft_loss
            _,  # pi_optimistic_loss
        ) = model.compute_losses(predictions, targets, weights)

        # Add regularization
        reg_loss = tf.math.add_n(model.losses)
        total_loss = loss + reg_loss
        scaled_loss = optimizer.scale_loss(total_loss)

        # Clip loss for numerical stability
        # clipped_loss = tf.clip_by_value(total_loss, -100.0, 100.0)

    gradients = g.gradient(scaled_loss, model.trainable_variables)
    # gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    scale = tf.cast(
        optimizer.dynamic_scale if optimizer.built else optimizer.initial_scale,
        tf.float32,
    )
    return TrainStepResult(
        predictions=predictions,
        total_loss=total_loss,
        grad_norm=tf.linalg.global_norm(gradients) / scale,
        policy_loss=policy_loss,
        policy_aux_loss=policy_aux_loss,
        outcome_loss=outcome_loss,
        q6_loss=q6_loss,
        q16_loss=q16_loss,
        q50_loss=q50_loss,
        score_pdf_loss=score_pdf_loss,
        score_cdf_loss=score_cdf_loss,
        own_loss=own_loss,
    )


@tf.function
def train_step_v1(
    input: tf.Tensor,
    input_global_state: tf.Tensor,
    targets: GroundTruth,
    weights: LossWeights,
    model: P3achyGoModel,
    optimizer,
    fvi_weight: float = 0.2,
    bn_weight: float = 0.8,
) -> TrainStepResult:
    """
    Training step for v1 models (with one-batch-norm).

    v1 always uses one-batch-norm: the model returns 46 outputs
    (23 FVI + 23 BN). We compute losses for both and combine them
    with weighted sum: 0.2 * fvi_loss + 0.8 * bn_loss.

    Args:
        input: Board state tensor
        input_global_state: Global state tensor
        targets: GroundTruth with labels
        weights: LossWeights with loss weights
        model: The model instance
        optimizer: The optimizer
        fvi_weight: Weight for FVI heads (default 0.2)
        bn_weight: Weight for BN heads (default 0.8)

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
        ) = model_outputs[:23]

        # Extract BN outputs (second 23)
        (
            pi_logits_bn,
            pi_bn,
            outcome_logits_bn,
            outcome_probs_bn,
            ownership_bn,
            score_logits_bn,
            score_probs_bn,
            gamma_bn,
            pi_logits_aux_bn,
            q6_pred_bn,
            q16_pred_bn,
            q50_pred_bn,
            q6_err_pred_bn,
            q16_err_pred_bn,
            q50_err_pred_bn,
            q6_score_pred_bn,
            q16_score_pred_bn,
            q50_score_pred_bn,
            q6_score_err_pred_bn,
            q16_score_err_pred_bn,
            q50_score_err_pred_bn,
            pi_logits_soft_bn,
            pi_logits_optimistic_bn,
        ) = model_outputs[23:]

        # Create FVI predictions
        predictions_fvi = ModelPredictions(
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

        # Create BN predictions
        predictions_bn = ModelPredictions(
            pi_logits=pi_logits_bn,
            pi_logits_aux=pi_logits_aux_bn,
            game_outcome=outcome_logits_bn,
            score_logits=score_logits_bn,
            own_pred=ownership_bn,
            q6_pred=q6_pred_bn,
            q16_pred=q16_pred_bn,
            q50_pred=q50_pred_bn,
            gamma=gamma_bn,
            q6_err_pred=q6_err_pred_bn,
            q16_err_pred=q16_err_pred_bn,
            q50_err_pred=q50_err_pred_bn,
            q6_score_pred=q6_score_pred_bn,
            q16_score_pred=q16_score_pred_bn,
            q50_score_pred=q50_score_pred_bn,
            q6_score_err_pred=q6_score_err_pred_bn,
            q16_score_err_pred=q16_score_err_pred_bn,
            q50_score_err_pred=q50_score_err_pred_bn,
            pi_logits_soft=pi_logits_soft_bn,
            pi_logits_optimistic=pi_logits_optimistic_bn,
        )

        # Compute losses for both heads
        loss_outputs_fvi = model.compute_losses(predictions_fvi, targets, weights)
        loss_outputs_bn = model.compute_losses(predictions_bn, targets, weights)

        # Weighted combination: 0.2 * FVI + 0.8 * BN
        loss_outputs = tuple(
            fvi_weight * loss_fvi + bn_weight * loss_bn
            for loss_fvi, loss_bn in zip(loss_outputs_fvi, loss_outputs_bn)
        )

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

    # Return FVI predictions for logging
    return TrainStepResult(
        predictions=predictions_fvi,
        total_loss=total_loss,
        grad_norm=tf.linalg.global_norm(gradients) / scale,
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


@tf.function
def val_step_v0(
    input: tf.Tensor,
    input_global_state: tf.Tensor,
    targets: GroundTruth,
    weights: LossWeights,
    model: P3achyGoModel,
) -> ValStepResult:
    """Validation step for v0 models."""
    # Get model outputs (v0: 12 outputs)
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
    ) = model(input, input_global_state, training=False)

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
    )

    # Compute losses
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
        _,
        _,
        _,
        _,
        _,
    ) = model.compute_losses(predictions, targets, weights)

    reg_loss = tf.math.add_n(model.losses)
    total_loss = loss + reg_loss

    return ValStepResult(
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
    )


@tf.function
def val_step_v1(
    input: tf.Tensor,
    input_global_state: tf.Tensor,
    targets: GroundTruth,
    weights: LossWeights,
    model: P3achyGoModel,
    fvi_weight: float = 0.2,
    bn_weight: float = 0.8,
) -> ValStepResult:
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

    # Extract BN outputs (second 23)
    (
        pi_logits_bn,
        pi_bn,
        outcome_logits_bn,
        outcome_probs_bn,
        ownership_bn,
        score_logits_bn,
        score_probs_bn,
        gamma_bn,
        pi_logits_aux_bn,
        q6_pred_bn,
        q16_pred_bn,
        q50_pred_bn,
        q6_err_pred_bn,
        q16_err_pred_bn,
        q50_err_pred_bn,
        q6_score_pred_bn,
        q16_score_pred_bn,
        q50_score_pred_bn,
        q6_score_err_pred_bn,
        q16_score_err_pred_bn,
        q50_score_err_pred_bn,
        pi_logits_soft_bn,
        pi_logits_optimistic_bn,
    ) = model_outputs[23:]

    # Create FVI predictions
    predictions_fvi = ModelPredictions(
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

    # Create BN predictions
    predictions_bn = ModelPredictions(
        pi_logits=pi_logits_bn,
        pi_logits_aux=pi_logits_aux_bn,
        game_outcome=outcome_logits_bn,
        score_logits=score_logits_bn,
        own_pred=ownership_bn,
        q6_pred=q6_pred_bn,
        q16_pred=q16_pred_bn,
        q50_pred=q50_pred_bn,
        gamma=gamma_bn,
        q6_err_pred=q6_err_pred_bn,
        q16_err_pred=q16_err_pred_bn,
        q50_err_pred=q50_err_pred_bn,
        q6_score_pred=q6_score_pred_bn,
        q16_score_pred=q16_score_pred_bn,
        q50_score_pred=q50_score_pred_bn,
        q6_score_err_pred=q6_score_err_pred_bn,
        q16_score_err_pred=q16_score_err_pred_bn,
        q50_score_err_pred=q50_score_err_pred_bn,
        pi_logits_soft=pi_logits_soft_bn,
        pi_logits_optimistic=pi_logits_optimistic_bn,
    )

    # Compute losses for both heads
    loss_outputs_fvi = model.compute_losses(predictions_fvi, targets, weights)
    loss_outputs_bn = model.compute_losses(predictions_bn, targets, weights)

    # Weighted combination
    loss_outputs = tuple(
        fvi_weight * loss_fvi + bn_weight * loss_bn
        for loss_fvi, loss_bn in zip(loss_outputs_fvi, loss_outputs_bn)
    )

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

    return ValStepResult(
        predictions=predictions_fvi,
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
        self.losses = []
        self.min_losses = defaultdict(lambda: self.MAX_LOSS)
        self.avg_losses = defaultdict(lambda: 0)

    def update_losses(
        self,
        loss,
        policy_loss,
        policy_aux_loss,
        outcome_loss,
        score_pdf_loss,
        score_cdf_loss,
        own_loss,
        q30_loss,
        q100_loss,
        q200_loss,
    ):
        self.losses.append(
            {
                "loss": loss,
                "policy": policy_loss,
                "policy_aux": policy_aux_loss,
                "outcome": outcome_loss,
                "score_pdf": score_pdf_loss,
                "score_cdf": score_cdf_loss,
                "own": own_loss,
                "q30": q30_loss,
                "q100": q100_loss,
                "q200": q200_loss,
            }
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
        self.min_losses["q30"] = min(self.min_losses["q30"], q30_loss)
        self.min_losses["q100"] = min(self.min_losses["q100"], q100_loss)
        self.min_losses["q200"] = min(self.min_losses["q200"], q200_loss)

        n = len(self.losses)
        r_m = (n - 1) / n
        r_c = 1 / n
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
        self.avg_losses["q30"] = self.avg_losses["q30"] * r_m + q30_loss * r_c
        self.avg_losses["q100"] = self.avg_losses["q100"] * r_m + q100_loss * r_c
        self.avg_losses["q200"] = self.avg_losses["q200"] * r_m + q200_loss * r_c


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

    # Select coefficients based on mode and model version
    if mode == Mode.SL:
        coeffs = LossCoeffs.SLCoeffs()
    elif model.version >= 1:
        coeffs = LossCoeffs.RLCoeffsV1()
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
        w_q6=coeffs.w_q30,
        w_q16=coeffs.w_q100,
        w_q50=coeffs.w_q200,
        w_gamma=coeffs.w_gamma,
        w_q_err=coeffs.w_q_err if model.version >= 1 else 0.0,
        w_q_score=coeffs.w_q_score if model.version >= 1 else 0.0,
        w_q_score_err=coeffs.w_q_score_err if model.version >= 1 else 0.0,
        w_pi_soft=coeffs.w_pi_soft if model.version >= 1 else 0.0,
        w_pi_optimistic=coeffs.w_pi_optimistic if model.version >= 1 else 0.0,
    )

    # Select train function based on model version
    train_fn = train_step_v1 if model.version >= 1 else train_step_v0

    optimizer = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=momentum,
        global_clipnorm=100.0,  # purely for numerical stability
    )
    if is_gpu:
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    losses_train = LossTracker()
    local_batch_num = 0
    for _ in range(epochs):
        # train
        for batch_data in train_ds:
            if save_path and save_interval and batch_num % save_interval == 0:
                save_model(model, batch_num, save_path)
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

            # transforms.expand always returns v1-sized tensors (15 planes, 8 features)
            # Slice to v0 size if needed (13 planes, 7 features)
            if model.version == 0:
                input = input[:, :, :, :13]  # Keep first 13 planes
                input_global_state = input_global_state[:, :7]  # Keep first 7 features

            if model.version == 0:
                # v0 dataset format

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
                )

                result = train_fn(
                    input,
                    input_global_state,
                    targets,
                    weights,
                    model,
                    optimizer,
                )
            else:
                # v1 dataset format (includes additional labels)

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

                result = train_fn(
                    input,
                    input_global_state,
                    targets,
                    weights,
                    model,
                    optimizer,
                )

            losses_train.update_losses(
                result.total_loss.numpy(),
                result.policy_loss.numpy(),
                result.policy_aux_loss.numpy(),
                result.outcome_loss.numpy(),
                result.score_pdf_loss.numpy(),
                result.score_cdf_loss.numpy(),
                result.own_loss.numpy(),
                result.q6_loss.numpy(),
                result.q16_loss.numpy(),
                result.q50_loss.numpy(),
            )

            local_batch_num += 1
            batch_num += 1

            # Query snapshot manager after every training step
            if ss_manager and ss_manager.should_take_snapshot(local_batch_num):
                ss_manager.take_snapshot(model)

            if local_batch_num % log_interval == 0:
                log_train(batch_num, losses_train, summary_writer, mode, model.version)

                # Log board position with predictions every 10th log interval
                if local_batch_num % (log_interval * 10) == 0:
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
        save_model(model, batch_num, save_path)
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
    return local_batch_num + batch_num


def log_train(
    batch_num: int,
    losses: LossTracker,
    summary_writer: tf.summary.SummaryWriter,
    mode: Mode,
    version: int,
):
    mode_str = "sl" if mode == Mode.SL else "rl"

    loss_avg = losses.avg_losses["loss"]
    policy_avg = losses.avg_losses["policy"]
    policy_aux_avg = losses.avg_losses["policy_aux"]
    outcome_avg = losses.avg_losses["outcome"]
    score_pdf_avg = losses.avg_losses["score_pdf"]
    score_cdf_avg = losses.avg_losses["score_cdf"]
    own_avg = losses.avg_losses["own"]
    q30_avg = losses.avg_losses["q30"]
    q100_avg = losses.avg_losses["q100"]
    q200_avg = losses.avg_losses["q200"]

    loss_min = losses.min_losses["loss"]
    policy_min = losses.min_losses["policy"]
    policy_aux_min = losses.min_losses["policy_aux"]
    outcome_min = losses.min_losses["outcome"]
    score_pdf_min = losses.min_losses["score_pdf"]
    score_cdf_min = losses.min_losses["score_cdf"]
    own_min = losses.min_losses["own"]
    q30_min = losses.min_losses["q30"]
    q100_min = losses.min_losses["q100"]
    q200_min = losses.min_losses["q200"]

    print(
        f"[batch {batch_num}] {mode_str}: "
        f"loss = {loss_avg:.4f} ({loss_min:.4f}), "
        f"policy = {policy_avg:.4f} ({policy_min:.4f}), "
        f"policy_aux = {policy_aux_avg:.4f} ({policy_aux_min:.4f}), "
        f"outcome = {outcome_avg:.4f} ({outcome_min:.4f}), "
        f"score_pdf = {score_pdf_avg:.4f} ({score_pdf_min:.4f}), "
        f"score_cdf = {score_cdf_avg:.4f} ({score_cdf_min:.4f}), "
        f"own = {own_avg:.4f} ({own_min:.4f}), "
        f"q30 = {q30_avg:.4f} ({q30_min:.4f}), "
        f"q100 = {q100_avg:.4f} ({q100_min:.4f}), "
        f"q200 = {q200_avg:.4f} ({q200_min:.4f})"
    )

    with summary_writer.as_default():
        tf.summary.scalar(f"{mode_str}/loss", loss_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/policy", policy_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/policy_aux", policy_aux_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/outcome", outcome_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/score_pdf", score_pdf_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/score_cdf", score_cdf_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/own", own_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/q30", q30_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/q100", q100_avg, step=batch_num)
        tf.summary.scalar(f"{mode_str}/q200", q200_avg, step=batch_num)

    # Reset losses
    losses.losses.clear()


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
    print(GoBoard.to_string(board))
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
    score_mean_pred = np.sum(score_pred * np.arange(-400, 400) * 0.05)
    score_std_pred = np.sqrt(
        np.sum(score_pred * ((np.arange(-400, 400) * 0.05 - score_mean_pred) ** 2))
    )
    print(f"\nScore Prediction:")
    print(f"  Predicted: {score_mean_pred:+.1f} ± {score_std_pred:.1f}")
    print(f"  Actual:    {score_target:+.1f}")

    # Top policy moves
    print(f"\nTop 5 Policy Moves:")
    print(f"  {'Predicted':<20} {'Target':<20}")
    for i in range(5):
        pred_move = move_to_coords(top_indices[i])
        pred_prob = policy_pred[top_indices[i]]
        target_move = move_to_coords(top_indices_target[i])
        target_prob = policy_target[top_indices_target[i]]
        print(
            f"  {pred_move:>6} {pred_prob:>6.1%}          {target_move:>6} {target_prob:>6.1%}"
        )

    print(f"{'='*60}\n")


def save_model(model: P3achyGoModel, batch_num: int, save_path: str):
    filename = f"model_{batch_num}.keras"
    filepath = Path(save_path) / filename
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

    # Select coefficients based on mode and model version
    if mode == Mode.SL:
        coeffs = LossCoeffs.SLCoeffs()
    elif model.version >= 1:
        coeffs = LossCoeffs.RLCoeffsV1()
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
        w_q6=coeffs.w_q30,
        w_q16=coeffs.w_q100,
        w_q50=coeffs.w_q200,
        w_gamma=coeffs.w_gamma,
        w_q_err=coeffs.w_q_err if model.version >= 1 else 0.0,
        w_q_score=coeffs.w_q_score if model.version >= 1 else 0.0,
        w_q_score_err=coeffs.w_q_score_err if model.version >= 1 else 0.0,
        w_pi_soft=coeffs.w_pi_soft if model.version >= 1 else 0.0,
        w_pi_optimistic=coeffs.w_pi_optimistic if model.version >= 1 else 0.0,
    )

    # Select val function based on model version
    val_fn = val_step_v1 if model.version >= 1 else val_step_v0

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

        # transforms.expand always returns v1-sized tensors (15 planes, 8 features)
        # Slice to v0 size if needed (13 planes, 7 features)
        if model.version == 0:
            input = input[:, :, :, :13]  # Keep first 13 planes
            input_global_state = input_global_state[:, :7]  # Keep first 7 features

        if model.version == 0:
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
            )

            result = val_fn(
                input,
                input_global_state,
                targets,
                weights,
                model,
            )
        else:
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

        losses_val.update_losses(
            result.total_loss.numpy(),
            result.policy_loss.numpy(),
            result.policy_aux_loss.numpy(),
            result.outcome_loss.numpy(),
            result.score_pdf_loss.numpy(),
            result.score_cdf_loss.numpy(),
            result.own_loss.numpy(),
            result.q6_loss.numpy(),
            result.q16_loss.numpy(),
            result.q50_loss.numpy(),
        )

        # Compute accuracy metrics
        num_moves = policy.shape[0]
        num_outcomes = score.shape[0]

        predicted_moves = keras.ops.argmax(result.predictions.pi_logits, axis=1)
        actual_moves = keras.ops.argmax(policy, axis=1)
        correct_moves = keras.ops.sum(
            keras.ops.cast(predicted_moves == actual_moves, "int32")
        ).numpy()

        predicted_outcomes = (
            result.predictions.game_outcome[:, 1] > 0.5
        )  # Win probability
        actual_outcomes = score >= 0  # Actual win
        correct_outcomes = keras.ops.sum(
            keras.ops.cast(predicted_outcomes == actual_outcomes, "int32")
        ).numpy()

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
    q30_avg = losses.avg_losses["q30"]
    q100_avg = losses.avg_losses["q100"]
    q200_avg = losses.avg_losses["q200"]

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
        f"q30 = {q30_avg:.4f}, "
        f"q100 = {q100_avg:.4f}, "
        f"q200 = {q200_avg:.4f}, "
        f"move_acc = {move_acc:.4f}, "
        f"outcome_acc = {outcome_acc:.4f}"
    )

    # Log a sample board position
    log_board_position(
        batch_num, input_planes, input_global_state, predictions, targets, model
    )
