from __future__ import annotations

import tensorflow as tf
from typing import NamedTuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only imports break the model→backend_shim→backend_tf.train→model
    # cycle at runtime.
    from model import P3achyGoModel, LossWeights


class ModelPredictions(NamedTuple):
    """Model prediction outputs."""

    pi_logits: tf.Tensor
    pi_logits_aux: tf.Tensor
    game_outcome: tf.Tensor
    score_logits: tf.Tensor
    own_pred: tf.Tensor
    q6_pred: tf.Tensor
    q16_pred: tf.Tensor
    q50_pred: tf.Tensor
    gamma: tf.Tensor
    # v1 predictions (optional)
    q6_err_pred: Optional[tf.Tensor] = None
    q16_err_pred: Optional[tf.Tensor] = None
    q50_err_pred: Optional[tf.Tensor] = None
    q6_score_pred: Optional[tf.Tensor] = None
    q16_score_pred: Optional[tf.Tensor] = None
    q50_score_pred: Optional[tf.Tensor] = None
    q6_score_err_pred: Optional[tf.Tensor] = None
    q16_score_err_pred: Optional[tf.Tensor] = None
    q50_score_err_pred: Optional[tf.Tensor] = None
    pi_logits_soft: Optional[tf.Tensor] = None
    pi_logits_optimistic: Optional[tf.Tensor] = None
    mcts_dist_logits: Optional[tf.Tensor] = None
    mcts_dist_probs: Optional[tf.Tensor] = None


class GroundTruth(NamedTuple):
    """Ground truth labels for training."""

    policy: tf.Tensor
    policy_aux: tf.Tensor
    score: tf.Tensor
    score_one_hot: tf.Tensor
    game_outcome: tf.Tensor
    own: tf.Tensor
    q6: tf.Tensor
    q16: tf.Tensor
    q50: tf.Tensor
    # v1 labels (optional)
    q6_score: Optional[tf.Tensor] = None
    q16_score: Optional[tf.Tensor] = None
    q50_score: Optional[tf.Tensor] = None
    # new optional labels
    policy_aux_dist: Optional[tf.Tensor] = None  # float32[NUM_MOVES]
    has_pi_aux_dist: Optional[tf.Tensor] = None  # bool scalar
    mcts_value_dist: Optional[tf.Tensor] = None  # int32[NUM_V_BUCKETS]
    has_mcts_value_dist: Optional[tf.Tensor] = None  # bool scalar


class TrainStepResult(NamedTuple):
    """Result from a training step."""

    predictions: ModelPredictions
    total_loss: tf.Tensor
    policy_loss: tf.Tensor
    policy_aux_dist_loss: tf.Tensor
    policy_aux_scalar_loss: tf.Tensor
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
    mcts_dist_loss: Optional[tf.Tensor] = None
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
            mcts_dist_logits,
            mcts_dist_probs,
        ) = model_outputs

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
            mcts_dist_logits=mcts_dist_logits,
            mcts_dist_probs=mcts_dist_probs,
        )

        # Compute losses for both heads
        loss_outputs = model.compute_losses(predictions, targets, weights)

        # Unpack loss outputs
        (
            loss,
            policy_loss,
            policy_aux_dist_loss,
            policy_aux_scalar_loss,
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
            mcts_dist_loss,
        ) = loss_outputs

        reg_loss = tf.math.add_n(model.losses)
        total_loss = loss + reg_loss
        scaled_loss = optimizer.scale_loss(total_loss)

    gradients = g.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    scale = tf.cast(
        optimizer.dynamic_scale if optimizer.built else optimizer.initial_scale,
        tf.float32,
    )
    unscaled_gradients = [g / scale for g in gradients]

    return TrainStepResult(
        predictions=predictions,
        total_loss=total_loss,
        policy_loss=policy_loss,
        policy_aux_dist_loss=policy_aux_dist_loss,
        policy_aux_scalar_loss=policy_aux_scalar_loss,
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
        mcts_dist_loss=mcts_dist_loss,
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
        mcts_dist_logits,
        mcts_dist_probs,
    ) = model_outputs

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
        mcts_dist_logits=mcts_dist_logits,
        mcts_dist_probs=mcts_dist_probs,
    )

    # Compute losses for both heads
    loss_outputs = model.compute_losses(predictions, targets, weights)

    # Unpack loss outputs
    (
        loss,
        policy_loss,
        policy_aux_dist_loss,
        policy_aux_scalar_loss,
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
        mcts_dist_loss,
    ) = loss_outputs

    reg_loss = tf.math.add_n(model.losses)
    total_loss = loss + reg_loss

    return TrainStepResult(
        predictions=predictions,
        total_loss=total_loss,
        policy_loss=policy_loss,
        policy_aux_dist_loss=policy_aux_dist_loss,
        policy_aux_scalar_loss=policy_aux_scalar_loss,
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
        mcts_dist_loss=mcts_dist_loss,
    )
