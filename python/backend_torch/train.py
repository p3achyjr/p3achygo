from __future__ import annotations

import torch
from typing import NamedTuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from model import P3achyGoModel, LossWeights


class ModelPredictions(NamedTuple):
    """Model prediction outputs."""

    pi_logits: torch.Tensor
    pi_logits_aux: torch.Tensor
    game_outcome: torch.Tensor
    score_logits: torch.Tensor
    own_pred: torch.Tensor
    q6_pred: torch.Tensor
    q16_pred: torch.Tensor
    q50_pred: torch.Tensor
    gamma: torch.Tensor
    # v1 predictions (optional)
    q6_err_pred: Optional[torch.Tensor] = None
    q16_err_pred: Optional[torch.Tensor] = None
    q50_err_pred: Optional[torch.Tensor] = None
    q6_score_pred: Optional[torch.Tensor] = None
    q16_score_pred: Optional[torch.Tensor] = None
    q50_score_pred: Optional[torch.Tensor] = None
    q6_score_err_pred: Optional[torch.Tensor] = None
    q16_score_err_pred: Optional[torch.Tensor] = None
    q50_score_err_pred: Optional[torch.Tensor] = None
    pi_logits_soft: Optional[torch.Tensor] = None
    pi_logits_optimistic: Optional[torch.Tensor] = None
    mcts_dist_logits: Optional[torch.Tensor] = None
    mcts_dist_probs: Optional[torch.Tensor] = None


class GroundTruth(NamedTuple):
    """Ground truth labels for training."""

    policy: torch.Tensor
    policy_aux: torch.Tensor
    score: torch.Tensor
    score_one_hot: torch.Tensor
    game_outcome: torch.Tensor
    own: torch.Tensor
    q6: torch.Tensor
    q16: torch.Tensor
    q50: torch.Tensor
    # v1 labels (optional)
    q6_score: Optional[torch.Tensor] = None
    q16_score: Optional[torch.Tensor] = None
    q50_score: Optional[torch.Tensor] = None
    # new optional labels
    policy_aux_dist: Optional[torch.Tensor] = None  # float32[NUM_MOVES]
    has_pi_aux_dist: Optional[torch.Tensor] = None  # bool scalar
    mcts_value_dist: Optional[torch.Tensor] = None  # int32[NUM_V_BUCKETS]
    has_mcts_value_dist: Optional[torch.Tensor] = None  # bool scalar


class TrainStepResult(NamedTuple):
    """Result from a training step."""

    predictions: ModelPredictions
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    policy_aux_dist_loss: torch.Tensor
    policy_aux_scalar_loss: torch.Tensor
    outcome_loss: torch.Tensor
    q6_loss: torch.Tensor
    q16_loss: torch.Tensor
    q50_loss: torch.Tensor
    score_pdf_loss: torch.Tensor
    score_cdf_loss: torch.Tensor
    own_loss: torch.Tensor
    # v1 only
    q_err_loss: Optional[torch.Tensor] = None
    q_score_loss: Optional[torch.Tensor] = None
    q_score_err_loss: Optional[torch.Tensor] = None
    pi_soft_loss: Optional[torch.Tensor] = None
    pi_optimistic_loss: Optional[torch.Tensor] = None
    mcts_dist_loss: Optional[torch.Tensor] = None
    grad_norm: float = 0.0


def train_step(
    input: torch.Tensor,
    input_global_state: torch.Tensor,
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
    # Zero gradients on all trainable variables (keras-side .value is the
    # underlying torch.Tensor parameter).
    for v in model.trainable_weights:
        if v.value.grad is not None:
            v.value.grad = None
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

    # L2 regularization is handled by weight_decay in the keras optimizer.
    total_loss = loss

    # Use keras's loss-scaling pre/post hooks if the optimizer is a
    # LossScaleOptimizer; otherwise the call is identity.
    scaled_loss = (
        optimizer.scale_loss(total_loss)
        if hasattr(optimizer, "scale_loss")
        else total_loss
    )
    scaled_loss.backward()

    # Compute the total grad norm for logging. Keras' optimizer applies its
    # configured clipping (e.g. `global_clipnorm`) inside `apply` itself, so
    # we don't clip here — that would double-clip and ignore the user's
    # `global_clipnorm` setting.
    #
    # When wrapped in LossScaleOptimizer, `.grad` is the loss-scaled gradient
    # (loss_scale_factor × true gradient). Keras unscales internally inside
    # `apply`, but we report grad_norm in the unscaled space so the value is
    # comparable to TF's output and the user-set `global_clipnorm`.
    params = [v.value for v in model.trainable_weights]
    grads = [p.grad for p in params]
    scaled_norm = torch.sqrt(
        sum(
            torch.sum(g.detach().to(torch.float32) ** 2) for g in grads if g is not None
        )
    )
    # LossScaleOptimizer scale lives on `dynamic_scale` post-build,
    # `initial_scale` pre-build; mirrors `LossScaleOptimizer.scale_loss`.
    if hasattr(optimizer, "scale_loss"):
        loss_scale = float(
            optimizer.dynamic_scale if optimizer.built else optimizer.initial_scale
        )
    else:
        loss_scale = 1.0
    grad_norm = scaled_norm / loss_scale

    # Apply via keras 3's unified optimizer API (`apply` works on either
    # backend and across LossScaleOptimizer / ConvMuon / SGD). Keras does
    # *not* zero out `.grad` after applying — torch's autograd accumulates,
    # so we zero out at the top of the next train_step.
    optimizer.apply(grads, model.trainable_weights)

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
        grad_norm=grad_norm.item(),
    )


@torch.no_grad()
def val_step(
    input: torch.Tensor,
    input_global_state: torch.Tensor,
    targets: GroundTruth,
    weights: LossWeights,
    model: P3achyGoModel,
) -> TrainStepResult:
    """Validation step — forward + loss only, no gradient update."""
    model_outputs = model(input, input_global_state, training=False)

    (
        pi_logits,
        _,
        outcome_logits,
        _,
        ownership,
        score_logits,
        _,
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
    ) = model.compute_losses(predictions, targets, weights)

    return TrainStepResult(
        predictions=predictions,
        total_loss=loss,
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
