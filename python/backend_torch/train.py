from __future__ import annotations

import torch
from typing import NamedTuple, Optional, TYPE_CHECKING

from backend_torch.losses import compute_losses as _torch_compute_losses

if TYPE_CHECKING:
    from backend_torch.model import P3achyGoModel
    from loss_coeffs import LossWeights


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
    # Zero gradients (set_to_none avoids a second pass to write zeros).
    model.zero_grad(set_to_none=True)
    # True mixed precision: weights stay fp32, compute drops to fp16 for
    # matmul/conv kernels.
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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

    # Native-torch loss computation (port of `model.compute_losses`).
    # See backend_torch/losses.py for parity notes.
    loss_outputs = _torch_compute_losses(predictions, targets, weights)

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

    # L2 regularization is handled by weight_decay in the optimizer.
    total_loss = loss

    # GradScaler-backed loss scaling on GPU; identity otherwise.
    scaled_loss = (
        optimizer.scale_loss(total_loss)
        if hasattr(optimizer, "scale_loss")
        else total_loss
    )
    scaled_loss.backward()

    # ConvMuon's step calls `clip_grad_norm_` which computes the global
    # norm in fp32 and stashes it on `last_grad_norm` — read it back, no
    # second fp32 reduction needed.
    optimizer.apply()  # GradScaler reads param.grad; takes no args
    grad_norm = float(getattr(optimizer, "last_grad_norm", 0.0))

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
        grad_norm=grad_norm,
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
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
    ) = _torch_compute_losses(predictions, targets, weights)

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
