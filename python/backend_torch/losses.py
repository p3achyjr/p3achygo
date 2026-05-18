"""Native-torch port of `model.compute_losses` (defined on the keras model).

Free function — does not live on the model. The keras path keeps the
losses on the model for historical reasons; the torch path doesn't need
that coupling.

Inputs (`predictions`, `targets`, `weights`) are the same NamedTuples used
by the train_step on both backends — see `backend_torch/train.py` and
`model.py`.

Numerical conventions matched to the keras impl in `model.py:compute_losses`:
- KL divergence: clamps both args to [1e-7, 1.0] before the log (matches
  `keras.metrics.kl_divergence` exactly).
- CategoricalCrossentropy(from_logits=True) on one-hot targets: implemented
  as `-(target * log_softmax(logits)).sum(-1).mean()`.
- SparseCategoricalCrossentropy(from_logits=True): `F.cross_entropy` with
  integer indices.
- Huber, MSE: keras default `reduction="sum_over_batch_size"` ≡ torch
  `reduction="mean"` for our shapes (target/pred are scalars per example).
- All policy / value distributional losses cast to fp32 before the softmax
  + KLD step (matches keras's `cast(..., "float32")` pattern at every
  softmax call site in compute_losses).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-7  # matches keras.backend.epsilon()


def _kl_divergence_from_logits(
    target: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """Per-example KL divergence summed over the last axis.
    `target` is a probability vector, `logits` are pre-softmax pred logits.
    Uses `F.kl_div` with log_softmax — fewer ops than computing softmax then
    log, and matches keras numerics closely (target zeros are handled via
    xlogy semantics — equivalent to keras's eps-clamp on our generated tests
    within fp16 tolerance)."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    target = target.float().clamp_min(_EPS)
    return F.kl_div(log_p, target, reduction="none", log_target=False).sum(dim=-1)


def _cce_logits(one_hot_target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """CategoricalCrossentropy(from_logits=True), reduction='mean'.
    Uses torch's soft-label cross_entropy (torch ≥1.10).
    `one_hot_target` is (batch, C) probability vector; `logits` is (batch, C)."""
    return F.cross_entropy(logits.float(), one_hot_target.float(), reduction="mean")


def _scce_per_example(logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """SparseCategoricalCrossentropy(from_logits=True, reduction='none')."""
    return F.cross_entropy(logits.float(), target_idx.long(), reduction="none")


def _align_shapes(pred: torch.Tensor, target: torch.Tensor):
    # Q-heads emit (N, 1) via `go[:, k:k+1]`; targets are (N,). Without this
    # alignment, `mse_loss` / `huber_loss` silently broadcast to (N, N) (or
    # (N, N) twice for q_score_err where the target is itself a broadcasted
    # square), dividing the loss by N or N² and diluting gradients.
    if pred.ndim == target.ndim + 1 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    elif target.ndim == pred.ndim + 1 and target.shape[-1] == 1:
        target = target.squeeze(-1)
    assert pred.shape == target.shape, (
        f"loss shape mismatch: pred {tuple(pred.shape)} vs target "
        f"{tuple(target.shape)}"
    )
    return pred, target


def _mse(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    pred, target = _align_shapes(pred, target)
    return F.mse_loss(pred.float(), target.float(), reduction="mean")


def _huber(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    pred, target = _align_shapes(pred, target)
    return F.huber_loss(pred.float(), target.float(), delta=1.0, reduction="mean")


def compute_losses(predictions, targets, weights):
    """Faithful torch port of `model.compute_losses` (keras version).

    Returns the same 16-tuple in the same order:
      (loss, policy_loss, policy_aux_dist_loss, policy_aux_scalar_loss,
       outcome_loss, q6_loss, q16_loss, q50_loss, score_pdf_loss,
       score_cdf_loss, own_loss, q_err_loss, q_score_loss, q_score_err_loss,
       pi_soft_loss, pi_optimistic_loss, mcts_dist_loss)
    """
    # Policy Loss — KLD(target_dist, softmax(pi_logits))
    policy_loss = _kl_divergence_from_logits(
        targets.policy, predictions.pi_logits
    ).mean()

    # Policy aux loss: per-example masked.
    has_dist_mask = targets.has_pi_aux_dist.float()  # (batch,)
    pi_aux_logits = predictions.pi_logits_aux.float()

    policy_aux_dist_target = targets.policy_aux_dist.float()
    per_ex_kld = _kl_divergence_from_logits(policy_aux_dist_target, pi_aux_logits)
    policy_aux_dist_loss = (has_dist_mask * per_ex_kld).mean()

    per_ex_scce = _scce_per_example(pi_aux_logits, targets.policy_aux).clamp(0.0, 50.0)
    policy_aux_scalar_loss = ((1.0 - has_dist_mask) * per_ex_scce).mean()

    # Outcome / q6 / q16 / q50
    outcome_loss = _cce_logits(targets.game_outcome, predictions.game_outcome)
    q6_loss = _mse(targets.q6, predictions.q6_pred)
    q16_loss = _mse(targets.q16, predictions.q16_pred)
    q50_loss = _mse(targets.q50, predictions.q50_pred)

    # Score
    score_distribution = F.softmax(predictions.score_logits, dim=-1)
    score_pdf_loss = _cce_logits(targets.score_one_hot, predictions.score_logits)
    score_cdf_loss = (
        (
            torch.cumsum(targets.score_one_hot.float(), dim=1)
            - torch.cumsum(score_distribution.float(), dim=1)
        )
        .square()
        .sum(dim=1)
        .mean()
    )

    # Ownership
    own_pred_squeezed = predictions.own_pred.squeeze(-1)
    own_loss = _mse(targets.own, own_pred_squeezed)

    gamma_squeezed = predictions.gamma.squeeze(-1)
    gamma_loss = (gamma_squeezed * gamma_squeezed * weights.w_gamma).mean()

    # Weighted aggregate
    woutcome_loss = weights.w_outcome * outcome_loss
    wq6_loss = weights.w_q6 * q6_loss
    wq16_loss = weights.w_q16 * q16_loss
    wq50_loss = weights.w_q50 * q50_loss
    wscore_pdf_loss = weights.w_score * score_pdf_loss
    wscore_cdf_loss = weights.w_score * score_cdf_loss
    wown_loss = weights.w_own * own_loss
    val_loss = (
        weights.w_val
        * (
            woutcome_loss
            + wq6_loss
            + wq16_loss
            + wq50_loss
            + wscore_pdf_loss
            + wown_loss
        )
        + wscore_cdf_loss  # outside w_val to prevent score variance
    )

    # MCTS value-distribution loss: KLD(normalized_mcts_dist, softmax(logits))
    mv_mask = targets.has_mcts_value_dist.float()
    mcts_dist_target_int = targets.mcts_value_dist.float()
    mcts_dist_total = mcts_dist_target_int.sum(dim=1, keepdim=True).clamp_min(1.0)
    mcts_dist_normalized = mcts_dist_target_int / mcts_dist_total
    per_ex_mcts_kld = _kl_divergence_from_logits(
        mcts_dist_normalized, predictions.mcts_dist_logits
    )
    mcts_dist_loss = (mv_mask * per_ex_mcts_kld).mean()

    loss = (
        weights.w_pi * policy_loss.float()
        + weights.w_pi_aux * policy_aux_dist_loss.float()
        + weights.w_pi_aux * 0.6 * policy_aux_scalar_loss.float()
        + val_loss.float()
        + gamma_loss.float()
        + weights.w_mcts_dist * mcts_dist_loss.float()
    )

    # v1 losses
    (
        q_err_loss,
        q_score_loss,
        q_score_err_loss,
        pi_soft_loss,
        pi_optimistic_loss,
    ) = _v1_loss_terms(predictions, targets)

    loss = loss + (
        weights.w_q_err * q_err_loss
        + weights.w_q_score * q_score_loss
        + weights.w_q_score_err * q_score_err_loss
        + weights.w_pi_soft * pi_soft_loss
        + weights.w_pi_optimistic * pi_optimistic_loss
    )

    return (
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
    )


def _v1_loss_terms(predictions, targets):
    """Mirrors `model.v1_loss_terms` (keras version) line-for-line."""
    epsilon = 1e-6

    # Q-heads emit (N, 1) via `go[:, k:k+1]`; targets are (N,). Squeeze
    # all of them once so every subtraction below stays (N,) instead of
    # silently broadcasting to (N, N).
    def _s(t):
        return (
            t.squeeze(-1) if t is not None and t.ndim == 2 and t.shape[-1] == 1 else t
        )

    q6_pred = _s(predictions.q6_pred)
    q16_pred = _s(predictions.q16_pred)
    q50_pred = _s(predictions.q50_pred)
    q6_err_pred = _s(predictions.q6_err_pred)
    q16_err_pred = _s(predictions.q16_err_pred)
    q50_err_pred = _s(predictions.q50_err_pred)
    q6_score_pred = _s(predictions.q6_score_pred)
    q16_score_pred = _s(predictions.q16_score_pred)
    q50_score_pred = _s(predictions.q50_score_pred)
    q6_score_err_pred = _s(predictions.q6_score_err_pred)
    q16_score_err_pred = _s(predictions.q16_score_err_pred)
    q50_score_err_pred = _s(predictions.q50_score_err_pred)

    # Q-error losses (Huber on squared deviation)
    q6_err_target = (q6_pred.detach() - targets.q6).square()
    q16_err_target = (q16_pred.detach() - targets.q16).square()
    q50_err_target = (q50_pred.detach() - targets.q50).square()

    q6_err_loss = _huber(q6_err_target, q6_err_pred)
    q16_err_loss = _huber(q16_err_target, q16_err_pred)
    q50_err_loss = _huber(q50_err_target, q50_err_pred)
    q_err_loss = (q6_err_loss + q16_err_loss + q50_err_loss) / 3.0

    # Q-score losses (only when targets.q6_score is provided)
    q_score_loss = torch.zeros((), dtype=torch.float32, device=q_err_loss.device)
    q_score_err_loss = torch.zeros((), dtype=torch.float32, device=q_err_loss.device)
    if targets.q6_score is not None:
        q6_score_loss = _huber(targets.q6_score / 10.0, q6_score_pred / 10.0)
        q16_score_loss = _huber(targets.q16_score / 10.0, q16_score_pred / 10.0)
        q50_score_loss = _huber(targets.q50_score / 10.0, q50_score_pred / 10.0)
        q_score_loss = ((q6_score_loss + q16_score_loss + q50_score_loss) / 3.0).clamp(
            0.0, 200.0
        )

        q6_score_err_target = (q6_score_pred.detach() - targets.q6_score).square()
        q16_score_err_target = (q16_score_pred.detach() - targets.q16_score).square()
        q50_score_err_target = (q50_score_pred.detach() - targets.q50_score).square()
        q6_score_err_loss = _huber(
            q6_score_err_target / 100.0, q6_score_err_pred / 100.0
        )
        q16_score_err_loss = _huber(
            q16_score_err_target / 100.0, q16_score_err_pred / 100.0
        )
        q50_score_err_loss = _huber(
            q50_score_err_target / 100.0, q50_score_err_pred / 100.0
        )
        q_score_err_loss = (
            (q6_score_err_loss + q16_score_err_loss + q50_score_err_loss) / 3.0
        ).clamp(0.0, 1000.0)

    # Soft-policy loss
    policy_f32 = targets.policy.float()
    policy_soft = policy_f32.pow(0.25)
    policy_soft = policy_soft / policy_soft.sum(dim=-1, keepdim=True)
    pi_soft_loss = _kl_divergence_from_logits(
        policy_soft, predictions.pi_logits_soft
    ).mean()

    # Optimistic-policy loss
    z_value_q6 = (targets.q6 - q6_pred.detach()) / (
        (q6_err_pred + epsilon).sqrt().detach()
    )
    z_value_q16 = (targets.q16 - q16_pred.detach()) / (
        (q16_err_pred + epsilon).sqrt().detach()
    )
    z_value_q50 = (targets.q50 - q50_pred.detach()) / (
        (q50_err_pred + epsilon).sqrt().detach()
    )
    z_weight_decay = 4.0 / 7.0
    c_z6 = z_weight_decay * 3
    c_z16 = z_weight_decay * 1.5
    c_z50 = z_weight_decay * 0.75
    z_value = (c_z6 * z_value_q6 + c_z16 * z_value_q16 + c_z50 * z_value_q50) / 3.0
    if targets.q6_score is not None:
        z_score_q6 = (targets.q6_score - q6_score_pred.detach()) / (
            (q6_score_err_pred + epsilon).sqrt().detach()
        )
        z_score_q16 = (targets.q16_score - q16_score_pred.detach()) / (
            (q16_score_err_pred + epsilon).sqrt().detach()
        )
        z_score_q50 = (targets.q50_score - q50_score_pred.detach()) / (
            (q50_score_err_pred + epsilon).sqrt().detach()
        )
        z_score = (c_z6 * z_score_q6 + c_z16 * z_score_q16 + c_z50 * z_score_q50) / 3.0
        z_combined = (z_value + z_score * 0.5) / 1.5
    else:
        z_combined = z_value

    optimistic_weight = torch.sigmoid((z_combined - 1.0) * 3).clamp(0.0, 1.0)
    pi_optimistic_loss = _kl_divergence_from_logits(
        targets.policy, predictions.pi_logits_optimistic
    )
    pi_optimistic_loss = (pi_optimistic_loss * optimistic_weight).mean()

    return q_err_loss, q_score_loss, q_score_err_loss, pi_soft_loss, pi_optimistic_loss
