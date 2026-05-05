from __future__ import annotations

import functools
import math
import numpy as np
from typing import Any, NamedTuple, Optional

from board import GoBoard
from collections import defaultdict
from constants import *
from model import P3achyGoModel, ModelPredictions, GroundTruth, LossWeights
from pathlib import Path
from loss_coeffs import LossCoeffs
from enum import Enum
from weight_snapshot import WeightSnapshotManager
from train_shim import *
import train_shim


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax over the last axis by default."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class Mode(Enum):
    SL = 1
    RL = 2


class LossTracker:
    MAX_LOSS = float("inf")

    def __init__(self):
        self.n = 0
        self.min_losses = defaultdict(lambda: self.MAX_LOSS)
        self.ema_losses = defaultdict(lambda: 0)
        self.avg_losses = defaultdict(lambda: 0)

    def update_losses(
        self,
        result: TrainStepResult,
    ):

        loss = float(result.total_loss)
        if math.isnan(loss) or math.isinf(loss):
            return
        policy_loss = float(result.policy_loss)
        policy_aux_dist_loss = float(result.policy_aux_dist_loss)
        policy_aux_scalar_loss = float(result.policy_aux_scalar_loss)
        outcome_loss = float(result.outcome_loss)
        score_pdf_loss = float(result.score_pdf_loss)
        score_cdf_loss = float(result.score_cdf_loss)
        own_loss = float(result.own_loss)
        q6_loss = float(result.q6_loss)
        q16_loss = float(result.q16_loss)
        q50_loss = float(result.q50_loss)
        q_err_loss = float(result.q_err_loss) if result.q_err_loss is not None else 0.0
        q_score_loss = (
            float(result.q_score_loss) if result.q_score_loss is not None else 0.0
        )
        q_score_err_loss = (
            float(result.q_score_err_loss)
            if result.q_score_err_loss is not None
            else 0.0
        )
        pi_soft_loss = (
            float(result.pi_soft_loss) if result.pi_soft_loss is not None else 0.0
        )
        pi_optimistic_loss = (
            float(result.pi_optimistic_loss)
            if result.pi_optimistic_loss is not None
            else 0.0
        )
        mcts_dist_loss = (
            float(result.mcts_dist_loss) if result.mcts_dist_loss is not None else 0.0
        )

        def update_mean_losses(r_m: float, r_c: float, losses: dict):
            losses["loss"] = losses["loss"] * r_m + loss * r_c
            losses["policy"] = losses["policy"] * r_m + policy_loss * r_c
            losses["policy_aux_dist"] = (
                losses["policy_aux_dist"] * r_m + policy_aux_dist_loss * r_c
            )
            losses["policy_aux_scalar"] = (
                losses["policy_aux_scalar"] * r_m + policy_aux_scalar_loss * r_c
            )
            losses["outcome"] = losses["outcome"] * r_m + outcome_loss * r_c
            losses["score_pdf"] = losses["score_pdf"] * r_m + score_pdf_loss * r_c
            losses["score_cdf"] = losses["score_cdf"] * r_m + score_cdf_loss * r_c
            losses["own"] = losses["own"] * r_m + own_loss * r_c
            losses["q6"] = losses["q6"] * r_m + q6_loss * r_c
            losses["q16"] = losses["q16"] * r_m + q16_loss * r_c
            losses["q50"] = losses["q50"] * r_m + q50_loss * r_c
            losses["q_err"] = losses["q_err"] * r_m + q_err_loss * r_c
            losses["q_score"] = losses["q_score"] * r_m + q_score_loss * r_c
            losses["q_score_err"] = losses["q_score_err"] * r_m + q_score_err_loss * r_c
            losses["pi_soft"] = losses["pi_soft"] * r_m + pi_soft_loss * r_c
            losses["pi_optimistic"] = (
                losses["pi_optimistic"] * r_m + pi_optimistic_loss * r_c
            )
            losses["mcts_dist"] = losses["mcts_dist"] * r_m + mcts_dist_loss * r_c

        self.min_losses["loss"] = min(self.min_losses["loss"], loss)
        self.min_losses["policy"] = min(self.min_losses["policy"], policy_loss)
        self.min_losses["policy_aux_dist"] = min(
            self.min_losses["policy_aux_dist"], policy_aux_dist_loss
        )
        self.min_losses["policy_aux_scalar"] = min(
            self.min_losses["policy_aux_scalar"], policy_aux_scalar_loss
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
        self.min_losses["mcts_dist"] = min(self.min_losses["mcts_dist"], mcts_dist_loss)

        r_m = 0.99 if self.n > 0 else 0.0
        r_c = 0.01 if self.n > 0 else 1.0
        update_mean_losses(r_m, r_c, self.ema_losses)

        r_m = self.n / (self.n + 1)
        r_c = 1 / (self.n + 1)
        update_mean_losses(r_m, r_c, self.avg_losses)
        self.n += 1


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
    train_ds,
    epochs: int,
    momentum: float,
    log_interval: int,
    mode: Mode,
    coeffs: LossCoeffs,
    optimizer: Optional[Any] = None,
    save_interval=1000,
    save_path="/tmp",
    tensorboard_log_dir="/tmp/logs",
    lr_schedule: Optional[Any] = None,
    is_gpu=True,
    batch_num=0,
    ss_manager: Optional[WeightSnapshotManager] = None,
    val_ds=None,
    num_val_batches: int = 10,
):
    """
    Training through single dataset.
    """
    assert is_gpu
    summary_writer = SummaryWriter(tensorboard_log_dir)

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
        w_mcts_dist=coeffs.w_mcts_dist,
    )

    assert optimizer is not None, (
        "train() requires a pre-built optimizer; use train_shim.make_optimizer "
        "to construct one (centralized factory)."
    )

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
                policy_aux_dist,
                has_pi_aux_dist,
                own,
                q6,
                q16,
                q50,
                q6_score,
                q16_score,
                q50_score,
                game_outcome,
                mcts_value_dist,
                has_mcts_value_dist,
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
                policy_aux_dist=policy_aux_dist,
                has_pi_aux_dist=has_pi_aux_dist,
                mcts_value_dist=mcts_value_dist,
                has_mcts_value_dist=has_mcts_value_dist,
            )

            result = train_step(
                input,
                input_global_state,
                targets,
                weights,
                model,
                optimizer,
            )

            if math.isnan(float(result.total_loss)) or math.isinf(
                float(result.total_loss)
            ):
                print(f"[batch {batch_num}] saw inf/nan gradients")

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
                    result,
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
                    pi_logits_np = train_shim.to_numpy(result.predictions.pi_logits[0])
                    pi_probs = _softmax(pi_logits_np)
                    policy_entropy = -float(np.sum(pi_probs * np.log(pi_probs + 1e-10)))
                    target_np = train_shim.to_numpy(targets.policy[0])
                    target_entropy = -float(
                        np.sum(target_np * np.log(target_np + 1e-10))
                    )
                    print(
                        f"Policy entropy - Predicted: {policy_entropy:.3f}, "
                        f"Target: {target_entropy:.3f} "
                        f"(max={math.log(362.0):.3f})"
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


def _vcategorical_side_by_side(
    target_counts: np.ndarray,  # (51,) uint32 raw counts
    pred_probs: np.ndarray,  # (51,) float probabilities
    granularity: int = 17,
    bar_width: int = 20,
) -> str:
    """Return a side-by-side ASCII histogram: target (left) vs predicted (right)."""
    NUM_BUCKETS = 51
    bucket_range = 2.0 / NUM_BUCKETS

    # Aggregate into display buckets
    target_display = np.zeros(granularity, dtype=np.float64)
    pred_display = np.zeros(granularity, dtype=np.float64)
    for i in range(NUM_BUCKETS):
        center = (i + 0.5) * bucket_range - 1.0
        j = int((center + 1.0) / 2.0 * granularity)
        j = max(0, min(granularity - 1, j))
        target_display[j] += target_counts[i]
        pred_display[j] += pred_probs[i]

    # Normalize target to probabilities
    total = target_display.sum()
    if total > 0:
        target_display /= total

    # Find union of occupied range
    lo, hi = granularity, -1
    for j in range(granularity):
        if target_display[j] > 0 or pred_display[j] > 0:
            if j < lo:
                lo = j
            if j > hi:
                hi = j
    if hi < 0:
        return "  (empty)\n"

    t_max = target_display[lo : hi + 1].max() or 1.0
    p_max = pred_display[lo : hi + 1].max() or 1.0
    bw = bucket_range * granularity / granularity  # display bucket width

    lines = []
    header = f"  {'Predicted':>{bar_width}}   val    {'Target':<{bar_width}}"
    lines.append(header)
    for j in range(lo, hi + 1):
        center = (j + 0.5) * (2.0 / granularity) - 1.0
        t_len = int(target_display[j] / t_max * bar_width)
        p_len = int(pred_display[j] / p_max * bar_width)
        p_bar = ("█" * p_len).rjust(bar_width)
        t_bar = ("█" * t_len).ljust(bar_width)
        lines.append(f"  {p_bar}  {center:+.2f}  {t_bar}")
    return "\n".join(lines) + "\n"


def log_train(
    batch_num: int,
    losses: LossTracker,
    result: TrainStepResult,
    summary_writer: SummaryWriter,
    mode: Mode,
):
    mode_str = "sl" if mode == Mode.SL else "rl"

    loss_avg = losses.ema_losses["loss"]
    policy_avg = losses.ema_losses["policy"]
    policy_aux_dist_avg = losses.ema_losses["policy_aux_dist"]
    policy_aux_scalar_avg = losses.ema_losses["policy_aux_scalar"]
    outcome_avg = losses.ema_losses["outcome"]
    score_pdf_avg = losses.ema_losses["score_pdf"]
    score_cdf_avg = losses.ema_losses["score_cdf"]
    own_avg = losses.ema_losses["own"]
    q6_avg = losses.ema_losses["q6"]
    q16_avg = losses.ema_losses["q16"]
    q50_avg = losses.ema_losses["q50"]
    q_err_avg = losses.ema_losses["q_err"]
    q_score_avg = losses.ema_losses["q_score"]
    q_score_err_avg = losses.ema_losses["q_score_err"]
    pi_soft_avg = losses.ema_losses["pi_soft"]
    pi_optimistic_avg = losses.ema_losses["pi_optimistic"]
    mcts_dist_avg = losses.ema_losses["mcts_dist"]

    loss_cur = float(result.total_loss)
    policy_cur = float(result.policy_loss)
    policy_aux_dist_cur = float(result.policy_aux_dist_loss)
    policy_aux_scalar_cur = float(result.policy_aux_scalar_loss)
    outcome_cur = float(result.outcome_loss)
    score_pdf_cur = float(result.score_pdf_loss)
    score_cdf_cur = float(result.score_cdf_loss)
    own_cur = float(result.own_loss)
    q6_cur = float(result.q6_loss)
    q16_cur = float(result.q16_loss)
    q50_cur = float(result.q50_loss)
    q_err_cur = float(result.q_err_loss) if result.q_err_loss is not None else 0.0
    q_score_cur = float(result.q_score_loss) if result.q_score_loss is not None else 0.0
    q_score_err_cur = (
        float(result.q_score_err_loss) if result.q_score_err_loss is not None else 0.0
    )
    pi_soft_cur = float(result.pi_soft_loss) if result.pi_soft_loss is not None else 0.0
    pi_optimistic_cur = (
        float(result.pi_optimistic_loss)
        if result.pi_optimistic_loss is not None
        else 0.0
    )
    mcts_dist_cur = (
        float(result.mcts_dist_loss) if result.mcts_dist_loss is not None else 0.0
    )
    grad_norm = float(result.grad_norm) if result.grad_norm is not None else 0.0

    print(
        f"[batch {batch_num}] {mode_str}: "
        f"loss = {loss_avg:.4f} ({loss_cur:.4f}), "
        f"policy = {policy_avg:.4f} ({policy_cur:.4f}), "
        f"policy_aux_dist = {policy_aux_dist_avg:.4f} ({policy_aux_dist_cur:.4f}), "
        f"policy_aux_scalar = {policy_aux_scalar_avg:.4f} ({policy_aux_scalar_cur:.4f}), "
        f"outcome = {outcome_avg:.4f} ({outcome_cur:.4f}), "
        f"score_pdf = {score_pdf_avg:.4f} ({score_pdf_cur:.4f}), "
        f"score_cdf = {score_cdf_avg:.4f} ({score_cdf_cur:.4f}), "
        f"own = {own_avg:.4f} ({own_cur:.4f}), "
        f"q6 = {q6_avg:.4f} ({q6_cur:.4f}), "
        f"q16 = {q16_avg:.4f} ({q16_cur:.4f}), "
        f"q50 = {q50_avg:.4f} ({q50_cur:.4f}), "
        f"q_err = {q_err_avg:.4f} ({q_err_cur:.4f}), "
        f"q_score = {q_score_avg:.4f} ({q_score_cur:.4f}), "
        f"q_score_err = {q_score_err_avg:.4f} ({q_score_err_cur:.4f}), "
        f"pi_soft = {pi_soft_avg:.4f} ({pi_soft_cur:.4f}), "
        f"pi_optimistic = {pi_optimistic_avg:.4f} ({pi_optimistic_cur:.4f}), "
        f"mcts_dist = {mcts_dist_avg:.4f} ({mcts_dist_cur:.4f}), "
        f"grad_norm = {grad_norm:.4f}"
    )

    summary_writer.scalar(f"{mode_str}/loss", loss_avg, batch_num)
    summary_writer.scalar(f"{mode_str}/policy", policy_avg, batch_num)
    summary_writer.scalar(f"{mode_str}/policy_aux_dist", policy_aux_dist_avg, batch_num)
    summary_writer.scalar(
        f"{mode_str}/policy_aux_scalar", policy_aux_scalar_avg, batch_num
    )
    summary_writer.scalar(f"{mode_str}/outcome", outcome_avg, batch_num)
    summary_writer.scalar(f"{mode_str}/score_pdf", score_pdf_avg, batch_num)
    summary_writer.scalar(f"{mode_str}/score_cdf", score_cdf_avg, batch_num)
    summary_writer.scalar(f"{mode_str}/own", own_avg, batch_num)
    summary_writer.scalar(f"{mode_str}/q6", q6_avg, batch_num)
    summary_writer.scalar(f"{mode_str}/q16", q16_avg, batch_num)
    summary_writer.scalar(f"{mode_str}/q50", q50_avg, batch_num)
    summary_writer.scalar(f"{mode_str}/mcts_dist", mcts_dist_avg, batch_num)


def log_board_position(
    batch_num: int,
    input_planes,
    input_global_state,
    predictions: ModelPredictions,
    targets: GroundTruth,
    model: P3achyGoModel,
):
    """Log a sample board position with model predictions."""
    # Take first example from batch
    planes = train_shim.to_numpy(input_planes[0])  # (19, 19, num_planes)
    global_state = train_shim.to_numpy(input_global_state[0])

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
    policy_pred = _softmax(train_shim.to_numpy(predictions.pi_logits[0]))
    outcome_pred = _softmax(train_shim.to_numpy(predictions.game_outcome[0]))
    score_pred = _softmax(train_shim.to_numpy(predictions.score_logits[0]))

    policy_target = train_shim.to_numpy(targets.policy[0])
    score_target = train_shim.to_numpy(targets.score[0])

    # Get top 5 policy moves
    top_indices = np.argsort(policy_pred)[-5:][::-1]
    top_indices_target = np.argsort(policy_target)[-5:][::-1]

    # short-term
    q6_pred, q6 = train_shim.to_numpy(predictions.q6_pred[0]), train_shim.to_numpy(
        targets.q6[0]
    )
    q16_pred, q16 = train_shim.to_numpy(predictions.q16_pred[0]), train_shim.to_numpy(
        targets.q16[0]
    )
    q50_pred, q50 = train_shim.to_numpy(predictions.q50_pred[0]), train_shim.to_numpy(
        targets.q50[0]
    )
    q6_err_pred, q6_err = train_shim.to_numpy(predictions.q6_err_pred[0]), np.square(
        q6 - q6_pred
    )
    q16_err_pred, q16_err = train_shim.to_numpy(predictions.q16_err_pred[0]), np.square(
        q16 - q16_pred
    )
    q50_err_pred, q50_err = train_shim.to_numpy(predictions.q50_err_pred[0]), np.square(
        q50 - q50_pred
    )

    # short-term score
    q6_score_pred, q6_score = (
        train_shim.to_numpy(predictions.q6_score_pred[0]),
        train_shim.to_numpy(targets.q6_score[0]),
    )
    q16_score_pred, q16_score = (
        train_shim.to_numpy(predictions.q16_score_pred[0]),
        train_shim.to_numpy(targets.q16_score[0]),
    )
    q50_score_pred, q50_score = (
        train_shim.to_numpy(predictions.q50_score_pred[0]),
        train_shim.to_numpy(targets.q50_score[0]),
    )
    q6_score_err_pred, q6_score_err = (
        train_shim.to_numpy(predictions.q6_score_err_pred[0]),
        np.square(q6_score - q6_score_pred),
    )
    q16_score_err_pred, q16_score_err = (
        train_shim.to_numpy(predictions.q16_score_err_pred[0]),
        np.square(q16_score - q16_score_pred),
    )
    q50_score_err_pred, q50_score_err = (
        train_shim.to_numpy(predictions.q50_score_err_pred[0]),
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
    own_pred = train_shim.to_numpy(predictions.own_pred[0]).squeeze()  # (19, 19)
    own_pred_abs = own_pred if to_play == BLACK else -own_pred
    own = train_shim.to_numpy(targets.own[0]).squeeze()
    own = own if to_play == BLACK else -own

    def ownership_char(x):
        bounds = [-1.0, -0.5, 0.0, 0.5, 1.0]
        chars = ["●", "◆", "⋅", "◇", "○"]  # positive=black=○, negative=white=●
        return chars[int(np.argmin([abs(x - b) for b in bounds]))]

    board_lines = GoBoard.to_string(board).split("\n")
    own_lines = []
    own_target_lines = []
    for i in range(19):
        own_lines.append(
            " ".join([ownership_char(own_pred_abs[i, j]) for j in range(19)])
        )
        own_target_lines.append(
            " ".join([ownership_char(own[i, j]) for j in range(19)])
        )
    col_gap = "    "
    own_lines.append(" ".join(list("ABCDEFGHJKLMNOPQRST")))
    own_target_lines.append(" ".join(list("ABCDEFGHJKLMNOPQRST")))
    print(
        f"  {'Board':<40}{col_gap}{'Own Target(○=black ●=white)':<37}{col_gap}Own Pred"
    )
    for bl, otl, ol in zip(board_lines, own_target_lines, own_lines):
        print(f"  {bl:<40}{col_gap}{otl:<37}{col_gap}{ol}")
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
    pi_soft_probs = _softmax(train_shim.to_numpy(predictions.pi_logits_soft[0]))
    pi_optimistic_probs = _softmax(
        train_shim.to_numpy(predictions.pi_logits_optimistic[0])
    )
    top_soft = np.argsort(pi_soft_probs)[-5:][::-1]
    top_optimistic = np.argsort(pi_optimistic_probs)[-5:][::-1]

    # Optimistic weight (mirrors v1_loss_terms computation)
    epsilon = 1e-6
    q6_p = train_shim.to_numpy(predictions.q6_pred[0])
    q16_p = train_shim.to_numpy(predictions.q16_pred[0])
    q50_p = train_shim.to_numpy(predictions.q50_pred[0])
    q6_err_p = train_shim.to_numpy(predictions.q6_err_pred[0])
    q16_err_p = train_shim.to_numpy(predictions.q16_err_pred[0])
    q50_err_p = train_shim.to_numpy(predictions.q50_err_pred[0])
    q6_score_p = train_shim.to_numpy(predictions.q6_score_pred[0])
    q16_score_p = train_shim.to_numpy(predictions.q16_score_pred[0])
    q50_score_p = train_shim.to_numpy(predictions.q50_score_pred[0])
    q6_score_err_p = train_shim.to_numpy(predictions.q6_score_err_pred[0])
    q16_score_err_p = train_shim.to_numpy(predictions.q16_score_err_pred[0])
    q50_score_err_p = train_shim.to_numpy(predictions.q50_score_err_pred[0])
    z6 = (train_shim.to_numpy(targets.q6[0]) - q6_p) / np.sqrt(q6_err_p + epsilon)
    z16 = (train_shim.to_numpy(targets.q16[0]) - q16_p) / np.sqrt(q16_err_p + epsilon)
    z50 = (train_shim.to_numpy(targets.q50[0]) - q50_p) / np.sqrt(q50_err_p + epsilon)
    z6_score = (train_shim.to_numpy(targets.q6_score[0]) - q6_score_p) / np.sqrt(
        q6_score_err_p + epsilon
    )
    z16_score = (train_shim.to_numpy(targets.q16_score[0]) - q16_score_p) / np.sqrt(
        q16_score_err_p + epsilon
    )
    z50_score = (train_shim.to_numpy(targets.q50_score[0]) - q50_score_p) / np.sqrt(
        q50_score_err_p + epsilon
    )

    def compute_opt_weight(z_wd, z6, z16, z50):
        return (z_wd * 3 * z6 + z_wd * 1.5 * z16 + z_wd * 0.75 * z50) / 3.0

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    z_wd = 4.0 / 7.0
    z_val = compute_opt_weight(z_wd, z6, z16, z50)
    z_score = compute_opt_weight(z_wd, z6_score, z16_score, z50_score)
    z_combined = (z_val + z_score * 0.5) / 1.5
    opt_weight = float(np.clip(sigmoid((z_combined - 1.0) * 3.0), 0.0, 1.0))

    # Top 5 moves table
    col_w = 20
    print(f"\nTop 5 Policy Moves:")
    print(
        f"  {'Predicted':<{col_w}}{'Target':<{col_w}}{'Soft Predicted':<{col_w}}{'Soft Target':<{col_w}}"
        f"Opt (w={opt_weight:.2f}, zv={z_val:.2f}, zs={z_score:.2f}, z={z_combined:.2f})"
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

    # Policy aux: top-4 predicted vs target (dist if available, else single move)
    pi_aux_logits = train_shim.to_numpy(predictions.pi_logits_aux[0])
    pi_aux_probs = np.exp(pi_aux_logits - pi_aux_logits.max())
    pi_aux_probs /= pi_aux_probs.sum()
    top_aux_pred = np.argsort(pi_aux_probs)[-4:][::-1]
    has_aux_dist = (
        bool(train_shim.to_numpy(targets.has_pi_aux_dist[0]))
        if targets.has_pi_aux_dist is not None
        else False
    )

    if has_aux_dist:
        aux_dist = train_shim.to_numpy(targets.policy_aux_dist[0]).astype(np.float32)
        aux_dist_sum = aux_dist.sum()
        if aux_dist_sum > 0:
            aux_dist /= aux_dist_sum
        top_aux_tgt = np.argsort(aux_dist)[-4:][::-1]
        tgt_label = "Target (dist)"
    else:
        target_aux_move = int(train_shim.to_numpy(targets.policy_aux[0]))
        top_aux_tgt = [target_aux_move] + [None] * 3
        aux_dist = None
        tgt_label = "Target"

    print(f"\nPolicy Aux (next player):")
    print(f"  {'Predicted':<20}  {tgt_label}")
    for i in range(4):
        pred_str = f"{move_to_coords(top_aux_pred[i]):>6} {pi_aux_probs[top_aux_pred[i]]:>6.1%}"
        if has_aux_dist and aux_dist is not None:
            tgt_str = (
                f"{move_to_coords(top_aux_tgt[i]):>6} {aux_dist[top_aux_tgt[i]]:>6.1%}"
            )
        elif top_aux_tgt[i] is not None:
            tgt_str = f"{move_to_coords(top_aux_tgt[i]):>6}"
        else:
            tgt_str = ""
        print(f"  {pred_str:<20}  {tgt_str}")

    # MCTS value distribution: side-by-side if available
    if (
        targets.has_mcts_value_dist is not None
        and bool(train_shim.to_numpy(targets.has_mcts_value_dist[0]))
        and predictions.mcts_dist_probs is not None
    ):
        target_counts = train_shim.to_numpy(targets.mcts_value_dist[0]).astype(
            np.float64
        )
        pred_probs = train_shim.to_numpy(predictions.mcts_dist_probs[0]).astype(
            np.float64
        )
        print(f"\nMCTS Value Distribution (pred | target):")
        print(_vcategorical_side_by_side(target_counts, pred_probs), end="")

    print(f"{'='*60}\n")


def save_model(
    model: P3achyGoModel,
    opt: Any,
    batch_num: int,
    save_path: str,
):
    """Mid-training checkpoint. Loses optimizer state — these are crash-resume
    artifacts within one chunk's training, not release artifacts. The end-of-
    generation save in rl_loop/train.py persists optimizer state separately."""
    del opt  # not bundled in the intermediate snapshot
    filename = f"model_{batch_num}{train_shim.MODEL_EXT}"
    filepath = Path(save_path) / filename
    train_shim.save_model(model, str(filepath))


def val(
    model: P3achyGoModel,
    val_ds,
    batch_num: int,
    num_batches=10,
    mode=Mode.SL,
    tensorboard_log_dir="/tmp/logs",
):
    """Validation on dataset."""
    summary_writer = SummaryWriter(tensorboard_log_dir)

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
        w_mcts_dist=coeffs.w_mcts_dist,
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
            policy_aux_dist,
            has_pi_aux_dist,
            own,
            q6,
            q16,
            q50,
            q6_score,
            q16_score,
            q50_score,
            game_outcome,
            mcts_value_dist,
            has_mcts_value_dist,
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
            policy_aux_dist=policy_aux_dist,
            has_pi_aux_dist=has_pi_aux_dist,
            mcts_value_dist=mcts_value_dist,
            has_mcts_value_dist=has_mcts_value_dist,
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
            predicted_moves = np.argmax(
                train_shim.to_numpy(predictions.pi_logits), axis=1
            )
            actual_moves = np.argmax(train_shim.to_numpy(policy), axis=1)
            correct_moves = int((predicted_moves == actual_moves).sum())

            predicted_outcomes = (
                np.argmax(train_shim.to_numpy(predictions.game_outcome), axis=1) == 1
            )
            actual_outcomes = train_shim.to_numpy(score) >= 0  # Actual win
            correct_outcomes = int((predicted_outcomes == actual_outcomes).sum())
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
    input_planes,
    input_global_state,
    predictions: ModelPredictions,
    targets: GroundTruth,
    model: P3achyGoModel,
):
    loss_avg = losses.avg_losses["loss"]
    policy_avg = losses.avg_losses["policy"]
    policy_aux_dist_avg = losses.avg_losses["policy_aux_dist"]
    policy_aux_scalar_avg = losses.avg_losses["policy_aux_scalar"]
    outcome_avg = losses.avg_losses["outcome"]
    score_pdf_avg = losses.avg_losses["score_pdf"]
    own_avg = losses.avg_losses["own"]
    q6_avg = losses.avg_losses["q6"]
    q16_avg = losses.avg_losses["q16"]
    q50_avg = losses.avg_losses["q50"]
    mcts_dist_avg = losses.avg_losses["mcts_dist"]

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
        f"policy_aux_dist = {policy_aux_dist_avg:.4f}, "
        f"policy_aux_scalar = {policy_aux_scalar_avg:.4f}, "
        f"outcome = {outcome_avg:.4f}, "
        f"score_pdf = {score_pdf_avg:.4f}, "
        f"own = {own_avg:.4f}, "
        f"q6 = {q6_avg:.4f}, "
        f"q16 = {q16_avg:.4f}, "
        f"q50 = {q50_avg:.4f}, "
        f"mcts_dist = {mcts_dist_avg:.4f}, "
        f"move_acc = {move_acc:.4f}, "
        f"outcome_acc = {outcome_acc:.4f}"
    )

    # Log a sample board position
    log_board_position(
        batch_num, input_planes, input_global_state, predictions, targets, model
    )
