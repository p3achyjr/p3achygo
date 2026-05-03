"""
Distributed training using MirroredStrategy.

train_step_distributed has the same signature as train_step but calls
strategy.run() internally so gradient computation runs in replica context.
strategy is obtained via tf.distribute.get_strategy(), which returns the active
MirroredStrategy when called inside strategy.scope().

Loss is divided by num_replicas before the GradientTape so that after
MirroredStrategy's all-reduce SUM the effective update equals the gradient of
the mean loss over the full global batch.
"""

from __future__ import annotations

import math
import tensorflow as tf
import keras
from typing import Optional

from model import P3achyGoModel, ModelPredictions, GroundTruth, LossWeights
from train import (
    Mode,
    TrainStepResult,
    LossTracker,
    log_train,
    log_board_position,
)
from loss_coeffs import LossCoeffs
from weight_snapshot import WeightSnapshotManager


@tf.function
def train_step_distributed(
    input: tf.Tensor,
    input_global_state: tf.Tensor,
    targets: GroundTruth,
    weights: LossWeights,
    model: P3achyGoModel,
    optimizer,
) -> TrainStepResult:
    """Distributed train step; same signature as train_step in train.py."""
    strategy = tf.distribute.get_strategy()
    num_replicas = strategy.num_replicas_in_sync

    def replica_fn(inp, gs, tgt):
        with tf.GradientTape() as tape:
            model_outputs = model(inp, gs, training=True)
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
            ) = model.compute_losses(predictions, tgt, weights)

            reg_loss = tf.math.add_n(model.losses)
            total_loss = loss + reg_loss
            scaled_loss = total_loss / num_replicas

        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        grad_norm = tf.linalg.global_norm(gradients)

        return (
            total_loss,
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
            grad_norm,
        )

    per_replica = strategy.run(replica_fn, args=(input, input_global_state, targets))

    def mean(x):
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None)

    (
        total_loss,
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
        grad_norm,
    ) = [mean(x) for x in per_replica]

    return TrainStepResult(
        predictions=None,
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


def _get_predictions(model, inp_r0, gs_r0):
    """Run inference on replica-0 batch outside strategy.run().

    Valid without a GradientTape — MirroredStrategy only blocks .handle access
    during gradient computation, not plain reads.
    """
    model_outputs = model(inp_r0, gs_r0, training=False)
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
    return ModelPredictions(
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


def train(
    model: P3achyGoModel,
    train_ds: tf.data.Dataset,
    epochs: int,
    momentum: float,
    log_interval: int,
    mode: Mode,
    coeffs: LossCoeffs,
    strategy: tf.distribute.Strategy,
    optimizer: Optional[keras.optimizers.Optimizer] = None,
    save_interval=None,
    save_path=None,
    tensorboard_log_dir="/tmp/logs",
    lr_schedule=None,
    is_gpu=True,
    batch_num=0,
    ss_manager: Optional[WeightSnapshotManager] = None,
    val_ds=None,
    num_val_batches=10,
):
    """Distributed training loop; returns (batch_num, optimizer)."""
    assert is_gpu
    summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)

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

    if not optimizer:
        optimizer = keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=momentum,
            global_clipnorm=20.0,
            nesterov=True,
        )
        if is_gpu:
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    if not optimizer.built:
        optimizer.build(model.trainable_variables)

    dist_ds = strategy.experimental_distribute_dataset(train_ds)
    losses_train = LossTracker()
    local_batch_num = 0

    for _ in range(epochs):
        for batch_data in dist_ds:
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

            result = train_step_distributed(
                input,
                input_global_state,
                targets,
                weights,
                model,
                optimizer,
            )

            if math.isnan(result.total_loss.numpy()) or math.isinf(
                result.total_loss.numpy()
            ):
                print(f"[batch {batch_num}] saw inf/nan gradients")

            losses_train.update_losses(result)

            local_batch_num += 1
            batch_num += 1

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

                if local_batch_num % (log_interval * 5) == 0:
                    # Extract replica-0 tensors for logging.
                    # experimental_local_results is Python-level PerReplica indexing
                    # — no GPU overhead.
                    inp_r0 = strategy.experimental_local_results(input)[0]
                    gs_r0 = strategy.experimental_local_results(input_global_state)[0]
                    tgt_r0 = tf.nest.map_structure(
                        lambda x: strategy.experimental_local_results(x)[0],
                        targets,
                    )
                    predictions = _get_predictions(model, inp_r0, gs_r0)
                    log_board_position(
                        batch_num, inp_r0, gs_r0, predictions, tgt_r0, model
                    )

                    pi_probs = keras.activations.softmax(predictions.pi_logits[0])
                    policy_entropy = -tf.reduce_sum(
                        pi_probs * tf.math.log(pi_probs + 1e-10)
                    )
                    target_entropy = -tf.reduce_sum(
                        tgt_r0.policy[0] * tf.math.log(tgt_r0.policy[0] + 1e-10)
                    )
                    print(
                        f"Policy entropy - Predicted: {policy_entropy.numpy():.3f}, "
                        f"Target: {target_entropy.numpy():.3f} "
                        f"(max={tf.math.log(362.0).numpy():.3f})"
                    )

    return batch_num, optimizer
