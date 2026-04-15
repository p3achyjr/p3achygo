"""
Distributed training utilities using tf.distribute.MirroredStrategy.

NOT IN USE. The active training path is the serial path in train.py.
This module is kept here for reference if multi-GPU training is revisited.

Design notes:
- train_step() must NOT call optimizer.apply_gradients() when used with
  strategy.run(). MirroredStrategy intercepts apply_gradients() called
  inside replica context and performs NCCL all-reduce automatically.
- Each replica receives batch_size/num_replicas samples
  (via strategy.experimental_distribute_dataset). The loss is divided by
  num_replicas before gradient computation so that after the all-reduce SUM,
  the effective update equals the gradient of the mean loss over the full batch.
- The optimizer MUST be constructed AND built inside strategy.scope() so that
  optimizer._distribution_strategy = MirroredStrategy and slot variables are
  MirroredVariables. Constructing outside scope captures default_strategy,
  causing UpdateContext(MirroredVariable) which breaks assign_sub with:
  "TypeError: tuple indices must be integers or slices, not MirroredVariable"
- A module-level tf.function cache (_STEP_FN_CACHE) ensures the compiled graph
  survives across train() calls (e.g. across generations in train_loop.py),
  avoiding per-generation XLA retracing.
"""

from __future__ import annotations

import tensorflow as tf
import keras
from typing import Optional

from model import P3achyGoModel, GroundTruth, LossWeights
from train import TrainStepResult


def _reduce_result(
    strategy: tf.distribute.Strategy,
    result: TrainStepResult,
) -> TrainStepResult:
    """Reduce per-replica TrainStepResult fields to scalars for logging.
    With a single-device strategy this is a no-op.

    result: TrainStepResult whose tensor fields are PerReplica objects
            (i.e., the direct output of strategy.run).
    """

    def r(x):
        if x is None:
            return None
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None)

    # Take first replica's predictions (used only for logging)
    predictions = strategy.experimental_local_results(result.predictions)[0]

    return TrainStepResult(
        predictions=predictions,
        total_loss=r(result.total_loss),
        policy_loss=r(result.policy_loss),
        policy_aux_dist_loss=r(result.policy_aux_dist_loss),
        policy_aux_scalar_loss=r(result.policy_aux_scalar_loss),
        outcome_loss=r(result.outcome_loss),
        q6_loss=r(result.q6_loss),
        q16_loss=r(result.q16_loss),
        q50_loss=r(result.q50_loss),
        score_pdf_loss=r(result.score_pdf_loss),
        score_cdf_loss=r(result.score_cdf_loss),
        own_loss=r(result.own_loss),
        q_err_loss=r(result.q_err_loss),
        q_score_loss=r(result.q_score_loss),
        q_score_err_loss=r(result.q_score_err_loss),
        pi_soft_loss=r(result.pi_soft_loss),
        pi_optimistic_loss=r(result.pi_optimistic_loss),
        mcts_dist_loss=r(result.mcts_dist_loss),
        grad_norm=r(result.grad_norm),
    )


# Keyed on (id(model), id(optimizer), id(strategy), num_replicas, weights).
# A new tf.function is only created when the model/optimizer/strategy identity
# or the loss weights change.
_STEP_FN_CACHE: dict = {}


def make_distributed_train_step(
    model: P3achyGoModel,
    optimizer,
    strategy: tf.distribute.Strategy,
    weights: LossWeights,
    num_replicas: int,
):
    """
    Returns a compiled @tf.function that runs one distributed training step.

    The returned function accepts (input, input_global_state, targets) where
    each argument may be a plain tensor (single GPU) or a PerReplica tensor
    (multi-GPU, from experimental_distribute_dataset), and returns a reduced
    TrainStepResult suitable for logging.

    Usage in train():
        if num_replicas > 1:
            train_ds = strategy.experimental_distribute_dataset(train_ds)
        distributed_step = make_distributed_train_step(
            model, optimizer, strategy, weights, num_replicas
        )
        ...
        result = distributed_step(input, input_global_state, targets)
    """
    _step_key = (id(model), id(optimizer), id(strategy), num_replicas, weights)
    if _step_key not in _STEP_FN_CACHE:

        def _train_step_replica(input, input_global_state, targets):
            """Single-replica train step for use inside strategy.run().

            Divides loss by num_replicas before gradient computation.
            MirroredStrategy all-reduces gradients across replicas when
            apply_gradients is called in replica context.
            """
            import tensorflow as tf
            from train import train_step as _serial_train_step

            # Run forward + backward + apply in replica context.
            # The loss / num_replicas scaling ensures that after all-reduce SUM,
            # effective gradient = mean gradient over the full global batch.
            return _serial_train_step(
                input,
                input_global_state,
                targets,
                weights,
                model,
                optimizer,
                num_replicas=num_replicas,
            )

        @tf.function
        def _distributed_step(input, input_global_state, targets):
            per_replica_result = strategy.run(
                _train_step_replica,
                args=(input, input_global_state, targets),
            )
            return _reduce_result(strategy, per_replica_result)

        _STEP_FN_CACHE[_step_key] = _distributed_step

    return _STEP_FN_CACHE[_step_key]
