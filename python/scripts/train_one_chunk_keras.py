"""Keras/TF parallel of train_one_chunk_torch.py — same loss subset for fair benchmark.

Loss subset (matches the torch script):
  policy CE + game-outcome CE + score-distribution CE + ownership BCE
  + Q-value MSE (q6/q16/q50)

Usage:
    KERAS_BACKEND=tensorflow PYTHONPATH=python python python/scripts/train_one_chunk_keras.py \\
        --model_path ~/p3achygo-data/v4-models/b9-legacy/model_0311.keras \\
        --chunk ~/p3achygo-data/v4-models/example-chunks/chunk_0390.tfrecord.zz \\
        --batch_size 128 --lr 1e-4 --max_batches 30
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
import keras

import sys

sys.path.insert(0, "python")
from model import P3achyGoModel
from optimizer import ConvMuon  # noqa: F401  (registers serializable)
from lr_schedule import ConstantLRSchedule  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--chunk", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_batches", type=int, default=30)
    p.add_argument(
        "--mixed_precision",
        default="mixed_float16",
        help='"mixed_float16" | "mixed_bfloat16" | "float32"',
    )
    args = p.parse_args()

    keras.mixed_precision.set_global_policy(args.mixed_precision)
    print(f"keras mixed_precision policy: {keras.mixed_precision.global_policy().name}")

    print(f"loading model from {args.model_path}")
    model = keras.models.load_model(
        os.path.expanduser(args.model_path),
        custom_objects=P3achyGoModel.custom_objects(),
        compile=False,
    )
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    print(f"model params: {n_params:,}")

    optimizer = ConvMuon(
        learning_rate=args.lr,
        weight_decay=0.02,
        adam_weight_decay=0.02,
        adam_lr_ratio=1.0,
        momentum=0.95,
        nesterov=True,
        rms_rate=0.2,
        ns_steps=5,
    )
    if args.mixed_precision == "mixed_float16":
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Load dataset (TF backend; reuse the existing keras ChunkDataset).
    from dataset import ChunkDataset

    chunk_path = os.path.expanduser(args.chunk)
    ds = ChunkDataset(chunk_path, args.batch_size)
    print(f"chunk: {chunk_path}")

    p_jit = os.environ.get("TF_JIT_COMPILE", "1") == "1"
    print(f"tf.function jit_compile={p_jit}")

    @tf.function(jit_compile=p_jit)
    def train_step(
        input_planes,
        input_global,
        policy,
        score_one_hot,
        own,
        q6,
        q16,
        q50,
        game_outcome,
    ):
        with tf.GradientTape() as tape:
            out = model(input_planes, input_global, training=True)
            (
                pi_logits,
                _pi_probs,
                outcome_logits,
                _outcome_probs,
                own_pred,
                score_logits,
                _score_probs,
                _gamma,
                _pi_aux,
                q6_pred,
                q16_pred,
                q50_pred,
                _q6e,
                _q16e,
                _q50e,
                _q6sc,
                _q16sc,
                _q50sc,
                _q6sce,
                _q16sce,
                _q50sce,
                _pi_soft,
                _pi_opt,
                _mcts_logits,
                _mcts_probs,
            ) = out

            # subset losses (match torch script)
            loss_pi = -tf.reduce_mean(
                tf.reduce_sum(
                    tf.cast(policy, tf.float32)
                    * tf.nn.log_softmax(tf.cast(pi_logits, tf.float32), axis=-1),
                    axis=-1,
                )
            )
            loss_outcome = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    tf.cast(game_outcome, tf.float32),
                    tf.cast(outcome_logits, tf.float32),
                    from_logits=True,
                )
            )
            loss_score = -tf.reduce_mean(
                tf.reduce_sum(
                    tf.cast(score_one_hot, tf.float32)
                    * tf.nn.log_softmax(tf.cast(score_logits, tf.float32), axis=-1),
                    axis=-1,
                )
            )
            own_t = (tf.cast(own, tf.float32) + 1.0) * 0.5
            own_pred_flat = tf.reshape(own_pred, (tf.shape(own_pred)[0], -1))
            own_t_flat = tf.reshape(own_t, (tf.shape(own_t)[0], -1))
            loss_own = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=own_t_flat, logits=tf.cast(own_pred_flat, tf.float32)
                )
            )

            def _squeeze_if_2d(t):
                return t if t.shape.rank == 1 else tf.reshape(t, [-1])

            loss_q = (
                tf.reduce_mean(
                    tf.square(_squeeze_if_2d(q6_pred) - tf.cast(q6, tf.float32))
                )
                + tf.reduce_mean(
                    tf.square(_squeeze_if_2d(q16_pred) - tf.cast(q16, tf.float32))
                )
                + tf.reduce_mean(
                    tf.square(_squeeze_if_2d(q50_pred) - tf.cast(q50, tf.float32))
                )
            )
            loss = (
                loss_pi
                + loss_outcome
                + 0.1 * loss_score
                + 0.1 * loss_own
                + 0.1 * loss_q
            )

            scaled = (
                optimizer.scale_loss(loss) if hasattr(optimizer, "scale_loss") else loss
            )

        grads = tape.gradient(scaled, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, loss_pi, loss_outcome, loss_score, loss_own, loss_q

    losses, times = [], []
    for i, batch in enumerate(ds):
        if i >= args.max_batches:
            break
        (
            input_planes,
            input_global,
            color,
            komi,
            score,
            score_one_hot,
            policy,
            _policy_aux,
            _policy_aux_dist,
            _has_pi_aux,
            own,
            q6,
            q16,
            q50,
            _q6_score,
            _q16_score,
            _q50_score,
            game_outcome,
            _mcts_dist,
            _has_mcts,
        ) = batch

        # Sync GPU before timing
        t0 = time.perf_counter()
        loss, lpi, lout, lsc, lown, lq = train_step(
            tf.cast(input_planes, tf.float32),
            tf.cast(input_global, tf.float32),
            policy,
            score_one_hot,
            own,
            q6,
            q16,
            q50,
            game_outcome,
        )
        # Force materialization (XLA is async)
        _ = float(loss.numpy())
        t1 = time.perf_counter()

        losses.append(_)
        times.append(t1 - t0)

        if i % 10 == 0 or i < 5:
            print(
                f"step {i:4d}  loss={_:.4f}  pi={float(lpi):.4f}  "
                f"out={float(lout):.4f}  score={float(lsc):.4f}  "
                f"own={float(lown):.4f}  q={float(lq):.4f}  ms={1000*(t1-t0):.1f}"
            )

    if not times:
        print("no batches")
        return
    warmup = min(10, max(1, len(times) // 10))
    steady = times[warmup:]
    print()
    print(f"summary: total={len(times)} batches, warmup={warmup}")
    print(f"  loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}")
    print(
        f"  ms/batch (all):    mean={1000*sum(times)/len(times):.1f} "
        f"min={1000*min(times):.1f} max={1000*max(times):.1f}"
    )
    if steady:
        print(
            f"  ms/batch (steady): mean={1000*sum(steady)/len(steady):.1f} "
            f"min={1000*min(steady):.1f} max={1000*max(steady):.1f}"
        )


if __name__ == "__main__":
    main()
