"""Quick single-chunk training script for smoke-testing a model checkpoint.

Usage:
    python scripts/train_one_chunk.py \
        --model_path /p3achygo-data/v4-models/b12wd0.02/model_0200_new.keras \
        --chunk /tmp/chunks/gen000_b000_g512_n21829_t3213748_test.tfrecord.zz
"""

import numpy as np
import keras

from absl import app, flags, logging
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import *
from model import P3achyGoModel, GroundTruth, LossWeights
from loss_coeffs import LossCoeffs
from optimizer import (
    ConvMuon,
)  # noqa: F401 — needed for model deserialization
from lr_schedule import ConstantLRSchedule  # noqa: F401
from dataset import ChunkDataset
from train import train_step, LossTracker, log_train, log_board_position, Mode


# A trivial no-op writer satisfies log_train's interface without TF.
class _NoOpWriter:
    def scalar(self, name, value, step):
        pass

    def close(self):
        pass


FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "", "Path to .keras model checkpoint.")
flags.DEFINE_string("chunk", "", "Path to .tfrecord.zz chunk.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("shuffle_buffer", 4096, "Shuffle buffer size.")
flags.DEFINE_float("lr", 1e-4, "Learning rate.")
flags.DEFINE_integer("log_interval", 10, "Print loss every N batches.")
flags.DEFINE_integer(
    "max_batches", 0, "Stop after this many batches (0 = run to chunk end)."
)
flags.DEFINE_boolean(
    "torch_compile", False, "Wrap the model with torch.compile (torch backend only)."
)


def make_dataset(chunk_path: str, batch_size: int, shuffle_buffer: int):
    # `shuffle_buffer` is unused after the migration to ChunkDataset, which
    # reads front-to-back. Kept in the signature for now to avoid touching
    # callers; remove when the flag is dropped.
    del shuffle_buffer
    return ChunkDataset(chunk_path, batch_size)


def main(_):
    if not FLAGS.model_path or not FLAGS.chunk:
        logging.error("--model_path and --chunk are required.")
        return

    logging.info(f"Loading model from {FLAGS.model_path} ...")
    model = keras.models.load_model(
        FLAGS.model_path,
        custom_objects=P3achyGoModel.custom_objects(),
        compile=False,
    )

    if FLAGS.torch_compile:
        import torch

        logging.info("Wrapping model with torch.compile ...")
        model = torch.compile(model)

    coeffs = LossCoeffs.RLCoeffs()
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

    optimizer = keras.mixed_precision.LossScaleOptimizer(
        ConvMuon(
            learning_rate=FLAGS.lr,
            weight_decay=0.02,
            adam_weight_decay=0.02,
            adam_lr_ratio=1.0,
            momentum=0.95,
            nesterov=True,
            global_clipnorm=20.0,
            exclude_layers=[r".*policy_head\/.*", r".*value_head\/.*"],
        )
    )

    ds = make_dataset(FLAGS.chunk, FLAGS.batch_size, FLAGS.shuffle_buffer)
    summary_writer = _NoOpWriter()
    loss_tracker = LossTracker()

    logging.info("Starting training ...")
    for batch_num, batch_data in enumerate(ds):
        if FLAGS.max_batches and batch_num >= FLAGS.max_batches:
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

        result = train_step(
            input, input_global_state, targets, weights, model, optimizer
        )
        loss_tracker.update_losses(result)

        if batch_num % FLAGS.log_interval == 0:
            log_train(batch_num, loss_tracker, result, summary_writer, Mode.RL)

    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
