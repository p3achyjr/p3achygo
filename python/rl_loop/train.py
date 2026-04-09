from __future__ import annotations

import math
import tensorflow as tf
import keras
import transforms
import train
import rl_loop.model_utils as model_utils

from absl import logging
from constants import *
from lr_schedule import ConstantLRSchedule, CyclicLRSchedule
from model import P3achyGoModel
from rl_loop.config import RunConfig
from weight_snapshot import WeightSnapshotManager
from loss_coeffs import LossCoeffs
from optimizer import ConvMuon

EPOCHS_PER_GEN = 1
MOMENTUM = 0.9
SWA_MOMENTUM = 0.75


def get_ss_timestamps(num_batches):
    TARGET_INTERVAL = 1000
    if num_batches < 1500:
        return []
    num_snapshots = (num_batches - 501) // TARGET_INTERVAL
    interval = int(num_batches / (num_snapshots + 1))
    return [(i + 1) * interval for i in range(num_snapshots)]


def get_lr(config: RunConfig, model_gen: int) -> float:
    lr_scale = 0.1 + 0.9 * min(1.0, model_gen / config.lr_growth_window)

    lr = config.lr
    next_gen, next_lr = None, None
    for gen, gen_lr in config.lr_schedule or []:
        if gen > model_gen:
            next_gen, next_lr = gen, gen_lr
            break
        lr = gen_lr

    window = config.lr_transition_window
    if window > 0 and next_gen is not None and (next_gen - model_gen) <= window:
        t = 0.5 * (1.0 - math.cos(math.pi * (1.0 - (next_gen - model_gen) / window)))
        lr = lr + t * (next_lr - lr)

    return lr_scale * lr


def train_one_gen(
    live_model: P3achyGoModel,
    last_swa_model: P3achyGoModel,
    optimizer: keras.optimizers.Optimizer,
    model_gen: int,
    chunk_path: str,
    val_ds: tf.data.TFRecordDataset,
    config: RunConfig,
    log_interval=100,
    is_gpu=True,
    batch_num=0,
    chunk_size=None,
):
    """
    Trains through dataset held at `chunk_path`.
    """

    def find_num_batches(ds: tf.data.TFRecordDataset) -> int:
        n = 0
        for _ in ds.batch(config.batch_size):
            n += 1

        return n

    def get_ss_timestamps(num_batches: int) -> list[int]:
        TARGET_INTERVAL = 1000
        if num_batches < 1500:
            return []
        num_snapshots = (num_batches - 501) // TARGET_INTERVAL
        interval = int(num_batches / (num_snapshots + 1))
        return [(i + 1) * interval for i in range(num_snapshots)]

    batch_size = config.batch_size
    lr_schedule = ConstantLRSchedule(get_lr(config, model_gen))
    # lr_schedule = CyclicLRSchedule(config.min_lr * lr_scale,
    #                                config.max_lr * lr_scale, num_batches)

    # logging.info(f'Running initial validation...')
    # train.val(model, mode=train.Mode.RL, val_ds=val_ds, val_batch_num=-1)

    ds = tf.data.TFRecordDataset(chunk_path, compression_type="ZLIB")
    num_batches = find_num_batches(ds)

    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Learning Rate Schedule: {lr_schedule.info()}")

    ds = ds.map(transforms.expand, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    if not optimizer:
        if config.optimizer == "muon":
            optimizer = ConvMuon(
                learning_rate=lr_schedule,
                exclude_layers=[r".*policy_head\/.*", r".*value_head\/.*"],
                adam_weight_decay=config.adam_wd,
                weight_decay=config.muon_wd,
                scale_weight_decay_by_rms=config.scale_weight_decay_by_rms,
                wd_lr_exponent=config.wd_lr_exponent,
                wd_lr_max=config.wd_lr_max,
            )
        else:
            optimizer = keras.optimizers.SGD(
                learning_rate=lr_schedule,
                momentum=MOMENTUM,
                global_clipnorm=20.0,
                nesterov=True,
            )
        if is_gpu:
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    inner_optimizer = getattr(optimizer, "inner_optimizer", optimizer)
    if isinstance(inner_optimizer, ConvMuon):
        logging.info(
            f"Using ConvMuon Optimizer"
            f"\n  Weight Decay={inner_optimizer.weight_decay}"
            f"\n  AdamW Weight Decay={inner_optimizer.adam_weight_decay}"
            f"\n  Momentum={inner_optimizer.momentum}"
            f"\n  Scale WD by RMS={inner_optimizer.scale_weight_decay_by_rms}"
            f"\n  WD Auto Scale={config.wd_auto_scale}"
            f"\n  WD LR Exponent={inner_optimizer.wd_lr_exponent}"
            f"\n  WD LR Max={inner_optimizer.wd_lr_max}"
        )
    else:
        logging.info(
            f"Using ConvMuon SGD"
            f"\n  Momentum={inner_optimizer.momentum}"
            f"\n  ClipNorm={inner_optimizer.global_clipnorm}"
        )

    ss_manager = WeightSnapshotManager(get_ss_timestamps(num_batches))
    last_swa_weights = last_swa_model.get_weights()
    loss_coeffs = LossCoeffs.RLCoeffs()
    if model_gen <= 100:
        # downweight some terms as at this point it is just noise.
        loss_coeffs.w_q_score *= 0.5
        loss_coeffs.w_q_score_err *= 0.5
        loss_coeffs.w_pi_soft *= 0.25
    if isinstance(inner_optimizer, ConvMuon):
        # observed severe overfitting for outcome head.
        loss_coeffs.w_outcome *= 0.4

    logging.info(f"Loss Coefficients: {loss_coeffs}")
    old_batch_num = batch_num
    batch_num, optimizer = train.train(
        live_model,
        ds,
        EPOCHS_PER_GEN,
        MOMENTUM,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        log_interval=log_interval,
        mode=train.Mode.RL,
        coeffs=loss_coeffs,
        save_interval=None,
        save_path=None,
        is_gpu=is_gpu,
        batch_num=batch_num,
        ss_manager=ss_manager,
    )

    print(
        f"SWA Momentum: {SWA_MOMENTUM}, "
        + f"Num Batches: {num_batches}, "
        + f"Num Batches in Chunk: {batch_num - old_batch_num}, "
        + f"Num Snapshots: {len(ss_manager.snapshots)}, "
        + f"Snapshots: {get_ss_timestamps(num_batches)}"
    )
    # num_batches_in_chunk = batch_num - old_batch_num
    # new_weights = model_utils.avg_weights(last_swa_weights, model.get_weights(),
    #                                       num_batches_in_chunk)
    new_weights = model_utils.swa_avg_weights(
        [last_swa_weights] + ss_manager.snapshots + [live_model.get_weights()],
        swa_momentum=SWA_MOMENTUM,
    )
    print(f"Last Model: {model_gen}, Next Model: {model_gen + 1}")
    print(f"Last SWA Model Weights: {last_swa_weights[0][0][0][0][0:8]}")
    print(f"Live Weights: {live_model.get_weights()[0][0][0][0][0:8]}")
    print(f"New SWA Model Weights: {new_weights[0][0][0][0][0:8]}")
    swa_model = keras.models.clone_model(live_model)
    swa_model.set_weights(new_weights)
    model_utils.recompute_bn_statistics(swa_model, ds)
    # model.set_weights(new_weights)
    logging.info(f"Running validation for live model...")
    train.val(live_model, mode=train.Mode.RL, val_ds=val_ds, batch_num=model_gen + 1)
    logging.info(f"Running validation for new model...")
    train.val(swa_model, mode=train.Mode.RL, val_ds=val_ds, batch_num=model_gen + 1)

    return batch_num, live_model, swa_model, optimizer
