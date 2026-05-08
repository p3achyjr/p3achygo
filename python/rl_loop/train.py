from __future__ import annotations

import math
import multiprocessing
from typing import Any
import train
import backend_shim
from dataset import ChunkDataset

from absl import logging
from constants import *
from lr_schedule import ConstantLRSchedule
from backend_shim import P3achyGoModel
from rl_loop.config import RunConfig
from weight_snapshot import WeightSnapshotManager
from loss_coeffs import LossCoeffs

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
    warmup_t = min(
        1.0, max(0.0, (model_gen - config.start_gen) / config.lr_growth_window)
    )
    lr_scale = 0.1 + 0.9 * warmup_t

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

    bs_scale = (
        math.sqrt(config.batch_size / 256)
        if config.optimizer == "muon"
        else config.batch_size / 256
    )
    return lr_scale * lr * bs_scale


def train_one_gen(
    live_model: P3achyGoModel,
    last_swa_model: P3achyGoModel,
    optimizer: Any,
    model_gen: int,
    chunk_path: str,
    val_ds: ChunkDataset,
    config: RunConfig,
    log_interval=100,
    is_gpu=True,
    batch_num=0,
    chunk_size=None,
):
    """
    Trains through dataset held at `chunk_path`.
    """

    def get_ss_timestamps(num_batches: int) -> list[int]:
        TARGET_INTERVAL = 1000
        if num_batches < 1500:
            return []
        num_snapshots = (num_batches - 501) // TARGET_INTERVAL
        interval = int(num_batches / (num_snapshots + 1))
        return [(i + 1) * interval for i in range(num_snapshots)]

    batch_size = config.batch_size
    lr_schedule = ConstantLRSchedule(get_lr(config, model_gen))

    # `num_workers=0` would block the GPU on single-threaded tfrecord
    # decompression — 60% of step time on b8c128tfmr / RTX 5090. Empirically
    # 2 workers wins ~80%, 8 saturates; cap at hw_threads/2 to avoid
    # over-subscribing small boxes. See profile_workers_sweep.py for the
    # measurement.
    num_workers = min(8, max(1, multiprocessing.cpu_count() // 2))
    ds = ChunkDataset(chunk_path, batch_size, num_workers=num_workers)
    num_batches = len(ds)

    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Learning Rate Schedule: {lr_schedule.info()}")

    # `optimizer` is opaque — None on cold start, an in-memory optimizer
    # on in-process hot-reload, or a state_dict on cross-process resume.
    # `make_optimizer` sniffs the type internally.
    optimizer = backend_shim.make_optimizer(
        live_model, config, lr_schedule, is_gpu, loaded_state=optimizer
    )
    inner_optimizer = getattr(optimizer, "inner_optimizer", optimizer)
    if isinstance(inner_optimizer, backend_shim.ConvMuon):
        logging.info(
            f"Using ConvMuon Optimizer"
            f"\n  Learning Rate={inner_optimizer.learning_rate}"
            f"\n  Weight Decay={inner_optimizer.weight_decay}"
            f"\n  AdamW Weight Decay={inner_optimizer.adam_weight_decay}"
            f"\n  AdamW LR Ratio={inner_optimizer.adam_lr_ratio}"
            f"\n  Momentum={inner_optimizer.momentum}"
            f"\n  WD Auto Scale={config.wd_auto_scale}"
            f"\n  WD LR Exponent={inner_optimizer.wd_lr_exponent}"
            f"\n  WD LR Max={inner_optimizer.wd_lr_max}"
            f"\n  Global ClipNorm={inner_optimizer.global_clipnorm}"
        )
    else:
        logging.info(
            f"Using SGD Optimizer"
            f"\n  Learning Rate={inner_optimizer.learning_rate}"
            f"\n  Momentum={getattr(inner_optimizer, 'momentum', None)}"
            f"\n  Global ClipNorm={inner_optimizer.global_clipnorm}"
        )

    ss_manager = WeightSnapshotManager(get_ss_timestamps(num_batches))
    last_swa_weights = backend_shim.get_weights(last_swa_model)
    loss_coeffs = LossCoeffs.RLCoeffs()
    if model_gen <= 100:
        # downweight some terms as at this point it is just noise.
        loss_coeffs.w_q_score *= 0.5
        loss_coeffs.w_q_score_err *= 0.5
        loss_coeffs.w_pi_soft *= 0.25
    if isinstance(inner_optimizer, backend_shim.ConvMuon):
        # observed severe overfitting for outcome head.
        loss_coeffs.w_outcome *= 0.4

    logging.info(f"Loss Coefficients: {loss_coeffs}")
    old_batch_num = batch_num
    _train_kwargs = dict(
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
    batch_num, optimizer = train.train(
        live_model, ds, EPOCHS_PER_GEN, MOMENTUM, **_train_kwargs
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
    new_weights = backend_shim.swa_avg_weights(
        [last_swa_weights]
        + ss_manager.snapshots
        + [backend_shim.get_weights(live_model)],
        swa_momentum=SWA_MOMENTUM,
    )
    print(f"Last Model: {model_gen}, Next Model: {model_gen + 1}")
    swa_model = backend_shim.clone_model(live_model)
    backend_shim.set_weights(swa_model, new_weights)
    backend_shim.recompute_bn_statistics(swa_model, ds)
    # model.set_weights(new_weights)
    logging.info(f"Running validation for live model...")
    train.val(live_model, mode=train.Mode.RL, val_ds=val_ds, batch_num=model_gen + 1)
    logging.info(f"Running validation for new model...")
    train.val(swa_model, mode=train.Mode.RL, val_ds=val_ds, batch_num=model_gen + 1)

    return batch_num, live_model, swa_model, optimizer
