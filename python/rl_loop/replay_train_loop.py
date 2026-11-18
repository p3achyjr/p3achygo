"""
Trainer for a sliding-window replay of an existing run's self-play data
(KataGo-style: a fixed number of training samples per new data row).

A separate builder (`replay_build_goldens.py`) writes temporary goldens
`chunk_NNNN.tfrecord.zz` with a manifest `chunk_NNNN.json` into --golden_dir, in
index order. This loop trains one epoch on each, in order, and differs from
`train_loop.py` in three ways:

  - LR follows KataGo's sample-keyed schedule (`lr_scale_auto` long-run decay and
    the 2M-sample warmup from katago/python/train.py), rescaled so the first
    stage is --base_lr.
  - SWA is decoupled from golden boundaries: the live weights are folded into
    the SWA every --swa_period_batches batches with momentum SWA_MOMENTUM.
  - An SWA model (with recomputed BN statistics) and a live checkpoint are saved
    when a manifest marks the last golden of a replay step, named by the step's
    window-end generation.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import time

import torch

import backend_shim
import train
from dataset import ChunkDataset
from backend_shim import (
    configure_gpu,
    load_with_optimizer,
    save_model,
    summary,
    compile_for_training,
    LIVE_MODEL_NAME,
    MODEL_EXT,
)
import rl_loop.model_utils as model_utils
import rl_loop.config
from rl_loop.train import SWA_MOMENTUM, EPOCHS_PER_GEN, MOMENTUM
from loss_coeffs import LossCoeffs
from weight_snapshot import WeightSnapshotManager

from absl import app, flags, logging
from pathlib import Path

FLAGS = flags.FLAGS

flags.DEFINE_string("run_id", "", "ID corresponding to the current run.")
flags.DEFINE_string("models_dir", "", "Directory containing model checkpoints.")
flags.DEFINE_string("golden_dir", "", "Directory the builder writes goldens to.")
flags.DEFINE_string("val_ds_path", "", "Path to validation dataset.")
flags.DEFINE_string("state_path", "", "JSON file with trainer progress.")
flags.DEFINE_float("base_lr", 1e-4, "LR of the first schedule stage.")
flags.DEFINE_integer("swa_period_batches", 1000, "Batches between SWA updates.")

POLL_INTERVAL_S = 30

# katago/python/train.py: (global_step_samples upper bound, factor).
_KATAGO_WARMUP = [
    (250_000, 1 / 20),
    (500_000, 1 / 14),
    (750_000, 1 / 10),
    (1_000_000, 1 / 7),
    (1_250_000, 1 / 5),
    (1_500_000, 1 / 3),
    (1_750_000, 1 / 2),
    (2_000_000, 1 / 1.4),
]
# katago/python/train.py lr_scale_auto (long runs): (global_step_samples upper bound, factor).
_KATAGO_LR_SCALE_AUTO = [
    (550e6, 8.0),
    (1.4e9, 6.0),
    (2.5e9, 4.0),
    (3.4e9, 3.0),
    (4.2e9, 2.0),
    (4.9e9, 1.5),
    (5.5e9, 1.0),
    (6.4e9, 0.7),
]
_KATAGO_LR_SCALE_AUTO_FINAL = 0.5


def katago_lr(samples: float, base_lr: float) -> float:
    """KataGo lr_scale_auto x warmup, rescaled so the first stage is `base_lr`."""
    factor = _KATAGO_LR_SCALE_AUTO_FINAL
    for bound, f in _KATAGO_LR_SCALE_AUTO:
        if samples < bound:
            factor = f
            break
    warmup = 1.0
    for bound, w in _KATAGO_WARMUP:
        if samples < bound:
            warmup = w
            break
    return base_lr * (factor / _KATAGO_LR_SCALE_AUTO[0][1]) * warmup


class _SampleKeyedLR:
    """LR callable for the optimizer (called with its own step counter).
    `batch_num0` is the run's batch counter when optimizer step `opt_step0` runs."""

    def __init__(self, base_lr, batch_size, batch_num0, opt_step0):
        self.base_lr, self.batch_size = base_lr, batch_size
        self.batch_num0, self.opt_step0 = batch_num0, opt_step0

    def __call__(self, opt_step=None):
        step = self.opt_step0 if opt_step is None else opt_step
        batch = self.batch_num0 + (step - self.opt_step0)
        return katago_lr(batch * self.batch_size, self.base_lr)

    def info(self) -> str:
        return f"KataGo auto, base {self.base_lr}, now {self():.3g}"


def _wait_for_golden(golden_dir: Path, idx: int):
    """Returns (chunk_path, manifest) or None when the builder is done."""
    chunk = golden_dir / f"chunk_{idx:04d}.tfrecord.zz"
    manifest = golden_dir / f"chunk_{idx:04d}.json"
    while not manifest.exists():
        if (golden_dir / "DONE").exists() and not manifest.exists():
            return None
        time.sleep(POLL_INTERVAL_S)
    return str(chunk), json.loads(manifest.read_text())


def _train_loop():
    config = rl_loop.config.parse(FLAGS.run_id)
    models_dir, golden_dir = Path(FLAGS.models_dir), Path(FLAGS.golden_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(FLAGS.state_path)
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"next_golden": 0, "batch_num": 0}
    )
    live_model_path = str(models_dir / LIVE_MODEL_NAME)
    swa_weights_path = models_dir / "swa_weights.pt"

    val_workers = min(8, max(1, multiprocessing.cpu_count() // 2))
    val_ds = ChunkDataset(FLAGS.val_ds_path, config.batch_size, num_workers=val_workers)

    if not Path(live_model_path).exists():
        logging.info(f"Creating new model from config '{config.model_config}'.")
        live_model = model_utils.new_model(
            "p3achygo", config.model_config, config.optimizer
        )
        save_model(live_model, live_model_path)
    live_model, optimizer = load_with_optimizer(live_model_path)
    swa_weights = (
        torch.load(swa_weights_path)
        if swa_weights_path.exists()
        else backend_shim.get_weights(live_model)
    )
    summary(live_model)
    live_model = compile_for_training(live_model)

    loss_coeffs = LossCoeffs.RLCoeffs()
    loss_coeffs.w_outcome *= 0.4  # as rl_loop/train.py does for ConvMuon
    num_workers = min(8, max(1, multiprocessing.cpu_count() // 2))
    period = FLAGS.swa_period_batches

    while True:
        got = _wait_for_golden(golden_dir, state["next_golden"])
        if got is None:
            logging.info("Builder is done and all goldens are trained.")
            return
        chunk_path, manifest = got
        batch_num = state["batch_num"]
        ds = ChunkDataset(chunk_path, config.batch_size, num_workers=num_workers)
        num_batches = len(ds)

        inner = getattr(optimizer, "inner_optimizer", optimizer)
        lr_schedule = _SampleKeyedLR(
            FLAGS.base_lr,
            config.batch_size,
            batch_num,
            getattr(inner, "_global_step", 0),
        )
        optimizer = backend_shim.make_optimizer(
            live_model, config, lr_schedule, True, loaded_state=optimizer
        )
        # Snapshot at local steps where the run's batch counter hits a multiple
        # of the SWA period.
        first = period - (batch_num % period)
        ss_manager = WeightSnapshotManager(list(range(first, num_batches + 1, period)))
        logging.info(
            f"Golden {state['next_golden']} ({manifest}): {num_batches} batches, "
            f"batch_num {batch_num}, LR {lr_schedule.info()}, "
            f"SWA snapshots at {ss_manager.ss_steps}"
        )
        batch_num, optimizer = train.train(
            live_model,
            ds,
            EPOCHS_PER_GEN,
            MOMENTUM,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            log_interval=100,
            mode=train.Mode.RL,
            coeffs=loss_coeffs,
            save_interval=None,
            save_path=None,
            is_gpu=True,
            batch_num=batch_num,
            ss_manager=ss_manager,
        )
        if ss_manager.snapshots:
            swa_weights = backend_shim.swa_avg_weights(
                [swa_weights] + ss_manager.snapshots, swa_momentum=SWA_MOMENTUM
            )

        if manifest.get("last_of_step"):
            end_gen = manifest["end_gen"]
            swa_model = backend_shim.clone_model(live_model)
            backend_shim.set_weights(swa_model, swa_weights)
            backend_shim.recompute_bn_statistics(swa_model, ds)
            logging.info(f"Validation, live model (window end gen {end_gen})...")
            train.val(live_model, mode=train.Mode.RL, val_ds=val_ds, batch_num=end_gen)
            logging.info(f"Validation, SWA model (window end gen {end_gen})...")
            train.val(swa_model, mode=train.Mode.RL, val_ds=val_ds, batch_num=end_gen)
            model_utils.save(swa_model, str(models_dir), end_gen)
            live_ckpt_dir = models_dir / "_live"
            live_ckpt_dir.mkdir(exist_ok=True)
            save_model(
                live_model,
                str(live_ckpt_dir / f"live_{end_gen:04d}{MODEL_EXT}"),
                optimizer=optimizer,
            )

        # Restart point: live + optimizer, SWA weights, then the state file last.
        save_model(live_model, live_model_path, optimizer=optimizer)
        torch.save(swa_weights, swa_weights_path)
        state = {"next_golden": state["next_golden"] + 1, "batch_num": batch_num}
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)
        # Marker for the builder: this golden has been consumed.
        (golden_dir / f"chunk_{state['next_golden'] - 1:04d}.trained").touch()


def main(_):
    for name in ("run_id", "models_dir", "golden_dir", "val_ds_path", "state_path"):
        if not getattr(FLAGS, name):
            logging.error(f"No --{name} specified.")
            return
    configure_gpu()
    _train_loop()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    app.run(main)
