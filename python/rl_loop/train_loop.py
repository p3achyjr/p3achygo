"""
Long-running training process that loops over available chunks without restarting.

Compared to train_one_gen.py (which is spawned as a subprocess per generation),
this script loads the models and traces the TF graph once and then loops
indefinitely — avoiding the grappler re-tracing cost on every generation.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import time

from concurrent.futures import ThreadPoolExecutor

import gcs_utils as gcs
from dataset import ChunkDataset
from backend_shim import (
    configure_gpu,
    load_model,
    load_with_optimizer,
    save_model,
    summary,
    compile_for_training,
    LIVE_MODEL_NAME,
    MODEL_EXT,
)
import rl_loop.model_utils as model_utils
import rl_loop.train
import rl_loop.config

from absl import app, flags, logging
from pathlib import Path
from rl_loop.constants import SELFPLAY_BATCH_SIZE

FLAGS = flags.FLAGS

flags.DEFINE_string("run_id", "", "ID corresponding to the current run.")
flags.DEFINE_string("models_dir", "", "Directory containing model checkpoints.")
flags.DEFINE_string("chunk_dir", "", "Directory containing training chunks.")
flags.DEFINE_string("val_ds_path", "", "Path to validation dataset.")
flags.DEFINE_string("batch_num_path", "", "File storing batch counter.")
flags.DEFINE_boolean("save_trt", False, "Whether to save TRT model after each gen.")
flags.DEFINE_string("trt_convert_path", "", "Path to TRT convert binary.")
flags.DEFINE_integer("max_gens", 0, "Number of generations to train (0 = run forever).")
flags.DEFINE_string(
    "source_run_id", "", "Run ID to fetch golden chunks from (defaults to run_id)."
)
flags.DEFINE_boolean("batch_renorm", False, "Batch Renorm on the trunk BNs.")
flags.DEFINE_float("brn_rmax", 3.0, "Batch Renorm r clip target.")
flags.DEFINE_float("brn_dmax", 5.0, "Batch Renorm d clip target.")
flags.DEFINE_float(
    "brn_ramp_samples",
    0.0,
    "Ramp rmax/dmax from 1/0 toward their targets over this many training "
    "samples, as 1 - exp(-samples / brn_ramp_samples) (KataGo). 0 = no ramp.",
)
flags.DEFINE_float(
    "trunk_bn_eps", 0.0, "If > 0, trunk BN eps (function-preserving). 0 = keep."
)
flags.DEFINE_string(
    "adaptive_wd_baseline_path",
    "",
    "If set, KataGo adaptive WD on the trunk weight matrices: decay x "
    "2^(2 tanh(3 ln(norm / baseline))). JSON {'norm': x}; written from the "
    "starting live model if missing.",
)
flags.DEFINE_float(
    "gamma_center1_wd_factor",
    0.0,
    "If > 0, trunk BN gammas decay toward 1 at this factor x the muon WD "
    "(KataGo gamma_weight_decay_center_1; KataGo uses 0.25 with Muon).",
)
flags.DEFINE_boolean(
    "trunk_beta_noreg", False, "No weight decay on trunk BN betas (KataGo noreg)."
)
RCLONE_REMOTE = "r2"
R2_BUCKET = "p3achygo"
POLL_INTERVAL_S = 60


def _trunk_roots(model) -> list:
    """Trunk modules: the blocks, plus fixscaleonenorm's trunk-final BN."""
    tfb = getattr(model, "trunk_final_bn", None)
    return [model.blocks] + ([tfb] if tfb is not None else [])


def _trunk_bns(model) -> list:
    import torch

    return [
        m
        for r in _trunk_roots(model)
        for m in r.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    ]


def _trunk_gammas_betas(model) -> tuple:
    """Trunk norm gammas and betas: BatchNorm weight/bias, FixScaleNorm
    gamma/beta, FixScaleBatchNorm beta (it has no gamma)."""
    import torch
    from backend_torch.model_layers_common import FixScaleBatchNorm, FixScaleNorm

    gammas, betas = [], []
    for r in _trunk_roots(model):
        for m in r.modules():
            if isinstance(m, FixScaleNorm):
                gammas += [m.gamma] if m.gamma is not None else []
                betas.append(m.beta)
            elif isinstance(m, FixScaleBatchNorm):
                betas.append(m.beta)
            elif isinstance(m, torch.nn.BatchNorm2d):
                gammas.append(m.weight)
                betas.append(m.bias)
    return gammas, betas


def _trunk_weight_params(model) -> list:
    """Trunk weight matrices (ndim >= 2): the set whose norm adaptive WD holds."""
    return [p for p in model.blocks.parameters() if p.requires_grad and p.ndim >= 2]


def _norm_control_on() -> bool:
    return bool(
        FLAGS.adaptive_wd_baseline_path
        or FLAGS.gamma_center1_wd_factor > 0
        or FLAGS.trunk_beta_noreg
    )


def _configure_norm_control(live_model, optimizer, config, model_gen, batch_num):
    """Build the optimizer up front (so its norm control is set before the
    first step) and configure adaptive WD / center-1 gamma WD / beta noreg."""
    from backend_shim import make_optimizer

    optimizer = make_optimizer(
        live_model,
        config,
        rl_loop.train.get_lr(config, model_gen),
        True,
        loaded_state=optimizer,
    )
    inner = getattr(optimizer, "inner_optimizer", optimizer)
    weights, baseline = _trunk_weight_params(live_model), None
    if FLAGS.adaptive_wd_baseline_path:
        bp = Path(FLAGS.adaptive_wd_baseline_path)
        if not bp.exists():
            bp.write_text(
                json.dumps(
                    {
                        "norm": inner.params_norm(weights),
                        "gen": model_gen,
                        "batch_num": batch_num,
                    }
                )
            )
        baseline = float(json.loads(bp.read_text())["norm"])
    all_gammas, all_betas = _trunk_gammas_betas(live_model)
    gammas = all_gammas if FLAGS.gamma_center1_wd_factor > 0 else []
    betas = all_betas if FLAGS.trunk_beta_noreg else []
    inner.configure_norm_control(
        adaptive_params=weights if baseline else (),
        baseline_norm=baseline,
        center1_params=gammas,
        center1_wd_factor=FLAGS.gamma_center1_wd_factor,
        noreg_params=betas,
    )
    logging.info(
        f"Norm control: adaptive WD on {len(weights) if baseline else 0} trunk "
        f"weights (baseline {baseline}, norm now {inner.params_norm(weights):.4f}); "
        f"center-1 WD x{FLAGS.gamma_center1_wd_factor} on {len(gammas)} trunk "
        f"gammas; noreg on {len(betas)} trunk betas"
    )
    return optimizer


def _log_norms(live_model, optimizer, gen):
    import torch

    inner = getattr(optimizer, "inner_optimizer", optimizer)
    gammas, betas = _trunk_gammas_betas(live_model)
    g = torch.cat([p.detach().flatten() for p in gammas])
    b = torch.cat([p.detach().flatten() for p in betas])
    logging.info(
        f"Norm control end of gen {gen}: trunk weight norm "
        f"{inner.params_norm(_trunk_weight_params(live_model)):.4f} "
        f"(ratio {inner.last_norm_ratio}, adaptive scale {inner._adaptive_scale:.4f}); "
        f"trunk gamma median {g.median():.4f} mean {g.mean():.4f}; "
        f"beta |median| {b.abs().median():.4f}; lr {inner.learning_rate:.4g}"
    )


def _get_starting_gen(models_dir: str, start_gen: int) -> tuple[str, int]:
    model_paths = [
        f for f in Path(models_dir).glob("**/*") if gcs.MODEL_RE.fullmatch(f.name)
    ]
    model_paths = sorted(
        model_paths, key=lambda f: int(gcs.MODEL_RE.fullmatch(f.name).group(1))
    )
    if not model_paths:
        return str(Path(models_dir, gcs.MODEL_FORMAT.format(start_gen))), start_gen
    latest = model_paths[-1]
    return str(latest), int(gcs.MODEL_RE.fullmatch(latest.name).group(1))


def _chunk_published(run_id: str, chunk_name: str) -> bool:
    """True if the chunk object exists in remote storage (R2, via rclone)."""
    remote_dir = f"{RCLONE_REMOTE}:{R2_BUCKET}/{run_id}/goldens/"
    result = subprocess.run(
        ["rclone", "lsf", remote_dir, "--include", chunk_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return chunk_name in result.stdout.split()


def _fetch_chunk(chunk_dir: str, run_id: str, gen: int) -> tuple[str, bool]:
    """Download chunk_{gen:04d}.tfrecord.zz from R2 and return (local path, was_downloaded).

    Blocks (polling every POLL_INTERVAL_S) until the chunk is published, so the
    loop can prefetch a generation ahead without crashing on a not-yet-produced
    chunk.
    """
    chunk_name = gcs.GOLDEN_CHUNK_FORMAT.format(gen)
    local_path = Path(chunk_dir, chunk_name)
    remote_uri = f"{RCLONE_REMOTE}:{R2_BUCKET}/{run_id}/goldens/{chunk_name}"
    if local_path.exists():
        logging.info(f"Chunk already exists at {local_path}, skipping fetch")
        return str(local_path), False
    while not _chunk_published(run_id, chunk_name):
        logging.info(
            f"Chunk {remote_uri} not published yet; waiting {POLL_INTERVAL_S}s"
        )
        time.sleep(POLL_INTERVAL_S)
    logging.info(f"Fetching {remote_uri} -> {local_path}")
    subprocess.run(["rclone", "copyto", remote_uri, str(local_path)], check=True)
    return str(local_path), True


def _train_loop():
    config = rl_loop.config.parse(FLAGS.run_id)

    Path(FLAGS.models_dir).mkdir(parents=True, exist_ok=True)
    Path(FLAGS.chunk_dir).mkdir(parents=True, exist_ok=True)

    swa_model_path, model_gen = _get_starting_gen(FLAGS.models_dir, config.start_gen)
    live_model_path = str(Path(FLAGS.models_dir, LIVE_MODEL_NAME))

    logging.info(f"Starting from generation {model_gen}")
    logging.info(f"Live model: {live_model_path}")
    logging.info(f"SWA model:  {swa_model_path}")

    val_workers = min(8, max(1, multiprocessing.cpu_count() // 2))
    val_ds = ChunkDataset(FLAGS.val_ds_path, config.batch_size, num_workers=val_workers)

    if not Path(FLAGS.batch_num_path).exists():
        with open(FLAGS.batch_num_path, "w") as f:
            f.write("0")
    with open(FLAGS.batch_num_path, "r") as f:
        batch_num = int(f.read())

    if not Path(live_model_path).exists():
        logging.info(
            f"No live model found at {live_model_path}. Creating new model from config '{config.model_config}'."
        )
        live_model = model_utils.new_model(
            "p3achygo", config.model_config, config.optimizer
        )
        if FLAGS.trunk_bn_eps > 0:
            from backend_torch.model_layers_common import set_bn_eps

            n = sum(set_bn_eps(r, FLAGS.trunk_bn_eps) for r in _trunk_roots(live_model))
            live_model._config["trunk_bn_eps"] = FLAGS.trunk_bn_eps
            logging.info(f"Trunk BN eps -> {FLAGS.trunk_bn_eps} on {n} BNs")
        save_model(live_model, live_model_path)

    if not Path(swa_model_path).exists():
        swa_model_path = live_model_path

    live_model, optimizer = load_with_optimizer(live_model_path)
    if optimizer is None:
        logging.info(
            "No optimizer state found in live model. "
            "This should only happen for model_0000."
        )
    swa_model = load_model(swa_model_path)
    if FLAGS.trunk_bn_eps > 0:
        bad = [m.eps for m in _trunk_bns(live_model) if m.eps != FLAGS.trunk_bn_eps]
        assert not bad, f"live model trunk BN eps {set(bad)} != {FLAGS.trunk_bn_eps}"
    brn_ramp_batches = 0
    if FLAGS.batch_renorm:
        from backend_torch.model_layers_common import enable_batch_renorm

        brn_ramp_batches = int(round(FLAGS.brn_ramp_samples / config.batch_size))
        n = sum(
            enable_batch_renorm(
                r,
                FLAGS.brn_rmax,
                FLAGS.brn_dmax,
                brn_ramp_batches,
                ramp_exp=True,
            )
            for r in _trunk_roots(live_model)
        )
        logging.info(
            f"Batch Renorm on {n} trunk BNs: rmax {FLAGS.brn_rmax} dmax "
            f"{FLAGS.brn_dmax}, exp ramp {brn_ramp_batches} batches"
        )

    summary(live_model)

    # Apply backend-specific training-time transforms (fp16 + channels_last
    # + torch.compile reduce-overhead for torch; no-op for TF).
    live_model = compile_for_training(live_model)
    if _norm_control_on():
        optimizer = _configure_norm_control(
            live_model, optimizer, config, model_gen, batch_num
        )

    max_gens = FLAGS.max_gens
    gens_trained = 0
    source_run_id = FLAGS.source_run_id or FLAGS.run_id

    # Prefetch the next chunk in a background thread so its download overlaps
    # with training the current generation. `rclone` runs as a subprocess, so a
    # single worker (which releases the GIL during the download) is enough.
    fetch_executor = ThreadPoolExecutor(max_workers=1)
    pending_fetch = fetch_executor.submit(
        _fetch_chunk, FLAGS.chunk_dir, source_run_id, model_gen + 1
    )

    while max_gens == 0 or gens_trained < max_gens:
        config = rl_loop.config.parse(FLAGS.run_id)
        next_gen = model_gen + 1
        # Block only if the prefetch hasn't finished yet.
        chunk_path, chunk_downloaded = pending_fetch.result()
        # Kick off the following generation's download before we start training.
        if max_gens == 0 or gens_trained + 1 < max_gens:
            pending_fetch = fetch_executor.submit(
                _fetch_chunk, FLAGS.chunk_dir, source_run_id, next_gen + 1
            )
        logging.info(f"Training generation {next_gen} on {chunk_path}")
        if FLAGS.batch_renorm:
            from backend_torch.model_layers_common import set_brn_step

            for r in _trunk_roots(live_model):
                set_brn_step(r, batch_num)

        batch_num, live_model, swa_model, optimizer = rl_loop.train.train_one_gen(
            live_model,
            swa_model,
            optimizer,
            model_gen,
            chunk_path,
            val_ds,
            config=config,
            is_gpu=True,
            batch_num=batch_num,
        )

        if _norm_control_on():
            _log_norms(live_model, optimizer, next_gen)

        if chunk_downloaded:
            logging.info(f"Deleting local chunk {chunk_path}")
            Path(chunk_path).unlink(missing_ok=True)

        # Save live model checkpoint, bundling optimizer state.
        save_model(live_model, live_model_path, optimizer=optimizer)

        # Save restart checkpoint.
        live_ckpt_dir = Path(FLAGS.models_dir, "_live")
        live_ckpt_dir.mkdir(exist_ok=True)
        save_model(
            live_model,
            str(live_ckpt_dir / f"live_{next_gen:04d}{MODEL_EXT}"),
            optimizer=optimizer,
        )

        # Save SWA model for selfplay.
        if FLAGS.save_trt:
            model_utils.save_onnx_trt(
                swa_model,
                FLAGS.val_ds_path,
                FLAGS.models_dir,
                next_gen,
                batch_size=SELFPLAY_BATCH_SIZE,
                trt_convert_path=FLAGS.trt_convert_path,
            )
        else:
            model_utils.save(swa_model, FLAGS.models_dir, next_gen)

        with open(FLAGS.batch_num_path, "w") as f:
            f.write(str(batch_num))

        model_gen = next_gen
        gens_trained += 1
        logging.info(
            f"Generation {next_gen} complete. Total trained this session: {gens_trained}."
        )


def main(_):
    for flag_name, val in [
        ("run_id", FLAGS.run_id),
        ("models_dir", FLAGS.models_dir),
        ("chunk_dir", FLAGS.chunk_dir),
        ("val_ds_path", FLAGS.val_ds_path),
        ("batch_num_path", FLAGS.batch_num_path),
    ]:
        if not val:
            logging.error(f"No --{flag_name} specified.")
            return
    if FLAGS.save_trt and not FLAGS.trt_convert_path:
        logging.error("No --trt_convert_path specified.")
        return

    configure_gpu()

    _train_loop()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    app.run(main)
