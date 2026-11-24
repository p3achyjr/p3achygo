"""
Trains a model for one generation and writes it back to disk.
"""

from __future__ import annotations

import sys
import torch
import gcs_utils as gcs
from dataset import ChunkDataset
from backend_shim import (
    configure_gpu,
    load_model,
    load_with_optimizer,
    save_model,
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
from typing import Tuple
from lr_schedule import ConstantLRSchedule  # noqa: F401 — registers schedule

BATCH_SIZE = 256

FLAGS = flags.FLAGS

flags.DEFINE_string("run_id", "", "ID corresponding to the current run.")
flags.DEFINE_string("models_dir", "", "Directory to find model.")
flags.DEFINE_integer("gen", -1, "Generation, or -1 for most recent")
flags.DEFINE_integer("next_gen", -1, "Next generation.")
flags.DEFINE_string("chunk_path", "", "Path to training chunk")
flags.DEFINE_integer(
    "chunk_size",
    None,
    "Size of chunk. Used for cyclic LR. Specify `None` to use linear LR.",
)
flags.DEFINE_string("val_ds_path", "", "Path to val ds")
flags.DEFINE_string("batch_num_path", "", "File storing batch counter.")
flags.DEFINE_boolean("save_trt", True, "Whether to save TRT converted model.")
flags.DEFINE_string("trt_convert_path", "", "Path to TRT convert binary.")
flags.DEFINE_boolean(
    "batch_renorm", False, "Train trunk BNs with Batch Renorm (torch only)."
)
flags.DEFINE_float("brn_rmax", 3.0, "Batch Renorm r clip.")
flags.DEFINE_float("brn_dmax", 5.0, "Batch Renorm d clip.")
flags.DEFINE_integer(
    "brn_ramp_batches", 0, "Ramp rmax/dmax from 1/0 over this many batches."
)
flags.DEFINE_float(
    "trunk_bn_eps",
    0.0,
    "If > 0, set every trunk BN eps to this value (function-preserving gamma "
    "and optimizer-moment rescale; saved in the checkpoint config). 0 = keep.",
)
flags.DEFINE_string(
    "adaptive_wd_baseline_path",
    "",
    "If set, KataGo-style adaptive WD on the trunk weight matrices: their decay "
    "is scaled by 2^(2 tanh(3 ln(norm / baseline))). JSON {'norm': x}; written "
    "from the starting live model if missing.",
)
flags.DEFINE_float(
    "gamma_pull_tau_steps",
    0.0,
    "If > 0, trunk BN gammas are pulled toward 1 with this time constant in "
    "steps (gamma -= (gamma - 1) / tau per step) instead of decaying toward 0.",
)


def _trunk_weight_params(model) -> list:
    """Trunk weight matrices (ndim >= 2): the set whose norm adaptive WD holds."""
    return [p for p in model.blocks.parameters() if p.requires_grad and p.ndim >= 2]


def _trunk_gammas(model) -> list:
    return [
        m.weight
        for m in model.blocks.modules()
        if isinstance(m, torch.nn.BatchNorm2d) and m.weight is not None
    ]


def get_model_path(models_dir: str, gen: int) -> Tuple[str, int]:
    if gen != -1:
        model_path = gcs.MODEL_FORMAT.format(gen)
        return str(Path(models_dir, model_path)), gen

    model_paths = Path(models_dir).glob("**/*")
    model_paths = [f for f in model_paths if gcs.MODEL_RE.fullmatch(f.name)]
    model_paths = sorted(
        model_paths, key=lambda f: gcs.MODEL_RE.fullmatch(f.name).group(1)
    )
    model_path = model_paths[-1]
    return str(model_path), gcs.MODEL_RE.fullmatch(model_path).group(1)


def main(_):
    if FLAGS.run_id == "":
        logging.error("No --run_id specified.")
        return
    if FLAGS.models_dir == "":
        logging.error("No --models_dir specified.")
        return
    if FLAGS.next_gen == -1:
        logging.error("No --next_gen specified.")
        return
    if FLAGS.chunk_path == "":
        logging.error("No --chunk_path specified.")
        return
    if FLAGS.val_ds_path == "":
        logging.error("No --val_ds_path specified.")
        return
    if FLAGS.batch_num_path == "":
        logging.error("No --batch_num_path specified.")
        return
    if FLAGS.save_trt and FLAGS.trt_convert_path == "":
        logging.error("No --trt_convert_path specified.")
        return

    configure_gpu()
    is_gpu = True

    config = rl_loop.config.parse(FLAGS.run_id)

    swa_model_path, _ = get_model_path(FLAGS.models_dir, FLAGS.gen)
    live_model_path = str(Path(FLAGS.models_dir, LIVE_MODEL_NAME))
    live_model, optimizer = load_with_optimizer(live_model_path)
    if optimizer is None:
        logging.info(
            "No optimizer state found in model. "
            "This should only happen for model_0000."
        )
    swa_model = load_model(swa_model_path)
    if FLAGS.batch_renorm:
        from backend_torch.model_layers_common import enable_batch_renorm

        n = enable_batch_renorm(
            live_model.blocks, FLAGS.brn_rmax, FLAGS.brn_dmax, FLAGS.brn_ramp_batches
        )
        logging.info(
            f"Batch Renorm on {n} trunk BNs: rmax={FLAGS.brn_rmax} "
            f"dmax={FLAGS.brn_dmax} ramp_batches={FLAGS.brn_ramp_batches}"
        )
    # Apply backend-specific training-time transforms (no-op on TF).
    live_model = compile_for_training(live_model)
    if FLAGS.trunk_bn_eps > 0 and any(
        m.eps != FLAGS.trunk_bn_eps
        for m in live_model.blocks.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    ):
        from backend_torch.model_layers_common import set_bn_eps
        from backend_shim import make_optimizer

        # Build the optimizer now (train_one_gen accepts the object) so the
        # gamma moments can be rescaled together with gamma.
        optimizer = make_optimizer(
            live_model,
            config,
            ConstantLRSchedule(rl_loop.train.get_lr(config, FLAGS.gen)),
            is_gpu,
            loaded_state=optimizer,
        )
        inner = getattr(optimizer, "inner_optimizer", optimizer)
        n_live = set_bn_eps(live_model.blocks, FLAGS.trunk_bn_eps, inner.state)
        n_swa = set_bn_eps(swa_model.blocks, FLAGS.trunk_bn_eps)
        live_model._config["trunk_bn_eps"] = FLAGS.trunk_bn_eps
        swa_model._config["trunk_bn_eps"] = FLAGS.trunk_bn_eps
        logging.info(
            f"Trunk BN eps -> {FLAGS.trunk_bn_eps}: {n_live} live / {n_swa} SWA BNs "
            "(gamma + optimizer moments rescaled)"
        )
    if FLAGS.adaptive_wd_baseline_path or FLAGS.gamma_pull_tau_steps > 0:
        import json
        from backend_shim import make_optimizer

        if optimizer is None or isinstance(optimizer, dict):
            optimizer = make_optimizer(
                live_model,
                config,
                ConstantLRSchedule(rl_loop.train.get_lr(config, FLAGS.gen)),
                is_gpu,
                loaded_state=optimizer,
            )
        inner = getattr(optimizer, "inner_optimizer", optimizer)
        weights, gammas, baseline = _trunk_weight_params(live_model), [], None
        if FLAGS.adaptive_wd_baseline_path:
            bp = Path(FLAGS.adaptive_wd_baseline_path)
            if not bp.exists():
                bp.write_text(
                    json.dumps({"norm": inner.params_norm(weights), "gen": FLAGS.gen})
                )
            baseline = float(json.loads(bp.read_text())["norm"])
        if FLAGS.gamma_pull_tau_steps > 0:
            gammas = _trunk_gammas(live_model)
        inner.configure_norm_control(
            adaptive_params=weights if baseline else (),
            baseline_norm=baseline,
            center1_params=gammas,
            center1_rate=1.0 / FLAGS.gamma_pull_tau_steps if gammas else 0.0,
        )
        g = torch.cat([x.detach().flatten() for x in gammas]) if gammas else None
        logging.info(
            f"Norm control: adaptive WD on {len(weights) if baseline else 0} trunk "
            f"weights, baseline {baseline}, norm now {inner.params_norm(weights):.4f}; "
            f"gamma pull on {len(gammas)} trunk BNs, tau {FLAGS.gamma_pull_tau_steps} steps"
            + (f", |gamma| median {g.abs().median():.4f}" if g is not None else "")
        )
    logging.info(f"Using Train Dataset: {FLAGS.chunk_path}")
    logging.info(f"Using Val Dataset: {FLAGS.val_ds_path}")
    logging.info(f"Using Model Checkpoint: {live_model_path}")
    logging.info(f"Using SWA Model: {swa_model_path}")
    val_ds = ChunkDataset(FLAGS.val_ds_path, config.batch_size)

    with open(FLAGS.batch_num_path, "r") as f:  # assumes file is already created.
        batch_num = int(f.read())

    logging.info(
        f"Model Path: {live_model_path}, Training on Chunk: {FLAGS.chunk_path}"
    )
    batch_num, live_model, swa_model, optimizer = rl_loop.train.train_one_gen(
        live_model,
        swa_model,
        optimizer,
        FLAGS.gen,
        FLAGS.chunk_path,
        val_ds,
        config=config,
        is_gpu=is_gpu,
        batch_num=batch_num,
        chunk_size=FLAGS.chunk_size,
    )
    if FLAGS.adaptive_wd_baseline_path or FLAGS.gamma_pull_tau_steps > 0:
        inner = getattr(optimizer, "inner_optimizer", optimizer)
        g = torch.cat([x.detach().flatten() for x in _trunk_gammas(live_model)])
        logging.info(
            f"Norm control end of gen: trunk weight norm "
            f"{inner.params_norm(_trunk_weight_params(live_model)):.4f} "
            f"(ratio {inner.last_norm_ratio}, adaptive scale {inner._adaptive_scale:.4f}); "
            f"trunk |gamma| median {g.abs().median():.4f} mean {g.mean():.4f}"
        )
    save_model(live_model, live_model_path, optimizer=optimizer)

    # save live checkpoint
    live_model_ckpt_path = str(
        Path(FLAGS.models_dir, "_live", f"live_{FLAGS.next_gen:04d}{MODEL_EXT}")
    )
    Path(FLAGS.models_dir, "_live").mkdir(exist_ok=True)
    save_model(live_model, live_model_ckpt_path, optimizer=optimizer)
    if FLAGS.save_trt:
        model_utils.save_onnx_trt(
            swa_model,
            FLAGS.val_ds_path,
            FLAGS.models_dir,
            FLAGS.next_gen,
            batch_size=SELFPLAY_BATCH_SIZE,
            trt_convert_path=FLAGS.trt_convert_path,
        )
    else:
        model_utils.save(swa_model, FLAGS.models_dir, FLAGS.next_gen)

    with open(FLAGS.batch_num_path, "w") as f:
        f.write(str(batch_num))


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    app.run(main)
