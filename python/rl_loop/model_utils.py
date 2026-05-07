from __future__ import annotations

from absl import logging
from typing import TYPE_CHECKING

import gcs_utils as gcs
import rl_loop.fs_utils as fs

from model_config import ModelConfig
from constants import *
import proc
import backend_shim

from pathlib import Path

if TYPE_CHECKING:
    from backend_shim import P3achyGoModel

NUM_BATCHES_FULL_CHECKPOINT = 1000


def new_model(name: str, model_config="small", optimizer="sgd"):
    """Build a fresh model from a `ModelConfig`. The active backend's
    `new_model` is responsible for any backend-specific initialization
    (e.g. keras variable materialization); the returned model is ready
    to save."""
    config = (
        ModelConfig.from_generic_arch(model_config)
        if isinstance(model_config, dict)
        else ModelConfig.from_str(model_config)
    )
    if optimizer == "muon":
        config.c_l2 = 0

    return backend_shim.new_model(
        config,
        board_len=BOARD_LEN,
        num_input_planes=num_input_planes(),
        num_input_features=num_input_features(),
        name=name,
    )


# Re-exported from backend_shim for callers that still import it via
# `rl_loop.model_utils`. The weights value is opaque — produced by
# `backend_shim.get_weights` and consumed by `backend_shim.set_weights`
# / `swa_avg_weights`.
swa_avg_weights = backend_shim.swa_avg_weights


def recompute_bn_statistics(model, ds, num_batches=150):
    return backend_shim.recompute_bn_statistics(model, ds, num_batches=num_batches)


def avg_weights(
    prev_weights: list,
    cur_weights: list,
    num_batches_in_chunk: int,
    swa_momentum: float = 0.4,
) -> list:
    # chunk_ratio = min(1.0,
    #                   float(num_batches_in_chunk) / NUM_BATCHES_FULL_CHECKPOINT)
    # m_swa_new = (1 - SWA_MOMENTUM) * chunk_ratio
    # swa_momentum = 1 - m_swa_new
    print("SWA Momentum:", swa_momentum, "Num Batches: ", num_batches_in_chunk)
    return [
        prev_layer_weights * swa_momentum + layer_weights * (1 - swa_momentum)
        for prev_layer_weights, layer_weights in zip(prev_weights, cur_weights)
    ]


def save_trt_and_upload(
    model: P3achyGoModel,
    calib_ds_path: str,
    local_model_dir: str,
    gen: int,
    run_id: str,
    batch_size: int,
) -> str:
    model_path = save_trt(model, calib_ds_path, local_model_dir, gen, batch_size)
    fs.upload_model(run_id, str(local_model_dir), gen)

    return model_path


def save_trt(
    model: P3achyGoModel,
    calib_ds_path: str,
    local_model_dir: str,
    gen: int,
    batch_size: int,
) -> str:
    """
    Saves model, converts to ONNX, and returns _base_ path of model.
    """
    model_path = save(model, local_model_dir, gen)

    logging.info("Converting to ONNX...")
    cmd = f"python -m python.scripts.convert_to_onnx --model_path={model_path}"
    proc.run_proc(cmd)

    return str(model_path)


def save_onnx_trt(
    model: P3achyGoModel,
    calib_ds_path: str,
    local_model_dir: str,
    gen: int,
    batch_size: int,
    trt_convert_path: str,
) -> str:
    """
    Saves model through ONNX -> TRT path.
    """
    model_path = save(model, local_model_dir, gen)
    logging.info("Converting to ONNX...")
    cmd = f"python -m python.scripts.convert_to_onnx --model_path={model_path} --fp16"
    proc.run_proc(cmd)

    logging.info("Converting to ONNX-TRT...")
    model_p = Path(model_path)
    onnx_path = str(model_p.parent / "_onnx" / (model_p.stem + ".onnx"))
    trt_cmd = (
        f"{trt_convert_path} --onnx_path={onnx_path}"
        + f" --ds_path={calib_ds_path}"
        + f" --batch_size={batch_size}"
    )

    proc.run_proc(trt_cmd)
    return str(model_p.parent / "_onnx" / (model_p.stem + ".trt"))


def save(model, local_model_dir: str, gen: int) -> str:
    """Save the per-generation SWA model to disk and return its path.

    Backend-aware: dispatches via `backend_shim.save_model`, so the file
    extension matches the active backend (`.keras` on TF, `.pt` on torch
    — handled by `gcs.MODEL_FORMAT`)."""
    model_path = Path(local_model_dir, gcs.MODEL_FORMAT.format(gen))
    backend_shim.save_model(model, str(model_path))
    return str(model_path)
