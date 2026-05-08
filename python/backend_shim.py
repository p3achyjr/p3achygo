"""
Backend-aware symbol resolution for training.

All branching happens here, at import time. Implementations live in
backend_tf/ and backend_torch/ and are themselves framework-specific
and clean (no conditional imports in function bodies).

Backend selection precedence:
  1. If both `P3ACHYGO_BACKEND` and `KERAS_BACKEND` are set and disagree,
     raise immediately — config conflict, refuse to guess.
  2. Otherwise, the active backend is whichever of the two is set, with
     `P3ACHYGO_BACKEND` taking precedence when both agree (or when only it
     is set), and falling back to `KERAS_BACKEND`, and ultimately to
     "tensorflow" if neither is set.

`load_model(path)` additionally inspects the file extension (`.keras` vs
`.pt`). If the suffix indicates a different backend than the active one,
it raises — the user must align their environment with the artifact.
This is a hard error rather than a silent re-dispatch because mixing
backends inside one process produces confusing failures elsewhere
(optimizer state, train_step references, mixed-precision policy).

Exported symbols:
  ConvMuon                       — optimizer for trunk/body
  ModelPredictions, GroundTruth  — data containers
  TrainStepResult                — train_step output
  train_step, val_step           — single forward/backward step
  step_begin                     — per-step CUDA-graph boundary marker (no-op on TF)
  configure_gpu                  — backend-agnostic GPU + mixed-precision setup
  gpu_count                      — number of visible GPUs
  export_to_onnx                 — spawn the right ONNX exporter script
  SummaryWriter                  — backend-agnostic TensorBoard scalar writer
  load_model, load_with_optimizer, save_model
                                 — checkpoint I/O (load auto-detects suffix)
  new_model, clone_model, summary, compile_for_training
                                 — model lifecycle
  get_weights, set_weights, to_numpy
                                 — weight / tensor utilities
  swa_avg_weights                — cascading-EMA across snapshots
  recompute_bn_statistics        — refresh BN running stats after SWA
  make_optimizer                 — build / hot-reload optimizer from RunConfig
  BACKEND                        — string, "tensorflow" or "torch"
  MODEL_EXT, LIVE_MODEL_NAME, MODEL_FORMAT, MODEL_RE
                                 — backend-appropriate filename / extension constants
"""

from __future__ import annotations

import re

from backend import BACKEND as _backend
from loss_coeffs import LossWeights  # re-export for backend-agnostic callers


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

if _backend == "torch":
    from backend_torch.train import (
        ModelPredictions,
        GroundTruth,
        TrainStepResult,
        train_step,
        val_step,
    )
    from backend_torch.runtime import configure_gpu, gpu_count, SummaryWriter
    from backend_torch.model_utils import (
        load_model as _native_load_model,
        load_with_optimizer as _native_load_with_optimizer,
        save_model,
        new_model,
        clone_model,
        get_weights,
        set_weights,
        to_numpy,
        summary,
        swa_avg_weights,
        recompute_bn_statistics,
        compile_for_training,
    )
    from backend_torch.optimizer import ConvMuon, make_optimizer
elif _backend == "tensorflow":
    from backend_tf.train import (
        ModelPredictions,
        GroundTruth,
        TrainStepResult,
        train_step,
        val_step,
    )
    from backend_tf.runtime import configure_gpu, gpu_count, SummaryWriter
    from backend_tf.model_utils import (
        load_model as _native_load_model,
        load_with_optimizer as _native_load_with_optimizer,
        save_model,
        new_model,
        clone_model,
        get_weights,
        set_weights,
        to_numpy,
        summary,
        swa_avg_weights,
        recompute_bn_statistics,
        compile_for_training,
        make_optimizer,
    )
    from backend_tf.optimizer import ConvMuon
else:
    raise ValueError(
        f"unsupported backend {_backend!r} "
        "(set P3ACHYGO_BACKEND=torch or tensorflow)"
    )

BACKEND = _backend


# ---------------------------------------------------------------------------
# Backend ↔ file-extension registry — single source of truth.
# ---------------------------------------------------------------------------

_BACKEND_EXTS = {"tensorflow": ".keras", "torch": ".pt"}
_SUFFIX_TO_BACKEND = {ext: name for name, ext in _BACKEND_EXTS.items()}

# Backend-aware constants for the active backend. Callers (gcs_utils,
# rl_loop training, save/load) read these instead of hardcoding `.keras`
# or `.pt`.
MODEL_EXT = _BACKEND_EXTS[BACKEND]
LIVE_MODEL_NAME = f"live_model{MODEL_EXT}"
_MODEL_PREFIX = "model"
MODEL_FORMAT = _MODEL_PREFIX + "_{:04d}" + MODEL_EXT
MODEL_RE = re.compile(_MODEL_PREFIX + r"_([0-9]+)" + re.escape(MODEL_EXT))


# ---------------------------------------------------------------------------
# Per-step CUDA-graph boundary marker. Required at the top of every training
# / validation iteration when the model is compiled with `mode="reduce-overhead"`
# so torch can reclaim the captured output buffers between iters.
# No-op on TF (keras handles its own compile/graph boundaries).
# ---------------------------------------------------------------------------

if BACKEND == "torch":
    import torch as _torch

    _mark_step_begin = getattr(
        _torch.compiler, "cudagraph_mark_step_begin", lambda: None
    )

    def step_begin() -> None:
        _mark_step_begin()

else:

    def step_begin() -> None:
        pass


# ---------------------------------------------------------------------------
# Suffix-aware load_model — raises if suffix disagrees with active backend.
# `_SUFFIX_TO_BACKEND` is derived from `_BACKEND_EXTS` above, so the
# extension list is single-sourced.
# ---------------------------------------------------------------------------


def _backend_from_path(path) -> str | None:
    p = str(path)
    for suffix, backend in _SUFFIX_TO_BACKEND.items():
        if p.endswith(suffix):
            return backend
    return None


def _check_suffix(path):
    suffix_backend = _backend_from_path(path)
    if suffix_backend is not None and suffix_backend != BACKEND:
        raise RuntimeError(
            f"Cannot load {path!r} (suffix indicates backend {suffix_backend!r}) "
            f"under active backend {BACKEND!r}. "
            f"Set P3ACHYGO_BACKEND={suffix_backend!r} (and KERAS_BACKEND to match) "
            "before importing backend_shim, or save the model in the active "
            "backend's format."
        )


def load_model(path):
    """Load a model checkpoint. Reads the file suffix (.keras / .pt) and
    raises if it indicates a different backend than the active one. The
    bundled optimizer (if any) is discarded; use `load_with_optimizer`
    for cross-process resume."""
    _check_suffix(path)
    return _native_load_model(path)


def load_with_optimizer(path):
    """Load model + bundled optimizer state from `path`. Returns
    `(model, opt_state)`; `opt_state` is None when the file didn't
    bundle one. Use this in production resume paths.

    `opt_state` is opaque to the caller — feed it back to
    `make_optimizer(..., loaded_state=opt_state)`, which sniffs the type
    (rehydrated keras Optimizer on TF, state_dict on torch)."""
    _check_suffix(path)
    return _native_load_with_optimizer(path)


# ---------------------------------------------------------------------------
# Backend-aware ONNX export. Spawns the right script (torch_to_onnx vs
# convert_to_onnx) and writes to `<dir>/_onnx/<stem>.onnx` regardless of
# backend, so the rest of the pipeline (TRT engine builder, sp_loop,
# eval) sees a single canonical naming.
# ---------------------------------------------------------------------------


def export_to_onnx(model_path: str, fp16: bool = True) -> str:
    """Export the saved model at `model_path` to ONNX, returning the
    output path. The output is always `<dir>/_onnx/<stem>.onnx` —
    matching the keras path's natural output, with the torch script
    invoked via `--onnx_path` to override its default `_pt` suffix."""
    from pathlib import Path

    import proc

    model_p = Path(model_path)
    onnx_path = str(model_p.parent / "_onnx" / (model_p.stem + ".onnx"))
    fp16_flag = " --fp16" if fp16 else ""
    if BACKEND == "torch":
        cmd = (
            f"python -m python.scripts.torch_to_onnx"
            f" --model_path={model_path}"
            f" --onnx_path={onnx_path}"
            f"{fp16_flag}"
        )
    else:
        cmd = (
            f"python -m python.scripts.convert_to_onnx"
            f" --model_path={model_path}"
            f"{fp16_flag}"
        )
    proc.run_proc(cmd)
    return onnx_path


# ---------------------------------------------------------------------------
# P3achyGoModel dispatch — eager. Imported at the bottom so that all
# symbols needed by the backend's model module (e.g., LossWeights,
# ModelPredictions) are already defined in this module's namespace.
# ---------------------------------------------------------------------------

if BACKEND == "torch":
    from backend_torch.model import P3achyGoModel
else:
    from backend_tf.model import P3achyGoModel
