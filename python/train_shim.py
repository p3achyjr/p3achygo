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
  configure_gpu                  — backend-agnostic GPU + mixed-precision setup
  SummaryWriter                  — backend-agnostic TensorBoard scalar writer
  load_model, save_model         — checkpoint I/O (load auto-detects suffix)
  new_model, clone_model         — model lifecycle
  get_weights, set_weights       — opaque per-backend snapshot/restore
  swa_avg_weights                — cascading-EMA across snapshots
  recompute_bn_statistics        — refresh BN running stats after SWA
  make_optimizer                 — build / hot-reload optimizer from RunConfig
  LIVE_MODEL_NAME, MODEL_EXT     — backend-appropriate filename / extension
  BACKEND                        — string, "tensorflow" or "torch"
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_p3 = os.environ.get("P3ACHYGO_BACKEND")
_kb = os.environ.get("KERAS_BACKEND")


# Normalize "tf" → "tensorflow" so the two envs can be compared directly.
def _norm(b):
    if b is None:
        return None
    return "tensorflow" if b in ("tf", "tensorflow") else b


_p3n, _kbn = _norm(_p3), _norm(_kb)
if _p3n and _kbn and _p3n != _kbn:
    raise RuntimeError(
        f"P3ACHYGO_BACKEND={_p3!r} disagrees with KERAS_BACKEND={_kb!r}. "
        "Set them to the same value (or unset one) before importing train_shim."
    )

_backend = _p3n or _kbn or "tensorflow"

if _backend == "torch":
    from backend_torch.train import (
        ModelPredictions,
        GroundTruth,
        TrainStepResult,
        train_step,
        val_step,
    )
    from backend_torch.runtime import configure_gpu, SummaryWriter
    from backend_torch.model_utils import (
        load_model as _native_load_model,
        save_model,
        new_model,
        clone_model,
        get_weights,
        set_weights,
        to_numpy,
        swa_avg_weights,
        recompute_bn_statistics,
        optimizer_state_from_model,
        LIVE_MODEL_NAME,
        MODEL_EXT,
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
    from backend_tf.runtime import configure_gpu, SummaryWriter
    from backend_tf.model_utils import (
        load_model as _native_load_model,
        save_model,
        new_model,
        clone_model,
        get_weights,
        set_weights,
        to_numpy,
        swa_avg_weights,
        recompute_bn_statistics,
        optimizer_state_from_model,
        make_optimizer,
        LIVE_MODEL_NAME,
        MODEL_EXT,
    )
    from optimizer import ConvMuon
else:
    raise ValueError(
        f"unsupported backend {_backend!r} "
        "(set P3ACHYGO_BACKEND=torch or tensorflow)"
    )

BACKEND = _backend


# ---------------------------------------------------------------------------
# Suffix-aware load_model — raises if suffix disagrees with active backend
# ---------------------------------------------------------------------------

_SUFFIX_TO_BACKEND = {".keras": "tensorflow", ".pt": "torch"}


def _backend_from_path(path) -> str | None:
    p = str(path)
    for suffix, backend in _SUFFIX_TO_BACKEND.items():
        if p.endswith(suffix):
            return backend
    return None


def load_model(path):
    """Load a model checkpoint. Reads the file suffix (.keras / .pt) and
    raises if it indicates a different backend than the active one."""
    suffix_backend = _backend_from_path(path)
    if suffix_backend is not None and suffix_backend != BACKEND:
        raise RuntimeError(
            f"Cannot load {path!r} (suffix indicates backend {suffix_backend!r}) "
            f"under active backend {BACKEND!r}. "
            f"Set P3ACHYGO_BACKEND={suffix_backend!r} (and KERAS_BACKEND to match) "
            "before importing train_shim, or save the model in the active "
            "backend's format."
        )
    return _native_load_model(path)
