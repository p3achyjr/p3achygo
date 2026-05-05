"""Backend-specific model lifecycle for the TF/keras backend.

Mirrors `backend_torch/model_utils.py`. Only the file extensions and the
underlying load/save calls differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import keras
import numpy as np

LIVE_MODEL_NAME = "live_model.keras"
MODEL_EXT = ".keras"


def save_model(model, path: str, optimizer=None) -> None:
    """Save model to a `.keras` file. If `optimizer` is given, compile() it
    onto the model first so its state is bundled into the keras zip."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if optimizer is not None:
        model.compile(optimizer=optimizer)
    model.save(path)


def load_model(path: str):
    # Lazy: model.py imports train_shim, which imports this module — deferring
    # the P3achyGoModel import here breaks the cycle.
    from model import P3achyGoModel

    # compile=True so any bundled optimizer is rehydrated and accessible via
    # `model.optimizer`. Cross-process resume relies on this.
    return keras.models.load_model(
        path,
        custom_objects=P3achyGoModel.custom_objects(),
        compile=True,
    )


def optimizer_state_from_model(model):
    """Return the optimizer attached to a freshly-loaded model, or None.
    On TF, this is the keras Optimizer rehydrated by `compile=True` in
    `load_model`."""
    return getattr(model, "optimizer", None)


def new_model(config: Dict):
    from model import P3achyGoModel

    c = dict(config)
    return P3achyGoModel.from_config(c)


def clone_model(model):
    return keras.models.clone_model(model)


# ---------------------------------------------------------------------------
# Weight utilities (backend-natural form: list of numpy arrays)
# ---------------------------------------------------------------------------


def get_weights(model):
    return model.get_weights()


def set_weights(model, weights):
    model.set_weights(weights)


def to_numpy(t):
    """Convert any TF/keras tensor to a numpy array."""
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def swa_avg_weights(weights_list, swa_momentum: float = 0.75):
    """Cascading EMA across snapshots:

        swa = w[0]
        for w[i] in w[1:]:
            swa = swa * momentum + w[i] * (1 - momentum)

    Operates on lists of numpy arrays (the form keras `get_weights()` returns).
    """
    swa = weights_list[0]
    for w in weights_list[1:]:
        swa = [a * swa_momentum + b * (1 - swa_momentum) for a, b in zip(swa, w)]
    return swa


def recompute_bn_statistics(model, ds, num_batches: int = 150):
    """Run forward passes with training=True so keras BN layers update their
    `moving_mean` / `moving_variance` in-place. Does NOT compute gradients."""
    bn_layers = _get_bn_layers(model)
    print(f"Found {len(bn_layers)} BatchNorm layers (TF backend)")
    for i, batch in enumerate(ds.take(num_batches)):
        input_board, input_global = batch[0], batch[1]
        _ = model(input_board, input_global, training=True)
        if (i + 1) % 20 == 0:
            print(
                f"=== recompute_bn_statistics: Processed {i + 1}/{num_batches} batches ==="
            )


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

_SGD_MOMENTUM = 0.9
_SGD_CLIPNORM = 20.0
_MUON_EXCLUDE_LAYERS = [r".*policy_head\/.*", r".*value_head\/.*"]


def _inner(opt):
    """Unwrap a possibly-LSO-wrapped keras optimizer."""
    return getattr(opt, "inner_optimizer", opt) if opt is not None else None


def make_optimizer(
    model: Any,
    config: Any,
    lr_schedule: Any,
    is_gpu: bool,
    optimizer: Optional[Any] = None,
    optimizer_state: Optional[Any] = None,
) -> Any:
    """Build (when `optimizer is None`) or hot-reload an optimizer.

    Returns an optimizer ready for the keras-on-{tf,torch} train_step.
    Wraps in `keras.mixed_precision.LossScaleOptimizer` when `is_gpu` and
    not already wrapped.

    `optimizer_state` (TF backend): a previously-built keras Optimizer
    rehydrated from disk via `load_model(compile=True)`. Treated equivalent
    to passing `optimizer=...` (in-process hot-reload). The keras flow
    bundles state on the optimizer object itself, so there is no separate
    state container to apply.
    """
    from optimizer import ConvMuon  # keras ConvMuon

    if optimizer is None and optimizer_state is not None:
        optimizer = optimizer_state

    inner = _inner(optimizer)
    if inner is None:
        if config.optimizer == "muon":
            inner = ConvMuon(
                learning_rate=lr_schedule,
                exclude_layers=_MUON_EXCLUDE_LAYERS,
                weight_decay=config.muon_wd,
                adam_weight_decay=config.adam_wd,
                adam_lr_ratio=config.adam_lr_ratio,
                wd_lr_exponent=config.wd_lr_exponent,
                wd_lr_max=config.wd_lr_max,
                global_clipnorm=config.global_clipnorm,
            )
        else:
            inner = keras.optimizers.SGD(
                learning_rate=lr_schedule,
                momentum=_SGD_MOMENTUM,
                global_clipnorm=_SGD_CLIPNORM,
                nesterov=True,
            )
    else:
        # Hot-reload config-driven fields. Mirrors the (legacy) inline body
        # in rl_loop/train.py.
        inner.learning_rate = lr_schedule
        if isinstance(inner, ConvMuon):
            inner.weight_decay = config.muon_wd
            inner.adam_weight_decay = config.adam_wd
            inner.adam_lr_ratio = config.adam_lr_ratio
            inner.wd_lr_exponent = config.wd_lr_exponent
            inner.wd_lr_max = config.wd_lr_max
        inner.global_clipnorm = config.global_clipnorm

    if is_gpu:
        if isinstance(optimizer, keras.mixed_precision.LossScaleOptimizer):
            return optimizer  # already wrapped, inner mutated above
        return keras.mixed_precision.LossScaleOptimizer(inner)
    return inner


def _get_bn_layers(model):
    """Recursive walk that picks up nested layers (custom layers expose
    `_layers` privately; `model.layers` only sees direct children)."""

    def _all_layers(layer):
        out = [layer]
        for sub in getattr(layer, "_layers", []):
            out.extend(_all_layers(sub))
        return out

    flat = []
    for layer in model.layers:
        flat.extend(_all_layers(layer))
    return [l for l in flat if isinstance(l, keras.layers.BatchNormalization)]
