"""Backend-specific model lifecycle for the TF/keras backend.

Mirrors `backend_torch/model_utils.py`. Only the file extensions and the
underlying load/save calls differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import keras
import numpy as np


def save_model(model, path: str, optimizer=None) -> None:
    """Save model to a `.keras` file. If `optimizer` is given, compile() it
    onto the model first so its state is bundled into the keras zip."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if optimizer is not None:
        model.compile(optimizer=optimizer)
    model.save(path)


def load_model(path: str):
    """Load model only; the keras-rehydrated optimizer (which lives on
    `model.optimizer` after compile=True) is discarded. Use
    `load_with_optimizer` for resume."""
    model, _ = load_with_optimizer(path)
    return model


def load_with_optimizer(path: str):
    """Load model + the keras Optimizer that was bundled in via
    `model.compile(optimizer=...)` at save time. Returns
    `(model, optimizer_or_None)`."""
    # Lazy: model.py imports backend_shim, which imports this module — deferring
    # the P3achyGoModel import here breaks the cycle.
    from model import P3achyGoModel

    model = keras.models.load_model(
        path,
        custom_objects=P3achyGoModel.custom_objects(),
        compile=True,
    )
    return model, getattr(model, "optimizer", None)


def new_model(
    config,
    board_len: int,
    num_input_planes: int,
    num_input_features: int,
    name: str,
):
    """Build a fresh keras `P3achyGoModel` from a `ModelConfig` + run-wide
    constants, and trigger variable materialization with a dummy forward
    pass so the returned model is ready to save / serialize."""
    from model import P3achyGoModel

    model = P3achyGoModel.create(
        config=config,
        board_len=board_len,
        num_input_planes=num_input_planes,
        num_input_features=num_input_features,
        name=name,
    )
    model(
        np.zeros([1, *model.input_planes_shape()], dtype=np.float32),
        np.zeros([1, *model.input_features_shape()], dtype=np.float32),
    )
    return model


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


def summary(model) -> None:
    """Print a human-readable model summary. Delegates to keras' built-in."""
    model.summary()


def compile_for_training(model, *, fp16: bool = True, channels_last: bool = True):
    """No-op on TF — `keras.mixed_precision.set_global_policy("mixed_float16")`
    set in `configure_gpu` already covers the equivalent transforms. The TF
    backend handles channels_last and graph compilation via XLA / @tf.function
    inside `train_step` itself."""
    del fp16, channels_last
    return model


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
    """Recompute BN moving mean/variance to be the EXACT batch-statistics
    average over `num_batches` forwards (not an EMA approximation).

    keras BN uses `moving = momentum * moving + (1 - momentum) * batch_stat`.
    With moving stats reset to zero and momentum set to `(i-1)/i` at the
    i-th batch (1-indexed), the recurrence becomes the cumulative running
    average — `moving_n = (1/n) * Σ batch_stat_k` — converging to the exact
    mean over the visited batches. Restores the original momentum at exit.
    Mirrors the `bn.momentum = None` trick on the torch backend (see
    `torch.optim.swa_utils.update_bn`).

    Does NOT compute gradients."""
    bn_layers = _get_bn_layers(model)
    print(f"Found {len(bn_layers)} BatchNorm layers (TF backend)")
    saved_momentum = [bn.momentum for bn in bn_layers]
    # Reset moving stats so the cumulative average starts from zero.
    for bn in bn_layers:
        bn.moving_mean.assign(keras.ops.zeros_like(bn.moving_mean))
        bn.moving_variance.assign(keras.ops.zeros_like(bn.moving_variance))
    try:
        for i, batch in enumerate(ds.take(num_batches)):
            n = i + 1  # 1-indexed batch count
            new_momentum = (n - 1) / n
            for bn in bn_layers:
                bn.momentum = new_momentum
            input_board, input_global = batch[0], batch[1]
            _ = model(input_board, input_global, training=True)
            if n % 20 == 0:
                print(
                    f"=== recompute_bn_statistics: Processed {n}/{num_batches} batches ==="
                )
    finally:
        for bn, mom in zip(bn_layers, saved_momentum):
            bn.momentum = mom


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
    *,
    loaded_state: Optional[Any] = None,
) -> Any:
    """Build (when `loaded_state is None`) or rehydrate an optimizer.

    `loaded_state` is opaque from the caller's perspective — pass back
    whatever `load_with_optimizer` (or a previous `make_optimizer`) gave
    you. On the keras (TF) backend the rehydrated value is always a
    previously-built keras Optimizer object (state lives on the object
    itself), so reload reduces to "mutate config-driven fields in place".

    Returns an optimizer ready for the keras-on-{tf,torch} train_step.
    Wraps in `keras.mixed_precision.LossScaleOptimizer` when `is_gpu` and
    not already wrapped.
    """
    from optimizer import ConvMuon  # keras ConvMuon

    optimizer = loaded_state
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
