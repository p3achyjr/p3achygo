"""Model utilities for the torch-native backend.

load_model(path)        — load a .pt checkpoint → P3achyGoModel
new_model(config_dict)  — create P3achyGoModel from config dict
save_model(model, path) — save model state + config to .pt

migrate_from_keras(keras_model) — copy weights from a loaded keras model
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from backend_torch.model import P3achyGoModel


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

LIVE_MODEL_NAME = "live_model.pt"
MODEL_EXT = ".pt"


def save_model(model: P3achyGoModel, path: str, optimizer=None) -> None:
    """Save model + (optional) optimizer state to a `.pt` file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = {"model": model.state_dict(), "config": model.config_dict()}
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    torch.save(blob, path)


def clone_model(model: P3achyGoModel) -> P3achyGoModel:
    """Build a structural copy with copied weights (CPU tensors)."""
    import copy

    cloned = _model_from_config(model.config_dict())
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    return cloned


def load_model(path: str) -> P3achyGoModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = _model_from_config(config)
    model.load_state_dict(ckpt["model"])
    # Stash any saved optimizer state so the caller can recover it via
    # `optimizer_state_from_model(model)`. Mirrors the TF flow where
    # keras.load_model(compile=True) sets `model.optimizer`.
    model._p3achygo_opt_state = ckpt.get("optimizer")
    return model


def optimizer_state_from_model(model: P3achyGoModel):
    """Return the saved optimizer state_dict attached during `load_model`,
    or None if absent. Cross-process resume uses this."""
    return getattr(model, "_p3achygo_opt_state", None)


def new_model(config: Dict) -> P3achyGoModel:
    return _model_from_config(config)


# ---------------------------------------------------------------------------
# Weight utilities (backend-natural form: state_dict)
# ---------------------------------------------------------------------------


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a contiguous CPU numpy array.
    Detaches first so this works on tensors with grad."""
    return t.detach().cpu().numpy()


def get_weights(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU state_dict (cloned tensors so subsequent training
    doesn't mutate the snapshot)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_weights(model: torch.nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    # Move incoming tensors onto the model's existing param devices/dtypes.
    target = model.state_dict()
    converted = {
        k: weights[k].to(dtype=target[k].dtype, device=target[k].device)
        for k in target.keys()
        if k in weights
    }
    missing = set(target.keys()) - set(converted.keys())
    if missing:
        raise KeyError(f"set_weights: missing keys: {sorted(missing)[:5]}…")
    model.load_state_dict(converted, strict=True)


def swa_avg_weights(
    weights_list, swa_momentum: float = 0.75
) -> Dict[str, torch.Tensor]:
    """Cascading EMA across snapshots — torch state_dict form.

    Skips integer-typed entries (e.g. BN `num_batches_tracked`); those carry
    no meaningful average and survive on the seed snapshot.
    """
    seed = weights_list[0]
    swa: Dict[str, torch.Tensor] = {k: v.clone().float() for k, v in seed.items()}
    int_keys = {
        k for k, v in seed.items() if v.dtype in (torch.int32, torch.int64, torch.long)
    }
    for w in weights_list[1:]:
        for k, v in swa.items():
            if k in int_keys:
                continue
            swa[k] = v * swa_momentum + w[k].float() * (1 - swa_momentum)
    # Restore original dtypes (float→{float, BN ints unchanged})
    out = {}
    for k, v in seed.items():
        out[k] = swa[k].to(dtype=v.dtype) if k not in int_keys else v.clone()
    return out


def recompute_bn_statistics(model: torch.nn.Module, ds, num_batches: int = 150) -> None:
    """Reset BN running stats (so they reflect ONLY the new SWA params), then
    run `num_batches` forwards in `train()` mode so each BN layer rebuilds
    its `running_mean` / `running_var` from observed activations.

    Does NOT compute gradients — wraps the whole loop in `torch.no_grad`.
    """
    bn_modules = [
        m
        for m in model.modules()
        if isinstance(
            m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        )
    ]
    print(f"Found {len(bn_modules)} BatchNorm modules (torch backend)")
    # Reset running stats so the recompute starts from a clean state. The
    # SWA-averaged weights already replaced the params; we want the running
    # stats to match those averaged params, not whatever lingered.
    for bn in bn_modules:
        bn.reset_running_stats()
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            for i, batch in enumerate(ds):
                if i >= num_batches:
                    break
                input_board, input_global = batch[0], batch[1]
                _ = model(input_board, input_global)
                if (i + 1) % 20 == 0:
                    print(
                        f"=== recompute_bn_statistics: Processed {i + 1}/{num_batches} batches ==="
                    )
    finally:
        if not was_training:
            model.eval()


def _model_from_config(config: Dict) -> P3achyGoModel:
    # Accept both old-style (num_head_channels) and new-style keys.
    c = dict(config)
    if "num_head_channels" in c and "num_policy_head_channels" not in c:
        c["num_policy_head_channels"] = c.pop("num_head_channels")
        c["num_value_head_channels"] = c.get("c_val", 64)
    c.pop("name", None)  # P3achyGoModel doesn't take 'name' as kwarg
    c.pop("is_transformer", None)
    return P3achyGoModel(**c)


# ---------------------------------------------------------------------------
# Weight migration: keras → torch state dict
# ---------------------------------------------------------------------------


def migrate_from_keras(keras_model) -> P3achyGoModel:
    """Build and populate a torch model from a loaded keras model.

    keras_model must already be built (forward pass called once).
    """
    config = keras_model.get_config()
    model = _model_from_config(config)
    _copy_weights(keras_model, model)
    return model


def _copy_weights(keras_model, torch_model: P3achyGoModel) -> None:
    """Walk the keras weight tree and copy into torch parameters."""
    # Build a flat name → numpy map from the keras model.
    kw: Dict[str, np.ndarray] = {v.path: v.numpy() for v in keras_model.variables}
    _copy_recursive("p3achygo", keras_model, torch_model, kw)


def _copy_recursive(
    prefix: str, k_layer, t_module: torch.nn.Module, kw: Dict[str, np.ndarray]
) -> None:
    """Recursively match keras sub-layers to torch sub-modules by structure."""
    # --- ConvPreActivation / ConvPostActivation ---
    from backend_torch.model_layers_common import ConvPreActivation, ConvPostActivation

    if isinstance(t_module, (ConvPreActivation, ConvPostActivation)):
        _copy_conv_block(k_layer, t_module, kw)
        return

    # --- nn.Conv2d ---
    if isinstance(t_module, torch.nn.Conv2d):
        _copy_conv2d(k_layer, t_module, kw)
        return

    # --- nn.BatchNorm2d ---
    if isinstance(t_module, torch.nn.BatchNorm2d):
        _copy_bn(k_layer, t_module, kw)
        return

    # --- nn.Linear ---
    if isinstance(t_module, torch.nn.Linear):
        _copy_linear(k_layer, t_module, kw)
        return

    # --- recurse ---
    # Try to match keras sub-layers to torch named children by order.
    k_sublayers = getattr(k_layer, "_sublayers", None)
    if k_sublayers is None:
        return

    k_children = list(k_sublayers)
    t_children = list(t_module.named_children())
    # Best-effort: same number of children
    for (t_name, t_child), k_child in zip(t_children, k_children):
        _copy_recursive(k_child.name, k_child, t_child, kw)


# ---------------------------------------------------------------------------
# Fine-grained copiers
# ---------------------------------------------------------------------------


def _find_var(kw: Dict[str, np.ndarray], *substrings) -> Optional[np.ndarray]:
    """Return the first kw entry whose path contains ALL substrings."""
    for path, arr in kw.items():
        if all(s in path for s in substrings):
            return arr
    return None


def _copy_conv2d(k_conv, t_conv: torch.nn.Conv2d, kw: Dict[str, np.ndarray]) -> None:
    kernel = k_conv.kernel.numpy()  # (H,W,Cin,Cout)
    with torch.no_grad():
        t_conv.weight.copy_(torch.tensor(kernel).permute(3, 2, 0, 1))
        if (
            t_conv.bias is not None
            and hasattr(k_conv, "bias")
            and k_conv.bias is not None
        ):
            t_conv.bias.copy_(torch.tensor(k_conv.bias.numpy()))


def _copy_bn(k_bn, t_bn: torch.nn.BatchNorm2d, kw: Dict[str, np.ndarray]) -> None:
    with torch.no_grad():
        t_bn.weight.copy_(torch.tensor(k_bn.gamma.numpy()))
        t_bn.bias.copy_(torch.tensor(k_bn.beta.numpy()))
        t_bn.running_mean.copy_(torch.tensor(k_bn.moving_mean.numpy()))
        t_bn.running_var.copy_(torch.tensor(k_bn.moving_variance.numpy()))


def _copy_linear(k_dense, t_linear: torch.nn.Linear, kw: Dict[str, np.ndarray]) -> None:
    kernel = k_dense.kernel.numpy()  # (Cin, Cout)
    with torch.no_grad():
        t_linear.weight.copy_(torch.tensor(kernel).T)
        if (
            t_linear.bias is not None
            and hasattr(k_dense, "bias")
            and k_dense.bias is not None
        ):
            t_linear.bias.copy_(torch.tensor(k_dense.bias.numpy()))


def _copy_conv_block(k_block, t_block, kw: Dict[str, np.ndarray]) -> None:
    """Copy a keras ConvPreActivation/ConvPostActivation → torch ConvPreActivation."""
    from backend_torch.model_layers_common import ConvPreActivation, ConvPostActivation

    _copy_conv2d(k_block.conv, t_block.conv, kw)
    _copy_bn(k_block.norm_layer, t_block.bn, kw)
