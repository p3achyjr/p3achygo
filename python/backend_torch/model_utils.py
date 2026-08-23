"""Model utilities for the torch-native backend.

load_model(path)        — load a .pt checkpoint → P3achyGoModel
load_with_optimizer(path) — load model + bundled optimizer state_dict
new_model(config, board_len, num_input_planes, num_input_features, name)
                        — build a fresh P3achyGoModel from a ModelConfig
save_model(model, path, optimizer=None)
                        — save model state + config (+ optional optimizer state) to .pt
compile_for_training(model) — CUDA + channels_last + torch.compile
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from backend_torch.model import P3achyGoModel


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def unwrap(model):
    """Return the underlying `nn.Module` if `model` was wrapped by
    `torch.compile`; otherwise return `model` unchanged. Used at every
    boundary that touches `state_dict` / `config_dict` / `load_state_dict`,
    so the rest of the code never has to think about the wrapper.
    """
    return getattr(model, "_orig_mod", model)


def save_model(model, path: str, optimizer=None) -> None:
    """Save model + (optional) optimizer state to a `.pt` file.

    Saves against the unwrapped module so the on-disk keys are always
    un-prefixed regardless of whether the live model was compiled.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    inner = unwrap(model)
    blob = {"model": inner.state_dict(), "config": inner.config_dict()}
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    torch.save(blob, path)


def clone_model(model: P3achyGoModel) -> P3achyGoModel:
    """Build a structural copy with copied weights (CPU tensors). The
    clone is always a fresh uncompiled module — call `compile_for_training`
    on the result if you intend to train it."""
    import copy

    inner = unwrap(model)
    cloned = _model_from_config(inner.config_dict())
    cloned.load_state_dict(copy.deepcopy(inner.state_dict()))
    return cloned


def load_model(path: str) -> P3achyGoModel:
    """Load model only; discard any bundled optimizer state. Use
    `load_with_optimizer` if you need the optimizer for resume."""
    model, _ = load_with_optimizer(path)
    return model


def load_with_optimizer(path: str):
    """Load model + optional optimizer state from a `.pt` checkpoint.
    Returns `(model, opt_state)` where `opt_state` is `None` if the file
    didn't bundle one."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = _model_from_config(config)
    model.load_state_dict(ckpt["model"])
    return model, ckpt.get("optimizer")


def new_model(
    config,
    board_len: int,
    num_input_planes: int,
    num_input_features: int,
    name: str,
) -> P3achyGoModel:
    """Build a fresh torch `P3achyGoModel` from a `ModelConfig` + run-wide
    constants. The `name` argument is accepted for parity with the TF
    backend's signature but is unused (torch `nn.Module` has no name slot).
    """
    del name
    return _model_from_config(
        config.to_torch_kwargs(
            board_len=board_len,
            num_input_planes=num_input_planes,
            num_input_features=num_input_features,
        )
    )


# ---------------------------------------------------------------------------
# Weight utilities (backend-natural form: state_dict)
# ---------------------------------------------------------------------------


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a contiguous CPU numpy array.
    Detaches first so this works on tensors with grad."""
    return t.detach().cpu().numpy()


def summary(model: torch.nn.Module) -> None:
    """Print a torch-native model summary: parameter count breakdown by
    top-level child module, plus the total. `nn.ModuleList` children
    (e.g. the trunk's `blocks`) are unwrapped so per-block sizes are
    visible. Mirrors the spirit of keras' `model.summary()`.
    """
    inner = unwrap(model)
    total = sum(p.numel() for p in inner.parameters())
    trainable = sum(p.numel() for p in inner.parameters() if p.requires_grad)
    print(f"Model summary: {type(inner).__name__}")
    for name, child in inner.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:<24} {type(child).__name__:<24} {n:>12,} params")
        if isinstance(child, torch.nn.ModuleList):
            for i, sub in enumerate(child):
                m = sum(p.numel() for p in sub.parameters())
                print(f"    [{i:>2}] {type(sub).__name__:<37} {m:>12,} params")
    print(f"  {'total':<49} {total:>12,} params")
    print(f"  {'trainable':<49} {trainable:>12,} params")


def _kill_compile_workers(_ac) -> None:
    for pool in list(_ac._pool_set):
        try:
            pool.process.kill()
        except Exception:
            pass


def compile_for_training(
    model: torch.nn.Module, *, channels_last: bool = True
) -> torch.nn.Module:
    """Apply the production training-time transforms to a torch model:
    - move to CUDA (weights stay fp32 — mixed-precision compute happens via
      `torch.amp.autocast` inside `train_step`, which keeps params fp32 and
      casts compute to fp16 for matmul/conv kernels)
    - optionally convert to channels_last memory format (cuDNN NHWC fast path)
    - wrap in `torch.compile(mode="reduce-overhead")` for CUDA-graph capture
      of forward + backward.

    Note: `mish` (in `model_layers_common.py`) is decorated with
    `@torch._dynamo.disable` because Inductor 2.11+cu13 miscompiles its
    backward under fp16/bf16 autocast and produces NaN grads. With that
    one op excluded, full Inductor + reduce-overhead works correctly.
    Override the backend via env var `P3ACHYGO_TORCH_COMPILE_BACKEND` if
    diagnosing regressions (e.g. "aot_eager", "eager").

    Caller must additionally call `backend_shim.step_begin()` once per
    training step so the captured output buffers can be reclaimed between
    iters. (`train.train` and `train.val` already do this.)
    """
    import atexit
    import os

    model = model.to("cuda")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    backend = os.environ.get("P3ACHYGO_TORCH_COMPILE_BACKEND")
    compiled = (
        torch.compile(model, backend=backend)
        if backend
        else torch.compile(model, mode="reduce-overhead")
    )

    # Replace the slow 300-second atexit shutdown with an immediate kill so
    # the process exits promptly when training ends.
    try:
        import torch._inductor.async_compile as _ac

        atexit.unregister(_ac.shutdown_compile_workers)
        atexit.register(_kill_compile_workers, _ac)
    except Exception:
        pass

    return compiled


def get_weights(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU state_dict (cloned tensors so subsequent training
    doesn't mutate the snapshot). Always reads from the unwrapped module
    so keys are stable across compiled/uncompiled forms."""
    return {k: v.detach().cpu().clone() for k, v in unwrap(model).state_dict().items()}


def set_weights(model: torch.nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    """Load weights into a model. Operates on the unwrapped module so
    callers don't need to know whether `model` is a `torch.compile`
    wrapper. Validates that all target keys are present in `weights`."""
    target_module = unwrap(model)
    target = target_module.state_dict()
    converted = {
        k: weights[k].to(dtype=target[k].dtype, device=target[k].device)
        for k in target.keys()
        if k in weights
    }
    missing = set(target.keys()) - set(converted.keys())
    if missing:
        raise KeyError(f"set_weights: missing keys: {sorted(missing)[:5]}…")
    target_module.load_state_dict(converted, strict=True)


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
    """Recompute BN running mean/variance to be the EXACT batch-statistics
    average over `num_batches` forwards (not an EMA approximation).

    Mirrors the technique in `torch.optim.swa_utils.update_bn`: setting
    `bn.momentum = None` makes BN use its `num_batches_tracked` counter to
    compute a true cumulative running average — `running = (n*running +
    batch_stat) / (n+1)` — which converges to the exact mean over the
    visited batches. Restores the original momentum at exit.

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
    saved_momentum = [bn.momentum for bn in bn_modules]
    for bn in bn_modules:
        bn.reset_running_stats()
        bn.momentum = None  # opt into exact-cumulative-average behavior
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            for i, batch in enumerate(ds):
                if i >= num_batches:
                    break
                input_board, input_global = batch[0], batch[1]
                if i == 0:
                    model.to(input_board.device)
                _ = model(input_board, input_global)
                if (i + 1) % 20 == 0:
                    print(
                        f"=== recompute_bn_statistics: Processed {i + 1}/{num_batches} batches ==="
                    )
    finally:
        for bn, mom in zip(bn_modules, saved_momentum):
            bn.momentum = mom
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


# Weight migration helpers (`migrate_from_keras` and friends) used to live
# here. They were unused by the active `scripts/migrate_keras_to_torch.py`,
# which carries its own complete set of per-layer copiers, so they were
# deleted rather than moved.
