"""
PyTorch-native ConvMuon optimizer.

Matches KataGo / torch upstream momentum convention (EMA), diverging from
Keras ConvMuon which uses standard accumulation.

Key design choices:
  - Momentum: EMA  buf = buf*β + g*(1-β)  (matches KataGo, torch upstream)
  - Nesterov:  update = g*(1-β) + buf*β  (grad.lerp(buf, β))
  - 4D conv:   flatten [H, W, in, out] → [H*W*in, out] before NS
  - LR scale:  adj_lr = lr × sqrt(max(flat, out)) × rms_rate  (Moonlight)
  - WD:        per-param (base_kind, factor) tag — base_kind ∈ {"muon","adam"}
               picks which scalar (`weight_decay` / `adam_weight_decay`) the
               factor multiplies, resolved at step time so the setters
               actually propagate.

Usage:
    groups, wd_factors = build_convmuon_param_groups(
        model, lr=1e-3, exclude_layers=[r".*policy_head.*"]
    )
    opt = ConvMuon(
        groups,
        wd_factors=wd_factors,
        weight_decay=0.1,
        adam_weight_decay=0.004,
        adam_lr_ratio=1.0,
    )
    opt.learning_rate = lr_schedule_or_float
    opt.weight_decay = new_muon_wd  # propagates on next step
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


# WD factor (relative to the base scalar named in the first slot)
# e.g. ("muon", 0.1) → wd = base_muon_wd * 0.1 * lr_scale
WdTag = Tuple[str, float]


# Per-category multipliers applied to the muon base WD. Mirrors keras
# ConvMuon._WD_SCALE_FACTORS exactly. Module-level so the helper can use
# it without importing the class; aliased onto ConvMuon below for callers
# that historically read `ConvMuon._WD_SCALE`.
_WD_SCALE = {
    "gamma": 0.1,
    "beta": 1e-3,
    "body_bias": 1e-2,
    "head_bias": 1e-2,
    "qkvo": 0.5,
}


@torch.compile
def _newtonschulz(G: Tensor, steps: int) -> Tensor:
    """NS orthogonalization on a 2D tensor. Matches Keras ConvMuon coefficients.

    Not used by `_muon_step` directly anymore (we always go through the
    batched variant for uniform handling). Kept for the parity test in
    `test_optimizer_torch.py` and as a reference implementation.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = torch.addmm(X, torch.addmm(A, A, A, beta=b, alpha=c), X, beta=a)
    if G.size(0) > G.size(1):
        X = X.mT
    return X.to(dtype=G.dtype)


@torch.compile(mode="reduce-overhead")
def _newtonschulz_batched(Gs: Tensor, steps: int) -> Tensor:
    """Batched NS5: input shape `(B, M, N)`, output same shape.

    Matches `_newtonschulz` element-wise in the batch dimension. All B
    tensors must share the same `(M, N)` (call once per shape group).
    Uses `bmm`/`baddbmm` so the inner matmul chain becomes one kernel
    launch per matmul *across all B params*, instead of B per matmul.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = Gs.bfloat16()
    if X.size(1) > X.size(2):
        X = X.mT  # (B, N, M)
    # Per-batch Frobenius norm (over the last two dims).
    norm = X.flatten(1).norm(dim=1).view(-1, 1, 1)  # (B, 1, 1)
    X = X / (norm + 1e-7)
    for _ in range(steps):
        A = torch.bmm(X, X.mT)  # (B, N, N)
        # baddbmm(mat=A, b1=A, b2=A, beta=b, alpha=c) → b*A + c*(A@A).
        inner = torch.baddbmm(A, A, A, beta=b, alpha=c)
        # baddbmm(mat=X, b1=inner, b2=X, beta=a, alpha=1) → a*X + inner @ X
        X = torch.baddbmm(X, inner, X, beta=a)
    if Gs.size(1) > Gs.size(2):
        X = X.mT
    return X.to(dtype=Gs.dtype)


# ---------------------------------------------------------------------------
# Param-group builder (free function — separates classification from update)
# ---------------------------------------------------------------------------


def _is_muon_param(name: str, param: Tensor, exclude_patterns) -> bool:
    if param.ndim < 2:
        return False
    if "embedding" in name.lower():
        return False
    if any(p.search(name) for p in exclude_patterns):
        return False
    if param.ndim == 4:
        # torch conv: (Cout, Cin, H, W). out=dim 0, flat=Cin*H*W.
        out_dim = param.shape[0]
    else:
        out_dim = param.shape[-1]
    flat_dim = param.numel() // out_dim
    return out_dim > 4 and flat_dim > 4


def _wd_category(name: str, is_bn: bool) -> Optional[str]:
    """Match keras' classifier on torch-style param names. See parity-gaps doc
    §A7 for the wrinkles around `transformer_attention.*.kernel` (torch will
    use `.weight` instead, so qkvo currently never matches in pure-torch)."""
    n = name.lower()
    if is_bn:
        if n.endswith(".weight"):
            return "gamma"
        if n.endswith(".bias"):
            return "beta"
    if n.endswith((".gamma", ".scale")):
        return "gamma"
    if n.endswith(".beta"):
        return "beta"
    if n.endswith(".bias"):
        return "head_bias" if ("policy_head" in n or "value_head" in n) else "body_bias"
    if "transformer_attention" in n and any(
        n.endswith(s)
        for s in (".query.kernel", ".key.kernel", ".value.kernel", ".output.kernel")
    ):
        return "qkvo"
    return None


def _ns_input_shape(p: Tensor) -> Tuple[int, int]:
    """The 2D shape NS5 sees after the keras-layout flatten.
    For 4D conv (Cout, Cin, H, W): (H*W*Cin, Cout).
    For 2D Linear (Cout, Cin): (Cout, Cin) — already 2D.
    """
    if p.ndim == 4:
        Cout, Cin, H, W = p.shape
        return (H * W * Cin, Cout)
    return tuple(p.shape)


def _to_keras_2d(g: Tensor) -> Tuple[Tensor, torch.Size]:
    """Flatten a Muon-path parameter gradient to the 2D shape NS5 expects.

    For 4D conv tensors (torch layout (Cout, Cin, H, W)), permute to keras
    layout (H, W, Cin, Cout) so the row-major reshape collapses (H, W, Cin)
    correctly. For 2D tensors no work is needed.

    Returns (g_2d, keras_intermediate_shape). The intermediate shape is
    needed by `_from_keras_2d` to reverse the permutation.
    """
    if g.ndim == 4:
        g = g.permute(2, 3, 1, 0).contiguous()  # (H, W, Cin, Cout)
        keras_shape = g.shape
        return g.reshape(-1, keras_shape[-1]), keras_shape
    # Non-4D: no permutation; keras_shape == g.shape.
    return g, g.shape


def _from_keras_2d(
    upd: Tensor, keras_shape: torch.Size, original_shape: torch.Size
) -> Tensor:
    """Inverse of `_to_keras_2d`. Restores the original (Cout, Cin, H, W) or
    (Cout, Cin) layout for `upd`, which came back from NS5 as 2D."""
    if len(original_shape) == 4:
        return upd.reshape(keras_shape).permute(3, 2, 0, 1).contiguous()
    if upd.shape != original_shape:
        return upd.reshape(original_shape)
    return upd


def _build_muon_ns_groups(muon_params: List[Tensor]) -> List[Dict]:
    """Bucket muon params by their NS-input shape.

    Returns a list of dicts:
        {"shape": (M, N), "indices": [...], "adj_lr_factor": float}
    where `indices` points into `muon_params` and `adj_lr_factor` is the
    LR-independent part of Moonlight scaling (`sqrt(max(M, N))`); the step
    just multiplies by the current `lr * rms_rate`.
    """
    shape_to_group: Dict[Tuple[int, int], int] = {}
    out: List[Dict] = []
    for i, p in enumerate(muon_params):
        s = _ns_input_shape(p)
        if s not in shape_to_group:
            shape_to_group[s] = len(out)
            out.append(
                {
                    "shape": s,
                    "indices": [],
                    "adj_lr_factor": math.sqrt(max(s[0], s[1])),
                }
            )
        out[shape_to_group[s]]["indices"].append(i)
    return out


def _collect_named_params(
    params_or_model,
) -> Tuple[List[Tuple[str, Tensor]], set]:
    """Accept an `nn.Module` or an iterable of (name, param) pairs.
    Returns (named_params, bn_param_ids)."""
    if isinstance(params_or_model, torch.nn.Module):
        named = list(params_or_model.named_parameters())
        bn_ids: set = set()
        for _, mod in params_or_model.named_modules():
            if isinstance(
                mod,
                (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
            ):
                if mod.weight is not None:
                    bn_ids.add(id(mod.weight))
                if mod.bias is not None:
                    bn_ids.add(id(mod.bias))
        return named, bn_ids
    return list(params_or_model), set()


def build_convmuon_param_groups(
    params_or_model,
    *,
    lr: float = 1e-3,
    adam_lr_ratio: float = 1.0,
    exclude_layers: Optional[List[str]] = None,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 6,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-7,
) -> Tuple[List[dict], Dict[int, WdTag]]:
    """Walk a torch model (or iterable of named parameters), classify each
    parameter into the muon or adamw update path, and compute its weight-decay
    *factor* relative to the optimizer's base scalars.

    Returns:
        param_groups: list of dicts ready to pass to `ConvMuon.__init__`.
                      Two groups: "muon" and "adamw" (either may be absent).
        wd_factors:   dict mapping `id(param) -> (base_kind, multiplier)`,
                      where `base_kind ∈ {"muon", "adam"}` selects which of
                      `weight_decay` / `adam_weight_decay` the multiplier
                      scales.

    Tags by case:
      - Categorized param (gamma/beta/body_bias/head_bias/qkvo): always
        ("muon", scale_factor) — keras applies the scaled WD against the
        muon base regardless of whether the param goes through the muon or
        adam update path.
      - Uncategorized muon-path param: ("muon", 1.0)
      - Uncategorized adam-path param: ("adam", 1.0)
    """
    exclude_patterns = [re.compile(p) for p in (exclude_layers or [])]
    named, bn_ids = _collect_named_params(params_or_model)

    muon_params, adamw_params = [], []
    wd_factors: Dict[int, WdTag] = {}

    for name, param in named:
        if not param.requires_grad:
            continue
        cat = _wd_category(name, is_bn=id(param) in bn_ids)
        scale = _WD_SCALE.get(cat) if cat else None
        if _is_muon_param(name, param, exclude_patterns):
            muon_params.append(param)
            wd_factors[id(param)] = ("muon", scale if scale is not None else 1.0)
        else:
            adamw_params.append(param)
            wd_factors[id(param)] = (
                ("muon", scale) if scale is not None else ("adam", 1.0)
            )

    groups: List[dict] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "lr": lr,
                "group": "muon",
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
                # Pre-bucket by NS-input shape so `_muon_step` can run a
                # batched NS5 per shape group instead of one NS5 per param.
                "ns_groups": _build_muon_ns_groups(muon_params),
            }
        )
    if adamw_params:
        groups.append(
            {
                "params": adamw_params,
                "lr": lr * adam_lr_ratio,
                "group": "adamw",
                "betas": adam_betas,
                "eps": adam_eps,
            }
        )
    return groups, wd_factors


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class ConvMuon(Optimizer):
    """
    PyTorch-native ConvMuon. Takes precomputed param groups built by
    `build_convmuon_param_groups`. Stores per-param `(base_kind, factor)`
    weight-decay tags so the `weight_decay` / `adam_weight_decay` setters
    propagate to the next step.

    `exclude_layers` is interpreted by `build_convmuon_param_groups`, NOT
    here — patterns that classify which update path a param takes belong
    on the helper.
    """

    # Alias the module-level table so existing test/callers that look up
    # `ConvMuon._WD_SCALE["gamma"]` keep working.
    _WD_SCALE = _WD_SCALE

    def __init__(
        self,
        param_groups: List[dict],
        *,
        wd_factors: Optional[Dict[int, WdTag]] = None,
        weight_decay: float = 0.1,
        adam_weight_decay: float = 0.004,
        adam_lr_ratio: float = 1.0,
        rms_rate: Optional[float] = 0.2,
        scale_weight_decay_by_rms: bool = False,
        wd_lr_exponent: Optional[float] = None,
        wd_lr_max: Optional[float] = None,
        global_clipnorm: float = float("inf"),
    ):
        self._base_muon_wd = weight_decay
        self._base_adam_wd = adam_weight_decay
        self._adam_lr_ratio = adam_lr_ratio
        self._rms_rate = rms_rate
        self._scale_wd_by_rms = scale_weight_decay_by_rms
        self._wd_lr_exponent = wd_lr_exponent
        self._wd_lr_max = wd_lr_max
        self._global_clipnorm = global_clipnorm
        self._wd_factors: Dict[int, WdTag] = dict(wd_factors or {})
        self._global_step = 0
        self._lr_schedule = None
        self.last_grad_norm: float = 0.0

        # Default any unmapped param to ("muon", 1.0) so out-of-band uses
        # (e.g. unit tests that pass a single Parameter without going through
        # the helper) keep working.
        for g in param_groups:
            for p in g["params"]:
                self._wd_factors.setdefault(id(p), ("muon", 1.0))

        super().__init__(param_groups, defaults={})

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _wd_for(self, param: Tensor) -> float:
        kind, factor = self._wd_factors[id(param)]
        base = self._base_muon_wd if kind == "muon" else self._base_adam_wd
        return base * factor

    def _lr_scale(self, lr: float) -> float:
        if self._wd_lr_exponent is None or self._wd_lr_max is None:
            return 1.0
        return min(lr / self._wd_lr_max, 1.0) ** self._wd_lr_exponent

    def _rms_wd_scale(self, param: Tensor) -> float:
        if not self._scale_wd_by_rms or self._rms_rate is None:
            return 1.0
        out_dim = param.shape[0] if param.ndim == 4 else param.shape[-1]
        flat_dim = param.numel() // out_dim
        return math.sqrt(max(flat_dim, out_dim)) * self._rms_rate

    def _adjusted_lr(self, lr: float, flat_dim: int, out_dim: int) -> float:
        """Moonlight LR scaling: lr × sqrt(max(flat, out)) × rms_rate."""
        if self._rms_rate is None:
            return lr
        return lr * math.sqrt(max(flat_dim, out_dim)) * self._rms_rate

    # ------------------------------------------------------------------ #
    # Keras-compatible LR / WD interface                                 #
    # ------------------------------------------------------------------ #

    @property
    def learning_rate(self) -> float:
        for g in self.param_groups:
            if g.get("group") == "muon":
                return g["lr"]
        return self.param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, value):
        self._lr_schedule = value if callable(value) else None
        lr = float(value(0)) if callable(value) else float(value)
        for group in self.param_groups:
            group["lr"] = lr if group["group"] == "muon" else lr * self._adam_lr_ratio

    @property
    def weight_decay(self) -> float:
        return self._base_muon_wd

    @weight_decay.setter
    def weight_decay(self, value: float):
        self._base_muon_wd = float(value)

    @property
    def adam_weight_decay(self) -> float:
        return self._base_adam_wd

    @adam_weight_decay.setter
    def adam_weight_decay(self, value: float):
        self._base_adam_wd = float(value)

    @property
    def adam_lr_ratio(self) -> float:
        return self._adam_lr_ratio

    @adam_lr_ratio.setter
    def adam_lr_ratio(self, value: float):
        self._adam_lr_ratio = float(value)
        # Re-derive adamw group LR from current muon LR.
        muon_lr = self.learning_rate
        for g in self.param_groups:
            if g.get("group") == "adamw":
                g["lr"] = muon_lr * self._adam_lr_ratio

    @property
    def global_clipnorm(self) -> float:
        return self._global_clipnorm

    @global_clipnorm.setter
    def global_clipnorm(self, value: float):
        self._global_clipnorm = float(value)

    @property
    def wd_lr_exponent(self):
        return self._wd_lr_exponent

    @wd_lr_exponent.setter
    def wd_lr_exponent(self, value):
        self._wd_lr_exponent = value

    @property
    def wd_lr_max(self):
        return self._wd_lr_max

    @wd_lr_max.setter
    def wd_lr_max(self, value):
        self._wd_lr_max = value

    @property
    def momentum(self) -> float:
        for g in self.param_groups:
            if g.get("group") == "muon":
                return g["momentum"]
        return 0.0

    def apply(self, grads=None, variables=None):
        """Keras-compatible interface. After backward(), call apply() with no args.
        Optionally accepts explicit grads and assigns them before stepping."""
        if grads is not None and variables is not None:
            for g, v in zip(grads, variables):
                if g is not None:
                    v.grad = g
        self.step()

    # ------------------------------------------------------------------ #
    # Step                                                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._lr_schedule is not None:
            lr = float(self._lr_schedule(self._global_step))
            for group in self.param_groups:
                group["lr"] = (
                    lr if group["group"] == "muon" else lr * self._adam_lr_ratio
                )
        self._global_step += 1

        all_params = [
            p for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        if self._global_clipnorm and all_params:
            self.last_grad_norm = torch.nn.utils.clip_grad_norm_(
                all_params, self._global_clipnorm
            ).item()
        else:
            self.last_grad_norm = 0.0

        # Use the muon group's lr for `_lr_scale` (sublinear WD-decay schedule
        # is keyed off the *base* lr, matching keras `_lr_scale`).
        muon_lr = self.learning_rate
        lr_scale = self._lr_scale(muon_lr)

        for group in self.param_groups:
            if group["group"] == "muon":
                self._muon_step(group, group["lr"], lr_scale)
            else:
                self._adamw_step(group, group["lr"], lr_scale)

        return loss

    def _muon_step(self, group: dict, lr: float, lr_scale: float):
        """Multi-tensor Muon update.

        - Pre-NS elementwise uses `_foreach_lerp_`.
        - NS5 is batched across same-shape params via `_newtonschulz_batched`
          (one matmul launch per matmul step *across all params in a shape
          bucket*, instead of one per param).
        - Post-NS update (mul_/add_) uses `_foreach_*`.

        Bucket assignment is precomputed in `build_convmuon_param_groups`
        and stored on `group["ns_groups"]` so the hot path does no Python
        grouping work per step.
        """
        momentum_coeff = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        ns_groups = group["ns_groups"]

        # Fast path: when all params have gradients (the common case —
        # full-loss training), skip the filter/remap dance.
        all_grads = [p.grad for p in group["params"]]
        if all(g is not None for g in all_grads):
            params = group["params"]
            grads = all_grads
            orig_to_filt = None  # signals fast path
        else:
            # None-grad params (e.g. loss subset that doesn't touch every head)
            # are skipped entirely — neither the Muon update nor the weight-decay
            # applies, matching torch.optim.Adam convention.
            active_orig_idxs = [i for i, g in enumerate(all_grads) if g is not None]
            if not active_orig_idxs:
                return
            params = [group["params"][i] for i in active_orig_idxs]
            grads = [all_grads[i] for i in active_orig_idxs]
            orig_to_filt = {o: f for f, o in enumerate(active_orig_idxs)}

        # Lazy state init (per-param, only on first step).
        bufs = []
        for p in params:
            state = self.state[p]
            if "buf" not in state:
                state["buf"] = torch.zeros_like(p)
            bufs.append(state["buf"])

        # buf = buf*β + g*(1-β)  [EMA momentum]
        torch._foreach_lerp_(bufs, grads, 1 - momentum_coeff)
        if nesterov:
            update_gs = list(torch._foreach_lerp(grads, bufs, momentum_coeff))
        else:
            update_gs = [b.clone() for b in bufs]

        # Batched NS5 per shape bucket. Singletons go through the same
        # batched path with B=1 — slight per-call overhead, but uniform code
        # and they get the CUDA-graph win for free.
        updates: List[Optional[Tensor]] = [None] * len(params)
        adj_lrs: List[float] = [0.0] * len(params)
        adj_lr_scalar = lr * (self._rms_rate if self._rms_rate is not None else 1.0)
        for bucket in ns_groups:
            # Fast path: all params active → bucket indices ARE filtered indices.
            if orig_to_filt is None:
                filt_idxs = bucket["indices"]
            else:
                filt_idxs = [
                    orig_to_filt[i] for i in bucket["indices"] if i in orig_to_filt
                ]
                if not filt_idxs:
                    continue
            stack_inputs: List[Tensor] = []
            originals: List[torch.Size] = []
            keras_shapes: List[torch.Size] = []
            for fi in filt_idxs:
                g_2d, keras_shape = _to_keras_2d(update_gs[fi])
                stack_inputs.append(g_2d)
                originals.append(update_gs[fi].shape)
                keras_shapes.append(keras_shape)
            stacked = torch.stack(stack_inputs, dim=0)
            # `.clone()` is required because reduce-overhead mode reuses the
            # output buffer across calls; without it, bucket N+1 silently
            # overwrites bucket N's output.
            stacked_upd = _newtonschulz_batched(stacked, ns_steps).clone()

            # Same per-bucket adj_lr (since adj_lr depends only on shape).
            bucket_adj_lr = (
                adj_lr_scalar * bucket["adj_lr_factor"]
                if self._rms_rate is not None
                else lr
            )
            for batch_i, fi in enumerate(filt_idxs):
                updates[fi] = _from_keras_2d(
                    stacked_upd[batch_i], keras_shapes[batch_i], originals[batch_i]
                )
                adj_lrs[fi] = bucket_adj_lr

        # Per-param weight-decay factor: depends on (kind, factor) tag and
        # rms-by-wd toggle. Only `lr_scale` and `_base_*_wd` change between
        # steps, so all the param-shape-dependent work is already cached.
        wd_factors_list = [
            self._wd_for(p) * self._rms_wd_scale(p) * lr_scale for p in params
        ]

        torch._foreach_mul_(params, [1.0 - lr * wd for wd in wd_factors_list])
        # `_foreach_add_` takes only a scalar alpha; pre-scale updates in place.
        torch._foreach_mul_(updates, [-a for a in adj_lrs])
        torch._foreach_add_(params, updates)

    def _adamw_step(self, group: dict, lr: float, lr_scale: float):
        """Multi-tensor AdamW update via `torch._foreach_*`."""
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        grads = [p.grad for p in params]

        # Lazy state init.
        ms, vs = [], []
        # All adamw params in one group share a single step counter.
        if "step" not in self.state[params[0]]:
            for p in params:
                self.state[p]["step"] = 0
                self.state[p]["m"] = torch.zeros_like(p)
                self.state[p]["v"] = torch.zeros_like(p)
        for p in params:
            self.state[p]["step"] += 1
            ms.append(self.state[p]["m"])
            vs.append(self.state[p]["v"])
        t = self.state[params[0]]["step"]
        bias_corr = math.sqrt(1 - beta2**t) / (1 - beta1**t)

        # m = m*β1 + g*(1-β1)
        torch._foreach_lerp_(ms, grads, 1 - beta1)
        # v = v*β2 + g²*(1-β2)
        grad_sq = torch._foreach_pow(grads, 2)
        torch._foreach_lerp_(vs, grad_sq, 1 - beta2)

        # param *= (1 - lr * wd)  per-param scalar
        wd_factors_list = [self._wd_for(p) * lr_scale for p in params]
        torch._foreach_mul_(params, [1.0 - lr * wd for wd in wd_factors_list])

        # param += -lr * bias_corr * m / (sqrt(v) + eps)
        denom = torch._foreach_sqrt(vs)
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(params, ms, denom, value=-lr * bias_corr)


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

_MUON_EXCLUDE_LAYERS_TORCH = [r".*policy_head.*", r".*value_head.*"]


def make_optimizer(
    model, config, lr_schedule, is_gpu, optimizer=None, optimizer_state=None
):
    """Build (when `optimizer is None`) or hot-reload the torch optimizer.

    Returns a torch-native optimizer. Mixed-precision wrapping (LSO/GradScaler)
    is NOT applied here — see context/torch_parity_gaps.md §B1; the keras-on-
    torch train_step uses a `hasattr(opt, 'scale_loss')` guard, so an
    unwrapped torch optimizer naturally takes the fp32 path.

    `optimizer_state` (torch backend): a state_dict produced by a prior
    `optimizer.state_dict()` and serialized to .pt. When `optimizer is None`
    and `optimizer_state` is provided, build fresh and call `load_state_dict`.

    SGD on torch backend is not wired yet — raises NotImplementedError.
    """
    del is_gpu  # unused; mixed precision is not configured per-optimizer here

    if optimizer is None:
        if config.optimizer == "muon":
            seed_lr = (
                float(lr_schedule(0)) if callable(lr_schedule) else float(lr_schedule)
            )
            groups, wd_factors = build_convmuon_param_groups(
                model,
                lr=seed_lr,
                adam_lr_ratio=config.adam_lr_ratio,
                exclude_layers=_MUON_EXCLUDE_LAYERS_TORCH,
            )
            optimizer = ConvMuon(
                groups,
                wd_factors=wd_factors,
                weight_decay=config.muon_wd,
                adam_weight_decay=config.adam_wd,
                adam_lr_ratio=config.adam_lr_ratio,
                wd_lr_exponent=config.wd_lr_exponent,
                wd_lr_max=config.wd_lr_max,
                global_clipnorm=config.global_clipnorm,
            )
            optimizer.learning_rate = lr_schedule  # store the schedule callable
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
        else:
            raise NotImplementedError(
                "SGD on torch backend not wired. Use config.optimizer='muon' "
                "or run on the keras (TF) backend."
            )
    else:
        # Hot-reload config-driven fields. Parity with TF backend's hot-reload.
        optimizer.learning_rate = lr_schedule
        if isinstance(optimizer, ConvMuon):
            optimizer.weight_decay = config.muon_wd
            optimizer.adam_weight_decay = config.adam_wd
            optimizer.adam_lr_ratio = config.adam_lr_ratio
            optimizer.wd_lr_exponent = config.wd_lr_exponent
            optimizer.wd_lr_max = config.wd_lr_max
        optimizer.global_clipnorm = config.global_clipnorm

    return optimizer
