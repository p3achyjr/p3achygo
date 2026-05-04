"""
PyTorch-native ConvMuon optimizer.

Matches KataGo / torch upstream momentum convention (EMA), diverging from
Keras ConvMuon which uses standard accumulation.

Key design choices:
  - Momentum: EMA  buf = buf*β + g*(1-β)  (matches KataGo, torch upstream)
  - Nesterov:  update = g*(1-β) + buf*β  (grad.lerp(buf, β))
  - 4D conv:   flatten [H, W, in, out] → [H*W*in, out] before NS
  - LR scale:  adj_lr = lr × sqrt(max(flat, out)) × rms_rate  (Moonlight)
  - WD:        per-param base WD set at construction, scaled by lr_ratio each step

Usage (train_torch.py):
    from optimizer_torch import ConvMuon
    opt = ConvMuon(model.named_parameters(), lr=..., ...)
    # LR updates:  opt.learning_rate = new_lr_or_schedule
"""

import math
import re

import torch
from torch import Tensor
from torch.optim import Optimizer


@torch.compile
def _newtonschulz(G: Tensor, steps: int) -> Tensor:
    """NS orthogonalization on a 2D tensor. Matches Keras ConvMuon coefficients."""
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


class ConvMuon(Optimizer):
    """
    PyTorch-native ConvMuon.

    exclude_layers patterns must use dot-separated PyTorch names, e.g.
    r".*policy_head.*" instead of r".*policy_head\\/.*".
    """

    _WD_SCALE = {
        "gamma": 0.1,
        "beta": 1e-3,
        "body_bias": 1e-2,
        "head_bias": 1e-2,
        "qkvo": 0.5,
    }

    def __init__(
        self,
        named_params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        adam_weight_decay: float = 0.004,
        adam_lr_ratio: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 6,
        rms_rate: float | None = 0.2,
        exclude_layers: list[str] | None = None,
        scale_weight_decay_by_rms: bool = False,
        wd_lr_exponent: float | None = None,
        wd_lr_max: float | None = None,
        global_clipnorm: float = 20.0,
    ):
        self._base_muon_wd = weight_decay
        self._base_adam_wd = adam_weight_decay
        self._adam_lr_ratio = adam_lr_ratio
        self._rms_rate = rms_rate
        self._scale_wd_by_rms = scale_weight_decay_by_rms
        self._wd_lr_exponent = wd_lr_exponent
        self._wd_lr_max = wd_lr_max
        self._global_clipnorm = global_clipnorm
        self._exclude_patterns = [re.compile(p) for p in (exclude_layers or [])]
        self._global_step = 0
        self._lr_schedule = None
        self.last_grad_norm: float = 0.0

        # Per-param base WD (before lr_scale and per-step rms_scale).
        # Keyed by id(param); populated in _build_groups.
        self._wd_base: dict[int, float] = {}

        groups = self._build_groups(named_params, lr, momentum, nesterov, ns_steps)
        super().__init__(groups, defaults={})

    # ------------------------------------------------------------------ #
    # Classification                                                       #
    # ------------------------------------------------------------------ #

    def _is_muon(self, name: str, param: Tensor) -> bool:
        if param.ndim < 2:
            return False
        if "embedding" in name.lower():
            return False
        if any(p.search(name) for p in self._exclude_patterns):
            return False
        out_dim = param.shape[-1]
        flat_dim = param.numel() // out_dim
        return out_dim > 4 and flat_dim > 4

    def _wd_category(self, name: str) -> str | None:
        n = name.lower()
        if n.endswith((".gamma", ".scale")):
            return "gamma"
        if n.endswith(".beta"):
            return "beta"
        if n.endswith(".bias"):
            return (
                "head_bias"
                if ("policy_head" in n or "value_head" in n)
                else "body_bias"
            )
        if "transformer_attention" in n and any(
            n.endswith(s)
            for s in (".query.kernel", ".key.kernel", ".value.kernel", ".output.kernel")
        ):
            return "qkvo"
        return None

    # ------------------------------------------------------------------ #
    # Group construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_groups(self, named_params, lr, momentum, nesterov, ns_steps):
        muon_params, adamw_params = [], []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            cat = self._wd_category(name)
            scale = self._WD_SCALE.get(cat)
            if self._is_muon(name, param):
                muon_params.append(param)
                self._wd_base[id(param)] = self._base_muon_wd * (scale or 1.0)
            else:
                adamw_params.append(param)
                self._wd_base[id(param)] = (
                    self._base_muon_wd * scale
                    if scale is not None
                    else self._base_adam_wd
                )

        groups = []
        if muon_params:
            groups.append(
                {
                    "params": muon_params,
                    "lr": lr,
                    "group": "muon",
                    "momentum": momentum,
                    "nesterov": nesterov,
                    "ns_steps": ns_steps,
                }
            )
        if adamw_params:
            groups.append(
                {
                    "params": adamw_params,
                    "lr": lr * self._adam_lr_ratio,
                    "group": "adamw",
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                }
            )
        return groups

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _lr_scale(self, lr: float) -> float:
        if self._wd_lr_exponent is None or self._wd_lr_max is None:
            return 1.0
        return min(lr / self._wd_lr_max, 1.0) ** self._wd_lr_exponent

    def _rms_wd_scale(self, param: Tensor) -> float:
        """Per-param RMS factor applied to WD when scale_weight_decay_by_rms=True."""
        if not self._scale_wd_by_rms or self._rms_rate is None:
            return 1.0
        out_dim = param.shape[-1]
        flat_dim = param.numel() // out_dim
        return math.sqrt(max(flat_dim, out_dim)) * self._rms_rate

    def _adjusted_lr(self, lr: float, flat_dim: int, out_dim: int) -> float:
        """Moonlight LR scaling: lr × sqrt(max(flat, out)) × rms_rate."""
        if self._rms_rate is None:
            return lr
        return lr * math.sqrt(max(flat_dim, out_dim)) * self._rms_rate

    # ------------------------------------------------------------------ #
    # LR interface (matches Keras ConvMuon attribute API)                #
    # ------------------------------------------------------------------ #

    @property
    def learning_rate(self) -> float:
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

        lr = self.param_groups[0]["lr"]
        lr_scale = self._lr_scale(lr)

        for group in self.param_groups:
            if group["group"] == "muon":
                self._muon_step(group, lr, lr_scale)
            else:
                self._adamw_step(group, group["lr"], lr_scale)

        return loss

    def _muon_step(self, group: dict, lr: float, lr_scale: float):
        momentum_coeff = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad

            state = self.state[param]
            if "buf" not in state:
                state["buf"] = torch.zeros_like(param)
            buf = state["buf"]

            # EMA momentum: buf = buf*β + g*(1-β)  (matches KataGo, torch upstream)
            buf.lerp_(grad, 1 - momentum_coeff)
            g = grad.lerp(buf, momentum_coeff) if nesterov else buf.clone()

            # Flatten 4D [H, W, in, out] → [H*W*in, out]
            original_shape = g.shape
            if g.ndim == 4:
                g = g.reshape(-1, original_shape[-1])
            flat_dim, out_dim = g.shape[0], g.shape[1]

            update = _newtonschulz(g, ns_steps)
            if update.shape != original_shape:
                update = update.reshape(original_shape)

            adj_lr = self._adjusted_lr(lr, flat_dim, out_dim)
            wd = self._wd_base[id(param)] * self._rms_wd_scale(param) * lr_scale

            param.mul_(1 - lr * wd)
            param.add_(update, alpha=-adj_lr)

    def _adamw_step(self, group: dict, lr: float, lr_scale: float):
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad

            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["m"] = torch.zeros_like(param)
                state["v"] = torch.zeros_like(param)
            state["step"] += 1
            t = state["step"]
            m, v = state["m"], state["v"]

            m.lerp_(grad, 1 - beta1)
            v.lerp_(grad.square(), 1 - beta2)
            bias_corr = math.sqrt(1 - beta2**t) / (1 - beta1**t)

            wd = self._wd_base[id(param)] * lr_scale
            param.mul_(1 - lr * wd)
            param.addcdiv_(m, v.sqrt().add_(eps), value=-lr * bias_corr)
