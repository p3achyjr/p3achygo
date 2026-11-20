"""Torch-native basic building blocks for P3achyGoModel.

All tensors use NCHW shape semantics with channels_last memory format for
cuDNN NHWC kernel performance. Call `model.to(memory_format=torch.channels_last)`
after construction and `input.contiguous(memory_format=torch.channels_last)` on
every (N, C, H, W) tensor before the first layer.

BatchNorm: keras uses `momentum` as the EMA *retention* rate (0.99 = keep 99%
of running stats). Torch uses the *update* rate, so we set momentum=0.01.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@torch._dynamo.disable
def mish(x: torch.Tensor) -> torch.Tensor:
    # `F.mish` is mathematically `x * tanh(softplus(x))`, but `F.mish` is a
    # single aten op while the manual form decomposes through `softplus`,
    # which torch.export lowers to `where(x > 20, x, log1p(exp(x)))` — adding
    # 2 extra nodes per call (Greater + Where) to the exported ONNX. The
    # single-op form lowers to a single `Mish` ONNX node (opset ≥18) that
    # TRT handles with one fused kernel.
    #
    # `@torch._dynamo.disable` forces this op to run in eager mode even when
    # the surrounding model is `torch.compile`d. Required because Inductor's
    # lowering of mish backward under fp16/bf16 autocast produces NaN
    # gradients (verified on torch 2.11+cu13). Without this, every trunk
    # parameter receives NaN grad on every step and weights never update.
    # See .claude_scripts/diag_mish_ablation.py for the repro.
    return F.mish(x)


def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    use_bias: bool = False,
) -> nn.Conv2d:
    """Padding='same' equivalent: symmetric padding to keep H×W unchanged."""
    pad = kernel_size // 2
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=pad,
        bias=use_bias,
    )


def make_dense(in_features: int, out_features: int, use_bias: bool = True) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=use_bias)


def make_bn(num_features: int) -> nn.BatchNorm2d:
    """BatchNorm with keras-equivalent defaults.

    keras: momentum=0.99 (retain), epsilon=1e-3.
    torch: momentum=0.01 (update = 1 - retain), eps=1e-3.
    """
    return nn.BatchNorm2d(num_features, momentum=0.01, eps=1e-3)


class BatchRenorm2d(nn.BatchNorm2d):
    """Batch Renormalization (Ioffe 2017). Same parameters, buffers,
    state_dict and eval forward as nn.BatchNorm2d; only the training forward
    differs: y = gamma * ((x - mu_B) / sigma_B * r + d) + beta with
    r = sg(sigma_B / sigma) clipped to [1/rmax, rmax] and
    d = sg((mu_B - mu) / sigma) clipped to [-dmax, dmax], so the loss sees the
    running-stat function while gradients still flow through batch stats.
    rmax / dmax ramp linearly from 1 / 0 over `ramp_batches` training
    forwards (0 = no ramp); with `ramp_exp` they instead approach their targets
    as 1 - exp(-t / ramp_batches) (KataGo train.py brenorm adjustment).
    `brn_t` counts training forwards; `set_brn_step` resets it, e.g. to the
    global batch number on restart. Created only by `enable_batch_renorm`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # momentum None = cumulative stat recompute (SWA BN recompute): plain BN.
        if not self.training or self.momentum is None:
            return super().forward(x)
        xf = x.float()
        mean = xf.mean((0, 2, 3))
        var = xf.var((0, 2, 3), unbiased=False)
        std = (var + self.eps).sqrt()
        with torch.no_grad():
            rstd = (self.running_var + self.eps).sqrt()
            self.brn_t.add_(1.0)
            if self.ramp_batches <= 0:
                a = torch.ones_like(self.brn_t)
            elif self.ramp_exp:
                a = 1.0 - torch.exp(-self.brn_t / self.ramp_batches)
            else:
                a = (self.brn_t / self.ramp_batches).clamp(max=1.0)
            rmax = 1.0 + (self.rmax - 1.0) * a
            dmax = self.dmax * a
            r = torch.minimum(torch.maximum(std / rstd, 1.0 / rmax), rmax)
            d = torch.minimum(
                torch.maximum((mean - self.running_mean) / rstd, -dmax), dmax
            )
            n = x.numel() / x.shape[1]
            self.running_mean.lerp_(mean, self.momentum)
            self.running_var.lerp_(var * n / (n - 1), self.momentum)
            self.num_batches_tracked.add_(1)
        scale = self.weight * r / std
        shift = self.bias + self.weight * (d - mean * r / std)
        return (xf * scale[None, :, None, None] + shift[None, :, None, None]).to(
            x.dtype
        )


def enable_batch_renorm(
    module: nn.Module,
    rmax: float,
    dmax: float,
    ramp_batches: int,
    ramp_exp: bool = False,
) -> int:
    """Switch every nn.BatchNorm2d under `module` to BatchRenorm2d in place.
    Checkpoint-compatible: no new persistent state. Returns the count."""
    n = 0
    for m in module.modules():
        if type(m) is nn.BatchNorm2d:
            m.__class__ = BatchRenorm2d
            m.rmax, m.dmax, m.ramp_batches = float(rmax), float(dmax), int(ramp_batches)
            m.ramp_exp = bool(ramp_exp)
            m.register_buffer(
                "brn_t", torch.zeros((), device=m.running_mean.device), persistent=False
            )
            n += 1
    return n


@torch.no_grad()
def set_brn_step(module: nn.Module, step: int) -> int:
    """Set the ramp counter of every BatchRenorm2d under `module` to `step`
    (in place, so compiled graphs keep reading the same buffer)."""
    n = 0
    for m in module.modules():
        if isinstance(m, BatchRenorm2d):
            m.brn_t.fill_(float(step))
            n += 1
    return n


@torch.no_grad()
def set_bn_eps(module: nn.Module, eps: float, opt_state=None) -> int:
    """Set eps of every BatchNorm2d under `module` to `eps` without changing
    the eval function: gamma *= s, s = sqrt((var + eps) / (var + eps_old)).
    The training-time normalized input scales by 1/s, so the gamma gradient
    scales by 1/s; `opt_state` (optimizer.state, keyed by param) gets its
    gamma moments rescaled to match (adam path m /= s, v /= s^2; muon buf /= s).
    Returns the number of BNs whose eps changed."""
    n = 0
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) and m.eps != eps:
            s = ((m.running_var + eps) / (m.running_var + m.eps)).sqrt()
            m.weight.mul_(s)
            st = opt_state.get(m.weight) if opt_state is not None else None
            if st:
                if "m" in st:
                    st["m"].div_(s)
                if "v" in st:
                    st["v"].div_(s * s)
                if "buf" in st:
                    st["buf"].div_(s)
            m.eps = eps
            n += 1
    return n


# ---------------------------------------------------------------------------
# Fixed-scale normalization (KataGo `fixscaleonenorm`)
# ---------------------------------------------------------------------------

NORM_BN = "bnorm"  # BatchNorm everywhere (every net trained before 2026-11-18)
NORM_FIXSCALEONENORM = "fixscaleonenorm"
# A BatchNorm net converted after training (scripts/fold_bn_to_fixscale.py): every
# trunk norm is a fixed-scale norm WITH per-channel gamma (so the eval-mode BN
# affine folds into it exactly), no trunk-final BN, heads keep BatchNorm.
NORM_FIXSCALE_FOLDED = "fixscalefolded"
NORM_KINDS = (NORM_BN, NORM_FIXSCALEONENORM, NORM_FIXSCALE_FOLDED)

# KataGo `compute_gain("mish")`: std gain that keeps activations at unit variance
# through mish for a variance-preserving linear layer.
MISH_GAIN = math.sqrt(2.210277)


class FixScaleNorm(nn.Module):
    """KataGo `NormMask` for norm_kind fixscaleonenorm (every norm except the trunk-final one).

    No batch statistics: y = x * (gamma * scale) + beta, or x * scale + beta
    without gamma. `scale` is a fixed constant set at init (what BatchNorm
    would apply if every input had unit variance), stored as a buffer so
    it is checkpointed and exported as an ONNX constant.
    """

    def __init__(self, num_features: int, use_gamma: bool = False):
        super().__init__()
        self.num_features = num_features
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.ones(num_features)) if use_gamma else None
        self.register_buffer("scale", torch.tensor(1.0))

    def set_scale(self, scale: float) -> None:
        self.scale.fill_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma is not None:
            mul = (self.gamma * self.scale)[None, :, None, None]
        else:
            mul = self.scale
        return x * mul + self.beta[None, :, None, None]


class FixScaleBatchNorm(nn.BatchNorm2d):
    """KataGo `norm_trunkfinal` for fixscaleonenorm: BatchNorm with no learnable
    gamma (fixup_use_gamma=False). y = BN(x) * scale + beta, with `scale` a fixed
    buffer set at init and `beta` learnable. Same eps/momentum as `make_bn`."""

    def __init__(self, num_features: int):
        super().__init__(num_features, momentum=0.01, eps=1e-3, affine=False)
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("scale", torch.tensor(1.0))

    def set_scale(self, scale: float) -> None:
        self.scale.fill_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self.scale + self.beta[None, :, None, None]


def make_norm(
    num_features: int, norm_kind: str = NORM_BN, use_gamma: bool = False
) -> nn.Module:
    """BatchNorm for `bnorm` (the default; identical to `make_bn`), else a
    fixed-scale norm. `use_gamma` follows KataGo's `fixup_use_gamma` and only
    applies to fixed-scale norms (BatchNorm always has its affine weight)."""
    if norm_kind == NORM_BN:
        return make_bn(num_features)
    if norm_kind == NORM_FIXSCALEONENORM:
        return FixScaleNorm(num_features, use_gamma=use_gamma)
    if norm_kind == NORM_FIXSCALE_FOLDED:
        return FixScaleNorm(num_features, use_gamma=True)
    raise ValueError(f"Unknown norm_kind: {norm_kind!r}")


@torch.no_grad()
def init_weights_katago(tensor: torch.Tensor, gain: float, scale: float) -> None:
    """KataGo `init_weights`: truncated normal with std = scale * gain / sqrt(fan_in)."""
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    target_std = scale * gain / math.sqrt(fan_in)
    std = target_std / 0.87962566103423978  # truncation at ±2 std shrinks the std
    if std < 1e-10:
        tensor.fill_(0.0)
    else:
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


# ---------------------------------------------------------------------------
# ConvBlock (abstract)
# ---------------------------------------------------------------------------


class ConvPreActivation(nn.Module):
    """BN(in_ch) → Mish → Conv2d(in_ch→out_ch)  [pre-activation / keras default]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_kind: str = NORM_BN,
        use_gamma: bool = False,
    ):
        super().__init__()
        # normalises the *input* to the conv
        self.bn = make_norm(in_channels, norm_kind, use_gamma=use_gamma)
        self.conv = make_conv(in_channels, out_channels, kernel_size, use_bias=False)

    def init_fixscale(self, norm_scale: float) -> None:
        """KataGo NormActConv.initialize(scale=1.0, norm_scale=norm_scale)."""
        self.bn.set_scale(norm_scale)
        init_weights_katago(self.conv.weight, MISH_GAIN, 1.0)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        return self.conv(mish(self.bn(x)))


class ConvPostActivation(nn.Module):
    """Conv2d(in_ch→out_ch) → BN(out_ch) → Mish."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = make_conv(in_channels, out_channels, kernel_size, use_bias=False)
        self.bn = make_bn(out_channels)  # normalises the *output* of the conv

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        return mish(self.bn(self.conv(x)))


def make_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    pre_activation: bool = True,
) -> nn.Module:
    cls = ConvPreActivation if pre_activation else ConvPostActivation
    return cls(in_channels, out_channels, kernel_size)
