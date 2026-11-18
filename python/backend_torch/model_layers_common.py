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


# ---------------------------------------------------------------------------
# Fixed-scale normalization (KataGo `fixscaleonenorm`)
# ---------------------------------------------------------------------------

NORM_BN = "bnorm"  # BatchNorm everywhere (every net trained before 2026-11-18)
NORM_FIXSCALEONENORM = "fixscaleonenorm"
NORM_KINDS = (NORM_BN, NORM_FIXSCALEONENORM)

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
