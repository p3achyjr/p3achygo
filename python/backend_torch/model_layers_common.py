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


def mish(x: torch.Tensor) -> torch.Tensor:
    # `F.mish` is mathematically `x * tanh(softplus(x))`, but `F.mish` is a
    # single aten op while the manual form decomposes through `softplus`,
    # which torch.export lowers to `where(x > 20, x, log1p(exp(x)))` — adding
    # 2 extra nodes per call (Greater + Where) to the exported ONNX. The
    # single-op form lowers to a single `Mish` ONNX node (opset ≥18) that
    # TRT handles with one fused kernel.
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
# ConvBlock (abstract)
# ---------------------------------------------------------------------------


class ConvPreActivation(nn.Module):
    """BN(in_ch) → Mish → Conv2d(in_ch→out_ch)  [pre-activation / keras default]."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.bn = make_bn(in_channels)  # normalises the *input* to the conv
        self.conv = make_conv(in_channels, out_channels, kernel_size, use_bias=False)

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
