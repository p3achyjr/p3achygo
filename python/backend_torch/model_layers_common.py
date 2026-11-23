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
    forwards (0 = no ramp). Created only by `enable_batch_renorm`.
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
            a = (
                (self.brn_t / self.ramp_batches).clamp(max=1.0)
                if self.ramp_batches > 0
                else torch.ones_like(self.brn_t)
            )
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
    module: nn.Module, rmax: float, dmax: float, ramp_batches: int
) -> int:
    """Switch every nn.BatchNorm2d under `module` to BatchRenorm2d in place.
    Checkpoint-compatible: no new persistent state. Returns the count."""
    n = 0
    for m in module.modules():
        if type(m) is nn.BatchNorm2d:
            m.__class__ = BatchRenorm2d
            m.rmax, m.dmax, m.ramp_batches = float(rmax), float(dmax), int(ramp_batches)
            m.register_buffer(
                "brn_t", torch.zeros((), device=m.running_mean.device), persistent=False
            )
            n += 1
    return n


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
