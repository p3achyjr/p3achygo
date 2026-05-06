"""
Torch-native transformer block for Go.

Port of `python/model_transformer.py` (keras). Same semantics:
- 2D spiral RoPE (paired-element rotations along 4 directions)
- Multi-head attention via `F.scaled_dot_product_attention` (FA-2 on sm_89+)
- SwiGLU FFN
- RMSNorm

Tensor conventions:
- Block input/output: NCHW `(N, C, H, W)` (matches the rest of `backend_torch/model.py`).
- Internally reshapes to `(N, S=H*W, C)` for attention/FFN.
- Q/K/V are reshaped to `(N, S, num_heads, head_dim)` (keras layout) and
  transposed to `(N, num_heads, S, head_dim)` immediately before SDPA, which
  is what torch's `scaled_dot_product_attention` expects.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 100.0 is appropriate for a 19x19 board (must be > 2 * pos_len to avoid aliasing).
ROPE_THETA = 100.0


def spiral_rope_cos_sin_table(
    num_rotations: int, embed_dim: int, grid_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Same construction as keras side. Numpy in / numpy out — pure math."""
    assert embed_dim % (num_rotations * 2) == 0
    assert embed_dim % 4 == 0
    elems_per_rotation = embed_dim // num_rotations
    seq_len = grid_len * grid_len

    t = np.arange(embed_dim // 4)
    thetas = ROPE_THETA ** (-t / (embed_dim // 4))
    theta_table = np.zeros((embed_dim,))
    for i in range(embed_dim):
        K = num_rotations
        k = i // elems_per_rotation
        k_norm = k % (K // 2)
        theta_base = 2 * k_norm
        rot_elem_offset = i % elems_per_rotation
        rot_offset = rot_elem_offset // 2
        theta_offset_base = (rot_offset // 2) * K
        theta_offset = theta_offset_base + (rot_offset % 2)
        theta_idx = min(len(thetas) - 1, theta_base + theta_offset)
        theta_table[i] = thetas[theta_idx]

    angles = np.arange(num_rotations) * (np.pi / num_rotations)
    x_coords = np.arange(grid_len)
    y_coords = np.arange(grid_len)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    angle_projs = np.zeros((seq_len, embed_dim))
    for d in range(embed_dim):
        angle_idx = d // elems_per_rotation
        angle = angles[angle_idx]
        angle_projs[:, d] = x_flat * np.cos(angle) + y_flat * np.sin(angle)

    rot_table = theta_table * angle_projs
    return np.cos(rot_table), np.sin(rot_table)


class RoPE(nn.Module):
    """Apply spiral RoPE to (B, S, num_heads, head_dim) input.

    Buffers (`_rope_cos`, `_rope_sin`, `_pair_swap_indices`, `_sign_cos`)
    are registered as fp32 and cast to the input dtype at call time —
    matches keras impl which also casts per-call.
    """

    def __init__(self, pos_len: int, head_dim: int, num_rotations: int):
        super().__init__()
        self.seq_len = pos_len * pos_len
        self.pos_len = pos_len
        self.head_dim = head_dim
        self.num_rotations = num_rotations

        cos, sin = spiral_rope_cos_sin_table(num_rotations, head_dim, pos_len)
        self.register_buffer("_rope_cos", torch.tensor(cos, dtype=torch.float32))
        self.register_buffer("_rope_sin", torch.tensor(sin, dtype=torch.float32))

        # Pair swap: (x0, x1, x2, x3, ...) → (x1, x0, x3, x2, ...)
        pair_swap = torch.empty(head_dim, dtype=torch.long)
        for i in range(head_dim // 2):
            pair_swap[2 * i] = 2 * i + 1
            pair_swap[2 * i + 1] = 2 * i
        self.register_buffer("_pair_swap_indices", pair_swap)

        # Sign pattern for cos: +1 on even indices, -1 on odd indices.
        sign_cos = torch.ones(head_dim, dtype=torch.float32)
        for i in range(head_dim // 2):
            sign_cos[2 * i + 1] = -1.0
        self.register_buffer("_sign_cos", sign_cos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, num_heads, head_dim)
        dtype = x.dtype
        cos = self._rope_cos.to(dtype).view(1, self.seq_len, 1, self.head_dim)
        sin = self._rope_sin.to(dtype).view(1, self.seq_len, 1, self.head_dim)
        sign_cos = self._sign_cos.to(dtype)

        # Swap pairs along last axis via gather.
        x_swapped = x.index_select(-1, self._pair_swap_indices)

        # x' = x * cos * sign_cos + x_swapped * sin
        return x * cos * sign_cos + x_swapped * sin


class TransformerAttention(nn.Module):
    """RMSNorm → Q/K/V → RoPE → SDPA → O. Input/output: (B, S, embed_dim)."""

    def __init__(self, embed_dim: int, num_heads: int, pos_len: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.pos_len = pos_len
        self.seq_len = pos_len * pos_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.rms = nn.RMSNorm(embed_dim)
        self.rope = RoPE(pos_len=pos_len, head_dim=self.head_dim, num_rotations=4)
        self.Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.O = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, embed_dim). Use -1 for the batch dim in every reshape so
        # torch.onnx emits Reshape with a constant shape tensor instead of
        # rebuilding it from Shape→Gather→Concat at runtime.
        x = self.rms(x)

        q = self.Q(x).view(-1, self.seq_len, self.num_heads, self.head_dim)
        k = self.K(x).view(-1, self.seq_len, self.num_heads, self.head_dim)
        v = self.V(x).view(-1, self.seq_len, self.num_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        # SDPA expects (B, H, S, D); we have (B, S, H, D).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Routes to FlashAttention-2 on sm_89 fp16/bf16, head_dim ≤ 256.
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # (B, H, S, D) → (B, S, H, D) → (B, S, embed_dim)
        out = out.transpose(1, 2).contiguous().view(-1, self.seq_len, self.embed_dim)
        return self.O(out)


class TransformerFFN(nn.Module):
    """RMSNorm → SwiGLU. Input/output: (B, S, embed_dim)."""

    def __init__(self, embed_dim: int, num_heads: int, pos_len: int):
        super().__init__()
        del num_heads, pos_len  # unused; kept for signature parity with keras
        self.embed_dim = embed_dim
        self.ffn_dim = 2 * embed_dim
        self.rms = nn.RMSNorm(embed_dim)
        self.ffn_gate = nn.Linear(embed_dim, self.ffn_dim, bias=False)
        self.ffn_up = nn.Linear(embed_dim, self.ffn_dim, bias=False)
        self.ffn_down = nn.Linear(self.ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rms(x)
        return self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))


class TransformerResidualBlock(nn.Module):
    """NCHW in / NCHW out. Two pre-norm residual connections (attn, ffn)."""

    def __init__(self, embed_dim: int, num_heads: int, pos_len: int):
        super().__init__()
        self.pos_len = pos_len
        self.seq_len = pos_len * pos_len
        self.embed_dim = embed_dim
        self.transformer_attn = TransformerAttention(embed_dim, num_heads, pos_len)
        self.transformer_ffn = TransformerFFN(embed_dim, num_heads, pos_len)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        # x: (N, C, H, W) → (N, S, C) for attn/FFN; back to NCHW on exit.
        # All non-batch dims are baked in at module construction time; using
        # them directly (instead of x.shape[1:]) keeps the exported Reshape
        # static and avoids Shape→Gather→Concat dynamic-shape arithmetic.
        del training  # unused; kept for trunk-loop signature parity
        flat = x.permute(0, 2, 3, 1).reshape(-1, self.seq_len, self.embed_dim)
        flat = flat + self.transformer_attn(flat)
        flat = flat + self.transformer_ffn(flat)
        return (
            flat.view(-1, self.pos_len, self.pos_len, self.embed_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class TransformerBottleneckBlock(nn.Module):
    """conv_down → N transformer blocks → conv_up, with outer residual.

    `input_channels` and `output_dim` should be equal so the residual works.
    `embed_dim` is the per-token width inside the transformer stack.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int,
        pos_len: int,
        num_blocks: int = 2,
    ):
        super().__init__()
        # Lazy import to avoid circular: model_layers_common imports torch.nn.
        from backend_torch.model_layers_common import make_conv_block

        self.pos_len = pos_len
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.conv_down = make_conv_block(input_channels, embed_dim, kernel_size=3)
        self.blocks = nn.ModuleList(
            [
                TransformerResidualBlock(embed_dim, num_heads, pos_len)
                for _ in range(num_blocks)
            ]
        )
        self.conv_up = make_conv_block(embed_dim, output_dim, kernel_size=3)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        del training
        res = x
        x = self.conv_down(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_up(x)
        return res + x
