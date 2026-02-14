"""
Transformer Block for Go.

Implements a Transformer block with:
- 2D Rotary Position Embeddings (RoPE)
- Group Query Attention (GQA)
- SwiGLU Feed-Forward Network
- RMSNorm

Ported from KataGo's TransformerRoPEGQABlock to Keras/TF.
"""

from __future__ import annotations

import math
import numpy as np
import tensorflow as tf
import keras

from constants import BOARD_LEN

# Base for RoPE frequency computation. 100.0 is appropriate for a 19x19 board
# (must be > 2 * pos_len to avoid aliasing).
ROPE_THETA = 100.0


def spiral_rope_cos_sin_table(num_rotations: int, embed_dim: int, grid_len: int):
    """
    See: https://arxiv.org/html/2602.03227v1
    """
    assert embed_dim % (num_rotations * 2) == 0
    assert embed_dim % 4 == 0
    elems_per_rotation = embed_dim // num_rotations
    seq_len = grid_len * grid_len

    # thetas are assigned in an interleaved way. it's a bit complicated.
    t = np.arange(embed_dim // 4)
    thetas = ROPE_THETA ** (-t / (embed_dim // 4))
    theta_table = np.zeros((embed_dim,))
    for i in range(embed_dim):
        K = num_rotations
        # which rotation partition
        k = i // elems_per_rotation
        # normalize directions 90d apart.
        k_norm = k % (K // 2)
        theta_base = 2 * k_norm
        # which element within the rotation partition
        rot_elem_offset = i % elems_per_rotation
        # which rotation (we work in pairs)
        rot_offset = rot_elem_offset // 2
        # thetas are assigned in pairs, with each pair K apart.
        theta_offset_base = (rot_offset // 2) * K
        theta_offset = theta_offset_base + (rot_offset % 2)
        theta_idx = min(len(thetas) - 1, theta_base + theta_offset)
        # print(
        #     f"i: {i}, k: {k}, k_norm: {k_norm}, theta_base: {theta_base}, rot_elem_offset: {rot_elem_offset}, rot_offset: {rot_offset}, theta_offset_base: {theta_offset_base}, theta_offset: {theta_offset}, theta_idx: {theta_idx}"
        # )
        theta_table[i] = thetas[theta_idx]

    # now compute angle projections.
    angles = np.arange(num_rotations) * (np.pi / num_rotations)
    x_coords = np.arange(grid_len)
    y_coords = np.arange(grid_len)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
    x_flat = x_grid.flatten()  # [seq_len]
    y_flat = y_grid.flatten()  # [seq_len]

    angle_projs = np.zeros((seq_len, embed_dim))
    for d in range(embed_dim):
        angle_idx = d // elems_per_rotation
        angle = angles[angle_idx]
        # Broadcast over all positions at once
        angle_projs[:, d] = x_flat * np.cos(angle) + y_flat * np.sin(angle)

    # compute rot tables
    rot_table = theta_table * angle_projs
    cos_table = np.cos(rot_table)
    sin_table = np.sin(rot_table)
    # print(f"shape: {(seq_len, embed_dim)}, rotations: {num_rotations}")
    # print("thetas\n", thetas)
    # print("theta_table\n", theta_table)
    # print("angles\n", angles)
    # print("angle_projs\n", angle_projs)
    # print("rot_table\n", rot_table)
    # print("cos\n", cos_table)
    # print("sin\n", sin_table)
    return cos_table, sin_table


@keras.saving.register_keras_serializable(package="custom")
class RoPE(keras.layers.Layer):
    """
    Layer that applies spiral RoPE to input tensors.

    Expects input shape (B, S, num_heads, head_dim) and applies the same RoPE
    to all heads. The RoPE is computed based on the sequence length S and the
    head dimension.

    This layer is separate from the attention block for modularity, but in
    practice it is only used within the TransformerBlock.
    """

    def __init__(
        self,
        pos_len,
        head_dim,
        num_rotations,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.seq_len = pos_len * pos_len
        self.pos_len = pos_len
        self.head_dim = head_dim
        self.num_rotations = num_rotations
        cos, sin = spiral_rope_cos_sin_table(
            num_rotations,
            head_dim,
            pos_len,
        )
        self._rope_cos = tf.constant(cos, dtype=tf.float32)
        self._rope_sin = tf.constant(sin, dtype=tf.float32)

        # Precompute permutation indices for pair swapping: [1, 0, 3, 2, 5, 4, ...]
        # This swaps elements in each pair (x0, x1) -> (x1, x0)
        pair_swap_indices = np.zeros(head_dim, dtype=np.int32)
        for i in range(head_dim // 2):
            pair_swap_indices[2 * i] = 2 * i + 1      # Even indices get odd values
            pair_swap_indices[2 * i + 1] = 2 * i      # Odd indices get even values
        self._pair_swap_indices = tf.constant(pair_swap_indices, dtype=tf.int32)

        # Precompute sign patterns for RoPE formula: x0' = x0*cos + x1*sin, x1' = x0*sin - x1*cos
        # We compute: x*cos*sign_cos + x_swapped*sin
        # For position 0 (x0): x0*cos*1 + x1*sin = x0*cos + x1*sin ✓
        # For position 1 (x1): x1*cos*(-1) + x0*sin = -x1*cos + x0*sin ✓
        sign_cos = np.ones(head_dim, dtype=np.float32)
        for i in range(head_dim // 2):
            sign_cos[2 * i] = 1.0  # x0 position: positive cos
            sign_cos[2 * i + 1] = -1.0  # x1 position: negative cos
        self._sign_cos = tf.constant(sign_cos, dtype=tf.float32)

    def call(self, x):
        # Optimized RoPE using gather + element-wise ops instead of reshape/slice/stack
        # This reduces memory-bound operations from 6+ to just 1 gather + 3 element-wise

        # Slice to current sequence length and reshape for broadcasting
        # cos = self._rope_cos[: self.seq_len]  # (S, head_dim)
        # sin = self._rope_sin[: self.seq_len]  # (S, head_dim)
        cos = self._rope_cos
        sin = self._rope_sin

        # Reshape: (S, head_dim) -> (1, S, 1, head_dim)
        cos = tf.reshape(cos, [1, self.seq_len, 1, self.head_dim])
        sin = tf.reshape(sin, [1, self.seq_len, 1, self.head_dim])

        # Swap pairs using gather: (x0, x1, x2, x3, ...) -> (x1, x0, x3, x2, ...)
        x_swapped = tf.gather(x, self._pair_swap_indices, axis=-1)

        # Apply RoPE formula: x' = x*cos*sign_cos + x_swapped*sin
        # For each pair (x0, x1): x0' = x0*cos + x1*sin, x1' = -x1*cos + x0*sin
        return x * cos * self._sign_cos + x_swapped * sin

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pos_len": self.pos_len,
                "head_dim": self.head_dim,
                "num_rotations": self.num_rotations,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="custom")
class TransformerBlock(keras.layers.Layer):
    """
    Transformer block with 2D RoPE, GQA, and SwiGLU FFN.

    Input/output shape: (B, H, W, C) in NHWC format.

    Architecture per block (pre-norm):
        x -> RMSNorm -> MHA(RoPE) -> + residual -> RMSNorm -> SwiGLU FFN -> + residual

    Weight matrix shapes (Keras Dense stores as (input_dim, output_dim)):
        W_q: (head_dim, head_dim)                  i.e. (head_dim, num_heads * head_dim)
        W_k: (head_dim, num_kv_heads * head_dim)
        W_v: (head_dim, num_kv_heads * head_dim)
        W_o: (head_dim, head_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        pos_len: int = BOARD_LEN,
        **kwargs,
    ):
        def round_to_multiple(value, multiple):
            return max(multiple * round(value / multiple), multiple)

        assert embed_dim % num_heads == 0
        super().__init__(**kwargs)
        # dims
        self.pos_len = pos_len
        self.seq_len = pos_len * pos_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = 2 * self.embed_dim
        # layers
        self.rms_in = keras.layers.RMSNormalization(name="rms_in")
        self.rope = RoPE(
            pos_len=self.pos_len,
            head_dim=self.head_dim,
            num_rotations=4,
            name="spiral_rope",
        )
        self.Q = keras.layers.Dense(embed_dim, use_bias=False, name="query")
        self.K = keras.layers.Dense(embed_dim, use_bias=False, name="key")
        self.V = keras.layers.Dense(embed_dim, use_bias=False, name="value")
        self.O = keras.layers.Dense(embed_dim, use_bias=False, name="output")
        self.rms_out = keras.layers.RMSNormalization(name="rms_out")
        # swiglu layers
        self.ffn_gate = keras.layers.Dense(
            self.ffn_dim,
            use_bias=False,
            name="swiglu_gate",
        )
        self.ffn_up = keras.layers.Dense(self.ffn_dim, use_bias=False, name="swiglu_up")
        self.ffn_down = keras.layers.Dense(
            embed_dim, use_bias=False, name="swiglu_down"
        )

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        x = keras.ops.reshape(x, (batch_size, self.seq_len, self.embed_dim))
        res = x
        x = self.rms_in(x)

        # project
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # reshape
        q = keras.ops.reshape(
            q, (batch_size, self.seq_len, self.num_heads, self.head_dim)
        )
        k = keras.ops.reshape(
            k, (batch_size, self.seq_len, self.num_heads, self.head_dim)
        )
        v = keras.ops.reshape(
            v, (batch_size, self.seq_len, self.num_heads, self.head_dim)
        )

        # spiral rope
        q = self.rope(q)
        k = self.rope(k)

        # mha
        attn_output = keras.ops.dot_product_attention(q, k, v)
        attn_output = keras.ops.reshape(
            attn_output, (batch_size, self.seq_len, self.embed_dim)
        )
        attn_output = self.O(attn_output)
        x = res + attn_output
        res = x
        x = self.rms_out(x)

        # swiglu
        gate = self.ffn_gate(x)
        up = self.ffn_up(x)
        x = keras.ops.silu(gate) * up
        x = self.ffn_down(x)
        x = x + res
        return tf.reshape(x, (batch_size, self.pos_len, self.pos_len, self.embed_dim))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pos_len": self.pos_len,
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
