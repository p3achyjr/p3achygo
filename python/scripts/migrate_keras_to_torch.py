"""One-time migration: .keras checkpoint → torch .pt checkpoint.

Usage:
    KERAS_BACKEND=tensorflow PYTHONPATH=python python scripts/migrate_keras_to_torch.py \
        --input  ~/p3achygo-data/v4-models/b9-legacy/model_0311.keras \
        --output ~/p3achygo-data/v4-models/b9-legacy/model_0311.pt

Weight transpositions:
  Conv2D kernel  (H,W,Cin,Cout) → (Cout,Cin,H,W)
  Dense kernel   (Cin,Cout)     → (Cout,Cin)
  BN stats       (C,)           → (C,)   copy verbatim
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# --------------------------------------------------------------------------
# Low-level copy primitives
# --------------------------------------------------------------------------


def _copy_conv2d(k_conv, t_conv) -> None:
    """keras Conv2D (H,W,Cin,Cout) → torch Conv2d (Cout,Cin,H,W)."""
    w = torch.tensor(k_conv.kernel.numpy()).permute(3, 2, 0, 1).contiguous()
    with torch.no_grad():
        t_conv.weight.copy_(w)
        if t_conv.bias is not None and getattr(k_conv, "bias", None) is not None:
            t_conv.bias.copy_(torch.tensor(k_conv.bias.numpy()))


def _copy_bn(k_bn, t_bn) -> None:
    """keras BatchNormalization → torch BatchNorm2d."""
    with torch.no_grad():
        t_bn.weight.copy_(torch.tensor(k_bn.gamma.numpy()))
        t_bn.bias.copy_(torch.tensor(k_bn.beta.numpy()))
        t_bn.running_mean.copy_(torch.tensor(k_bn.moving_mean.numpy()))
        t_bn.running_var.copy_(torch.tensor(k_bn.moving_variance.numpy()))


def _copy_linear(k_dense, t_linear) -> None:
    """keras Dense (Cin,Cout) → torch Linear (Cout,Cin)."""
    w = torch.tensor(k_dense.kernel.numpy()).T.contiguous()
    with torch.no_grad():
        t_linear.weight.copy_(w)
        if t_linear.bias is not None and getattr(k_dense, "bias", None) is not None:
            t_linear.bias.copy_(torch.tensor(k_dense.bias.numpy()))


def _copy_conv_pre_act_to_bn_conv(k_block, t_bn, t_conv) -> None:
    """keras ConvPreActivation → separate torch (BN, Conv2d) pair."""
    _copy_bn(k_block.norm_layer, t_bn)
    _copy_conv2d(k_block.conv, t_conv)


def _copy_conv_pre_act(k_block, t_block) -> None:
    """keras ConvPreActivation → torch ConvPreActivation."""
    _copy_bn(k_block.norm_layer, t_block.bn)
    _copy_conv2d(k_block.conv, t_block.conv)


# --------------------------------------------------------------------------
# Block-level copiers (explicit structural knowledge)
# --------------------------------------------------------------------------


def _copy_classic_res(k_classic, t_classic) -> None:
    """keras ClassicResidualBlock → torch ClassicResidualBlock.

    keras: blocks = [ConvPreAct0, ConvPreAct1]
    torch: layers = [ConvPreActivation0, ConvPreActivation1]
    """
    for k_b, t_l in zip(k_classic.blocks, t_classic.layers):
        _copy_conv_pre_act(k_b, t_l)


def _copy_nbt(k_nbt, t_nbt) -> None:
    """keras NbtResidualBlock → torch NbtResidualBlock.

    keras blocks[]:
      [0] ConvPreActivation (reduce_dim)
      [1] ClassicResidualBlock (nbt_res0)
      [2] ClassicResidualBlock (nbt_res1)
      [3] ConvPreActivation (expand_dim)
    torch:
      reduce_bn, reduce_conv, res0, res1, expand_bn, expand_conv
    """
    k = k_nbt.blocks
    _copy_conv_pre_act_to_bn_conv(k[0], t_nbt.reduce_bn, t_nbt.reduce_conv)
    _copy_classic_res(k[1], t_nbt.res0)
    _copy_classic_res(k[2], t_nbt.res1)
    _copy_conv_pre_act_to_bn_conv(k[3], t_nbt.expand_bn, t_nbt.expand_conv)


def _copy_broadcast(k_bc, t_bc) -> None:
    """keras BroadcastResidualBlock → torch BroadcastResidualBlock.

    keras blocks[]:
      [0] ConvPreActivation (broadcast_conv_first)
      [1] BroadcastPreAct   (broadcast_mix) — has .dense
      [2] ConvPreActivation (broadcast_conv_last)
    torch:
      conv_first (ConvPreActivation), dense_mix (Linear), conv_last (ConvPreActivation)
    """
    k = k_bc.blocks
    _copy_conv_pre_act(k[0], t_bc.conv_first)
    _copy_linear(k[1].dense, t_bc.dense_mix)
    _copy_conv_pre_act(k[2], t_bc.conv_last)


def _copy_policy_head(k_ph, t_ph) -> None:
    _copy_conv2d(k_ph.conv_p, t_ph.conv_p)
    _copy_conv2d(k_ph.conv_g, t_ph.conv_g)
    # GlobalPoolBias: k_ph.gpool has .g_norm_layer (BN) and .dense
    _copy_bn(k_ph.gpool.g_norm_layer, t_ph.gpool.bn)
    _copy_linear(k_ph.gpool.dense, t_ph.gpool.dense)
    # output layers
    _copy_conv2d(k_ph.output_moves, t_ph.output_moves)
    _copy_linear(k_ph.output_pass, t_ph.output_pass)
    _copy_conv2d(k_ph.soft_policy_moves, t_ph.soft_moves)
    _copy_linear(k_ph.soft_policy_pass, t_ph.soft_pass)
    _copy_conv2d(k_ph.optimistic_policy_moves, t_ph.opt_moves)
    _copy_linear(k_ph.optimistic_policy_pass, t_ph.opt_pass)


def _copy_rms_norm(k_rms, t_rms) -> None:
    """keras.layers.RMSNormalization (.scale, shape (C,)) → torch.nn.RMSNorm (.weight, shape (C,))."""
    with torch.no_grad():
        t_rms.weight.copy_(torch.tensor(k_rms.scale.numpy()))


def _copy_transformer_attention(k_attn, t_attn) -> None:
    """keras TransformerAttention → torch TransformerAttention.

    Both have: rms (RMSNorm), rope (no params — buffers), Q/K/V/O Linear.
    RoPE has no trainable params; the cos/sin/sign tables are computed
    deterministically from (pos_len, head_dim, num_rotations) so they
    match by construction when the configs match.
    """
    _copy_rms_norm(k_attn.rms, t_attn.rms)
    _copy_linear(k_attn.Q, t_attn.Q)
    _copy_linear(k_attn.K, t_attn.K)
    _copy_linear(k_attn.V, t_attn.V)
    _copy_linear(k_attn.O, t_attn.O)


def _copy_transformer_ffn(k_ffn, t_ffn) -> None:
    """keras TransformerFFN → torch TransformerFFN. RMS + 3 Linears."""
    _copy_rms_norm(k_ffn.rms, t_ffn.rms)
    _copy_linear(k_ffn.ffn_gate, t_ffn.ffn_gate)
    _copy_linear(k_ffn.ffn_up, t_ffn.ffn_up)
    _copy_linear(k_ffn.ffn_down, t_ffn.ffn_down)


def _copy_transformer_residual(k_blk, t_blk) -> None:
    """keras TransformerResidualBlock → torch TransformerResidualBlock."""
    _copy_transformer_attention(k_blk.transformer_attn, t_blk.transformer_attn)
    _copy_transformer_ffn(k_blk.transformer_ffn, t_blk.transformer_ffn)


def _copy_transformer_btl(k_btl, t_btl) -> None:
    """keras TransformerBottleneckBlock → torch TransformerBottleneckBlock.

    keras: conv_down (ConvPreActivation), blocks=[TransformerResidualBlock, ...], conv_up (ConvPreActivation)
    torch: same structure with `nn.ModuleList` blocks.
    """
    _copy_conv_pre_act(k_btl.conv_down, t_btl.conv_down)
    for k_b, t_b in zip(k_btl.blocks, t_btl.blocks):
        _copy_transformer_residual(k_b, t_b)
    _copy_conv_pre_act(k_btl.conv_up, t_btl.conv_up)


def _copy_value_head(k_vh, t_vh) -> None:
    _copy_conv2d(k_vh.conv, t_vh.conv)
    # gpool is GlobalPool — no weights
    _copy_linear(k_vh.outcome_q_embed, t_vh.outcome_q_embed)
    _copy_linear(k_vh.outcome_q_output, t_vh.outcome_q_output)
    _copy_linear(k_vh.outcome_mcts_dist, t_vh.outcome_mcts)
    _copy_conv2d(k_vh.conv_ownership, t_vh.conv_own)
    # score / gamma (shared score_pre)
    _copy_linear(k_vh.score_pre, t_vh.score_pre)
    _copy_linear(k_vh.gamma_output, t_vh.gamma_output)
    _copy_linear(k_vh.score_output, t_vh.score_output)


# --------------------------------------------------------------------------
# Top-level migration
# --------------------------------------------------------------------------

_BLOCK_TYPES = {
    "NbtResidualBlock": _copy_nbt,
    "BroadcastResidualBlock": _copy_broadcast,
    "TransformerResidualBlock": _copy_transformer_residual,
    "TransformerBottleneckBlock": _copy_transformer_btl,
}


def migrate(input_path: str, output_path: str) -> None:
    print(f"Loading keras model from {input_path} …")
    import keras
    from backend_tf.model import P3achyGoModel as KP3achyGoModel
    from backend_torch.model import P3achyGoModel as TP3achyGoModel

    km = keras.models.load_model(
        input_path,
        custom_objects=KP3achyGoModel.custom_objects(),
        compile=False,
    )
    km.trainable = False
    print(f"  keras params: {sum(v.numpy().size for v in km.trainable_weights):,}")

    def _plain(obj):
        # Strip keras TrackedDict/TrackedList wrappers (they hold non-picklable
        # closures from Layer._initialize_tracker).
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_plain(v) for v in obj]
        return obj

    config = _plain(dict(km.get_config()))
    if "num_head_channels" in config and "num_policy_head_channels" not in config:
        config["num_policy_head_channels"] = config.pop("num_head_channels")
        config["num_value_head_channels"] = config.get("c_val", 64)
    config.pop("name", None)
    config.pop("is_transformer", None)

    print("Building torch model …")
    tm = TP3achyGoModel(**config)
    tm.eval()
    print(f"  torch params: {sum(p.numel() for p in tm.parameters()):,}")

    print("Copying weights …")
    # Stem
    _copy_conv2d(km.init_board_conv, tm.init_board_conv)
    _copy_linear(km.init_game_layer, tm.init_game_layer)

    # Trunk blocks (matched by position, type verified)
    k_blocks = list(km.blocks)
    t_blocks = list(tm.blocks)
    if len(k_blocks) != len(t_blocks):
        raise ValueError(
            f"Block count mismatch: keras {len(k_blocks)} vs torch {len(t_blocks)}"
        )

    for i, (kb, tb) in enumerate(zip(k_blocks, t_blocks)):
        block_type = type(kb).__name__
        copy_fn = _BLOCK_TYPES.get(block_type)
        if copy_fn is None:
            raise NotImplementedError(f"No copier for block type {block_type!r}")
        copy_fn(kb, tb)
        print(f"  block {i:2d} ({block_type}) … ok")

    # Heads
    _copy_policy_head(km.policy_head, tm.policy_head)
    print("  policy_head … ok")
    _copy_value_head(km.value_head, tm.value_head)
    print("  value_head … ok")

    print(f"Saving to {output_path} …")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Use the torch model's own config_dict — keras's get_config() contains
    # non-picklable closures (Layer._initialize_tracker.<locals>.<lambda>).
    torch.save({"model": tm.state_dict(), "config": tm.config_dict()}, output_path)
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    migrate(args.input, args.output)


if __name__ == "__main__":
    main()
