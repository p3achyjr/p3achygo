"""Torch-native P3achyGoModel.

All tensors use NCHW shape with channels_last memory storage. Call
    model = model.to(memory_format=torch.channels_last)
and
    board_state = board_state.permute(0,3,1,2).contiguous(
        memory_format=torch.channels_last)
before the forward pass for cuDNN NHWC fast-path kernels.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend_torch.model_layers_common import (
    ConvPreActivation,
    make_bn,
    make_conv,
    make_conv_block,
    make_dense,
    mish,
)

# ---------------------------------------------------------------------------
# Global pooling helpers
# ---------------------------------------------------------------------------


class GlobalPool(nn.Module):
    """Concatenate spatial mean and max → shape (N, 2C)."""

    def forward(self, x: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=keepdims)
        mx = x.amax(dim=(2, 3), keepdim=keepdims)
        return torch.cat([mean, mx], dim=1)


class GlobalPoolBias(nn.Module):
    """Mirrors keras GlobalPoolBias: x += dense(gpool(BN(mish(g)))).

    Both x and g have shape (N, C, H, W). g is BN'd, activated, globally
    pooled (→ N, 2C), projected to C, then added as a spatial bias to x.
    Returns (x_biased, g_pooled) where g_pooled is shape (N, 2C).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.bn = make_bn(channels)  # applied to g before pooling
        self.pool = GlobalPool()
        self.dense = make_dense(2 * channels, channels)

    def forward(
        self, x: torch.Tensor, g: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = mish(self.bn(g))  # BN + activation on g
        g_pooled = self.pool(g)  # (N, 2C)
        bias = self.dense(g_pooled)  # (N, C)
        # `bias[..., None, None]` is two static `Unsqueeze` ops in ONNX —
        # unlike `bias.view(bias.shape[0], bias.shape[1], 1, 1)` which would
        # rebuild the shape via Shape→Gather→Concat at runtime.
        return x + bias[..., None, None], g_pooled


# ---------------------------------------------------------------------------
# Squeeze-and-excitation / spatial attention
# ---------------------------------------------------------------------------


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = GlobalPool()
        self.dense = make_dense(2 * channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.dense(self.pool(x)))  # (N, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = make_conv(2, 1, kernel_size, use_bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)  # (N,1,H,W)
        mx = x.amax(dim=1, keepdim=True)  # (N,1,H,W)
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------


class ClassicResidualBlock(nn.Module):
    """Stack of ConvPreActivation + optional SE/SpatialAttention + skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        conv_size: int,
        use_se: bool = False,
        use_spatial_attn: bool = False,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            layers.append(ConvPreActivation(ch_in, out_channels, conv_size))
        self.layers = nn.ModuleList(layers)
        self.se = SqueezeExcitation(out_channels) if use_se else None
        self.spatial_attn = SpatialAttention() if use_spatial_attn else None
        # projection when in_channels != out_channels (not used in NBT wrapper)
        self.proj = (
            make_conv(in_channels, out_channels, 1, use_bias=False)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        residual = x if self.proj is None else self.proj(x)
        for layer in self.layers:
            x = layer(x, training)
        if self.se is not None:
            x = self.se(x)
        if self.spatial_attn is not None:
            x = self.spatial_attn(x)
        return x + residual


class NbtResidualBlock(nn.Module):
    """1×1 reduce → Classic(inner) → Classic(inner) → 1×1 expand + skip."""

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int,
        conv_size: int,
        use_se: bool = False,
        use_spatial_attn: bool = False,
    ):
        super().__init__()
        self.reduce_bn = make_bn(channels)
        self.reduce_conv = make_conv(channels, bottleneck_channels, 1, use_bias=False)
        self.res0 = ClassicResidualBlock(
            bottleneck_channels,
            bottleneck_channels,
            2,
            conv_size,
            use_se=use_se,
            use_spatial_attn=use_spatial_attn,
        )
        self.res1 = ClassicResidualBlock(
            bottleneck_channels,
            bottleneck_channels,
            2,
            conv_size,
        )
        self.expand_bn = make_bn(bottleneck_channels)
        self.expand_conv = make_conv(bottleneck_channels, channels, 1, use_bias=False)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        residual = x
        # pre-act reduce
        h = mish(self.reduce_bn(x))
        h = self.reduce_conv(h)
        # inner blocks
        h = self.res0(h, training)
        h = self.res1(h, training)
        # pre-act expand
        h = mish(self.expand_bn(h))
        h = self.expand_conv(h)
        return h + residual


class BroadcastResidualBlock(nn.Module):
    """Mirrors keras BroadcastResidualBlock exactly.

    Flow: ConvPreActivation(1×1) → mish+dense_mix → ConvPreActivation(1×1) + skip.
    Each ConvPreActivation has its own BN (2 BNs total per block).
    The dense branch uses mish only, no BN — matching keras BroadcastPreAct.

    In NCHW layout the NHWC↔NCHW transpose of keras BroadcastPreAct
    disappears; the dense operates on the last dim of (N, C, H*W).
    """

    def __init__(self, channels: int, board_len: int = 19):
        super().__init__()
        spatial = board_len * board_len
        # Cache for forward (used as constants in Reshape calls).
        self._channels = channels
        self._board_len = board_len
        self._spatial = spatial
        # Two ConvPreActivation blocks (each owns a BN)
        self.conv_first = ConvPreActivation(channels, channels, 1)
        self.dense_mix = make_dense(spatial, spatial, use_bias=True)
        self.conv_last = ConvPreActivation(channels, channels, 1)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        residual = x
        h = self.conv_first(x)  # BN → mish → conv
        # broadcast mix: mish + spatial dense (no BN — matches BroadcastPreAct)
        h = mish(h)
        # Use -1 for batch and the cached static C/H/W to keep the exported
        # Reshape ops constant-shape (no Shape→Gather→Concat dynamic arithmetic).
        h = self.dense_mix(h.view(-1, self._channels, self._spatial)).view(
            -1, self._channels, self._board_len, self._board_len
        )
        h = self.conv_last(h)  # BN → mish → conv
        return h + residual


def _build_trunk_from_generic_arch(
    generic_arch: Dict,
    board_len: int,
) -> nn.ModuleList:
    """Build trunk blocks from a generic_arch dict (new-style config)."""
    from backend_torch.model_transformer import (
        TransformerBottleneckBlock,
        TransformerResidualBlock,
    )

    blocks = []
    # Track running trunk channel count so transformer_btl knows its input.
    trunk_ch = None
    for block_type, cfg in generic_arch["trunk"]:
        if block_type == "transformer":
            ed = cfg["embed_dim"]
            nh = cfg["num_heads"]
            blocks.append(TransformerResidualBlock(ed, nh, board_len))
            trunk_ch = ed
            continue
        if block_type == "transformer_btl":
            ed = cfg["embed_dim"]
            nh = cfg["num_heads"]
            out_dim = cfg.get("output_dim", trunk_ch or ed)
            n_inner = cfg.get("num_blocks", 2)
            in_ch = trunk_ch or out_dim
            blocks.append(
                TransformerBottleneckBlock(in_ch, out_dim, ed, nh, board_len, n_inner)
            )
            trunk_ch = out_dim
            continue

        out_ch = cfg.get("output_channels", cfg.get("channels", 384))
        bot_ch = cfg.get("bottleneck_channels", out_ch // 2)
        ks = cfg.get("conv_size", 3)
        use_se = cfg.get("use_se", False)
        if block_type == "nbt":
            blocks.append(NbtResidualBlock(out_ch, bot_ch, ks, use_se=use_se))
        elif block_type == "broadcast":
            blocks.append(BroadcastResidualBlock(out_ch, board_len))
        elif block_type in ("btl", "classic"):
            blocks.append(ClassicResidualBlock(out_ch, out_ch, 4, ks, use_se=use_se))
        else:
            raise ValueError(f"Unknown block type: {block_type!r}")
        trunk_ch = out_ch
    return nn.ModuleList(blocks)


# ---------------------------------------------------------------------------
# Policy head
# ---------------------------------------------------------------------------


class PolicyHead(nn.Module):
    """Mirrors keras PolicyHead exactly.

    Architecture:
      p = conv_p(x), g = conv_g(x)
      (p, g_pooled) = gpool(p, g)   [BN(g)→mish→pool→dense→bias on p]
      p = mish(p)
      board = conv(48→2, 1×1)(p)    [ch0=main, ch1=aux]
      pass  = dense(96→2)(g_pooled) - 3
      soft  = conv(48→1)(p) ++ dense(96→1)(g_pooled) - 3
      opt   = conv(48→1)(p) ++ dense(96→1)(g_pooled) - 3
    """

    def __init__(self, in_channels: int, head_channels: int, board_len: int = 19):
        super().__init__()
        self.board_len = board_len
        self.conv_p = make_conv(in_channels, head_channels, 1, use_bias=False)
        self.conv_g = make_conv(in_channels, head_channels, 1, use_bias=False)
        self.gpool = GlobalPoolBias(head_channels)
        # main policy output (conv has no bias; pass dense has bias — matches keras)
        self.output_moves = make_conv(head_channels, 2, 1, use_bias=False)
        self.output_pass = make_dense(2 * head_channels, 2)
        # soft auxiliary policy
        self.soft_moves = make_conv(head_channels, 1, 1, use_bias=False)
        self.soft_pass = make_dense(2 * head_channels, 1)
        # optimistic auxiliary policy
        self.opt_moves = make_conv(head_channels, 1, 1, use_bias=False)
        self.opt_pass = make_dense(2 * head_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # All non-batch dims are static (board_len fixed at module construction).
        # Use -1 for batch in every Reshape so the exported ONNX has constant
        # shape tensors instead of Shape→Gather→Concat dynamic arithmetic.
        spatial = self.board_len * self.board_len
        p = self.conv_p(x)
        g = self.conv_g(x)
        p, g_pooled = self.gpool(p, g)  # g_pooled: (N, 2*head_ch)
        p = mish(p)  # activate after bias addition

        # main + aux policy
        board = self.output_moves(p)  # (N,2,H,W)
        pass_logits = self.output_pass(g_pooled) - 3  # (N,2)
        board_pi = board[:, 0].reshape(-1, spatial)
        board_aux = board[:, 1].reshape(-1, spatial)
        pi_logits = torch.cat([board_pi, pass_logits[:, :1]], dim=1)
        pi_aux = torch.cat([board_aux, pass_logits[:, 1:]], dim=1)

        # soft auxiliary
        soft_board = self.soft_moves(p).reshape(-1, spatial)
        soft_pass = self.soft_pass(g_pooled) - 3  # (N,1)
        pi_soft = torch.cat([soft_board, soft_pass], dim=1)

        # optimistic auxiliary
        opt_board = self.opt_moves(p).reshape(-1, spatial)
        opt_pass = self.opt_pass(g_pooled) - 3
        pi_opt = torch.cat([opt_board, opt_pass], dim=1)

        return pi_logits, pi_aux, pi_soft, pi_opt


# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Mirrors keras ValueHead exactly.

    Architecture:
      v   = conv(in→C, 1×1)(x)
      g   = GlobalPool(v)                      # (N, 2C)
      emb = mish(dense(2C→c_val)(g))
      game_outcome = dense(c_val→14)(emb)       # outcome[0:2], q×3, q_err×3, q_sc×3, q_sc_err×3
      mcts_logits  = dense(c_val→51)(emb)
      own = tanh(conv(C→1, 1×1)(v))             # (N,H,W)
      gamma = dense(c_val→1)(mish(dense(97→c_val)(cat(g, 0))))  where 97 = 2C+1
      score_logits:
          v_scores = cat(g.expand(N,800,2C), scores.expand(N,800,1))  # (N,800,2C+1)
          score_logits = dense(c_val→1)(mish(dense(97→c_val)(v_scores)))  # (N,800)
          score_logits = clamp(softplus(gamma), max=10) * score_logits
    """

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        c_val: int,
        board_len: int = 19,
        score_range: int = 800,
        n_v_buckets: int = 51,
    ):
        super().__init__()
        self.board_len = board_len
        self.score_range = score_range

        pool_dim = 2 * head_channels  # GlobalPool output size

        self.conv = make_conv(in_channels, head_channels, 1, use_bias=False)
        self.pool = GlobalPool()
        # game outcome / Q subhead
        self.outcome_q_embed = make_dense(pool_dim, c_val)
        self.outcome_q_output = make_dense(c_val, 14)
        self.outcome_mcts = make_dense(c_val, n_v_buckets)
        # ownership subhead (no bias — matches keras)
        self.conv_own = make_conv(head_channels, 1, 1, use_bias=False)
        # score / gamma subhead.
        # score_pre is SHARED between gamma computation (N, 97) and
        # per-bin score distribution (N, 800, 97) — same linear applied broadcast.
        self.score_pre = make_dense(pool_dim + 1, c_val)  # 97 = 2C+1, shared
        self.gamma_output = make_dense(c_val, 1)  # gamma scalar
        self.score_output = make_dense(c_val, 1)  # per-bin logit

    def _default_scores(self, device, dtype):
        mid = self.score_range // 2
        return (
            torch.arange(self.score_range, device=device, dtype=dtype) - mid + 0.5
        ) * 0.05

    def forward(
        self, x: torch.Tensor, scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        v = self.conv(x)  # (N, C, H, W)
        g = self.pool(v)  # (N, 2C)

        # Game outcome / Q values
        emb = mish(self.outcome_q_embed(g))  # (N, c_val)
        go = self.outcome_q_output(emb)  # (N, 14)
        outcome_logits = go[:, 0:2]
        outcome_probs = F.softmax(outcome_logits, dim=-1)

        q6 = torch.tanh(go[:, 2:3])
        q16 = torch.tanh(go[:, 3:4])
        q50 = torch.tanh(go[:, 4:5])
        q6_err = 4 * torch.sigmoid(go[:, 5:6])
        q16_err = 4 * torch.sigmoid(go[:, 6:7])
        q50_err = 4 * torch.sigmoid(go[:, 7:8])
        q6_sc = go[:, 8:9]
        q16_sc = go[:, 9:10]
        q50_sc = go[:, 10:11]
        q6_sc_err = go[:, 11:12].abs()
        q16_sc_err = go[:, 12:13].abs()
        q50_sc_err = go[:, 13:14].abs()

        mcts_logits = self.outcome_mcts(emb)
        mcts_probs = F.softmax(mcts_logits, dim=-1)

        # Ownership
        own = torch.tanh(self.conv_own(v)).squeeze(1)  # (N, H, W)

        # Gamma (score scale) — uses SHARED score_pre.
        # `F.pad(g, (0, 1))` appends one zero column on the last dim — single
        # Pad op vs the cat-with-zeros pattern which needs an explicit
        # zero-tensor of dynamic batch shape.
        gamma_in = F.pad(g, (0, 1))  # (N, 2C+1)
        gamma = self.gamma_output(mish(self.score_pre(gamma_in)))  # (N, 1)

        # Score distribution — per-bin MLP over 800 bins.
        if scores is None:
            scores = self._default_scores(g.device, g.dtype)  # (800,)
        scores = scores.to(g.dtype)
        # Concatenate g (broadcast over score_range) with scores (broadcast
        # over batch) into (N, 800, 2C+1). `expand(-1, ...)` keeps existing
        # dim sizes — the dynamic batch dim N stays a -1 in the exported
        # Reshape rather than being looked up via Shape→Gather.
        g_exp = g.unsqueeze(1).expand(-1, self.score_range, -1)  # (N, 800, 2C)
        s_exp = scores.view(1, self.score_range, 1).expand(
            g.size(0), -1, -1
        )  # (N, 800, 1)
        v_scores = torch.cat([g_exp, s_exp], dim=-1)  # (N, 800, 2C+1)
        score_logits = self.score_output(mish(self.score_pre(v_scores))).squeeze(
            -1
        )  # (N, 800)
        score_logits = torch.clamp(F.softplus(gamma), max=10) * score_logits
        score_probs = F.softmax(score_logits, dim=-1)

        return (
            outcome_logits,
            outcome_probs,
            own,
            score_logits,
            score_probs,
            gamma,
            q6,
            q16,
            q50,
            q6_err,
            q16_err,
            q50_err,
            q6_sc,
            q16_sc,
            q50_sc,
            q6_sc_err,
            q16_sc_err,
            q50_sc_err,
            mcts_logits,
            mcts_probs,
        )


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class P3achyGoModel(nn.Module):
    """Torch-native P3achyGoModel.

    Mirrors the keras P3achyGoModel interface:
      forward(board_state, game_state, training=False, scores=None)
        board_state: (N, H, W, C) NHWC — permuted to NCHW at entry
        game_state : (N, num_features)

    Returns a 25-tuple matching the keras model output order.
    """

    def __init__(
        self,
        board_len: int,
        num_input_planes: int,
        num_input_features: int,
        num_blocks: int,
        num_channels: int,
        num_bottleneck_channels: int,
        num_policy_head_channels: int,
        num_value_head_channels: int,
        c_val: int,
        bottleneck_length: int,
        conv_size: int,
        broadcast_interval: int,
        trunk_block_type: str = "nbt",
        generic_arch: Optional[Dict] = None,
        c_l2: float = 1e-4,
        name: str = "p3achygo",
    ):
        super().__init__()
        self._config = dict(
            board_len=board_len,
            num_input_planes=num_input_planes,
            num_input_features=num_input_features,
            num_blocks=num_blocks,
            num_channels=num_channels,
            num_bottleneck_channels=num_bottleneck_channels,
            num_policy_head_channels=num_policy_head_channels,
            num_value_head_channels=num_value_head_channels,
            c_val=c_val,
            bottleneck_length=bottleneck_length,
            conv_size=conv_size,
            broadcast_interval=broadcast_interval,
            trunk_block_type=trunk_block_type,
            generic_arch=generic_arch,
            c_l2=c_l2,
            name=name,
        )
        self.board_len = board_len
        self.num_input_planes = num_input_planes
        self.num_input_features = num_input_features

        # Stem
        stem_kernel = conv_size + 2  # matches keras: 5 for conv_size=3
        self.init_board_conv = make_conv(
            num_input_planes, num_channels, stem_kernel, use_bias=False
        )
        self.init_game_layer = make_dense(
            num_input_features, num_channels, use_bias=True
        )

        # Trunk
        if generic_arch is not None:
            self.blocks = _build_trunk_from_generic_arch(generic_arch, board_len)
            # read head channel counts from generic_arch if present
            num_policy_head_channels = generic_arch.get(
                "policy_head_channels", num_policy_head_channels
            )
            num_value_head_channels = generic_arch.get(
                "value_head_channels", num_value_head_channels
            )
        else:
            self.blocks = nn.ModuleList(
                _build_legacy_trunk(
                    num_blocks,
                    num_channels,
                    num_bottleneck_channels,
                    bottleneck_length,
                    conv_size,
                    broadcast_interval,
                    trunk_block_type,
                    board_len,
                )
            )

        # Heads
        self.policy_head = PolicyHead(num_channels, num_policy_head_channels, board_len)
        self.value_head = ValueHead(
            num_channels, num_value_head_channels, c_val, board_len
        )

    # ------------------------------------------------------------------
    # keras-compatible interface helpers
    # ------------------------------------------------------------------

    def config_dict(self) -> Dict:
        return dict(self._config)

    def input_planes_shape(self) -> List[int]:
        return [self.board_len, self.board_len, self.num_input_planes]

    def input_features_shape(self) -> List[int]:
        return [self.num_input_features]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        board_state: torch.Tensor,
        game_state: torch.Tensor,
        training: bool = False,
        scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        # board_state: (N, H, W, C) NHWC → (N, C, H, W) NCHW channels_last
        x = board_state.permute(0, 3, 1, 2).contiguous(
            memory_format=torch.channels_last
        )

        # Stem: conv + game-state broadcast bias
        x = self.init_board_conv(x)
        g = self.init_game_layer(game_state)  # (N, channels)
        # `g[..., None, None]` is two static Unsqueeze ops in ONNX; `g.view(N, -1, 1, 1)`
        # would rebuild the shape via Shape→Gather→Concat at runtime.
        x = x + g[..., None, None]  # broadcast (N,C,1,1) over H,W

        # Trunk
        for block in self.blocks:
            x = block(x, training)

        # Heads (policy head pools internally; value head uses the same game state)
        ph = self.policy_head(x)
        vh = self.value_head(x, scores)

        # Unpack
        pi_logits, pi_logits_aux, pi_logits_soft, pi_logits_optimistic = ph
        (
            outcome_logits,
            outcome_probs,
            own,
            score_logits,
            score_probs,
            gamma,
            q6,
            q16,
            q50,
            q6_err,
            q16_err,
            q50_err,
            q6_sc,
            q16_sc,
            q50_sc,
            q6_sc_err,
            q16_sc_err,
            q50_sc_err,
            mcts_logits,
            mcts_probs,
        ) = vh

        pi_probs = F.softmax(pi_logits, dim=-1)

        # Cast to float32 (matches keras model output)
        def _f32(t: torch.Tensor) -> torch.Tensor:
            return t.float()

        return (
            _f32(pi_logits),
            _f32(pi_probs),
            _f32(outcome_logits),
            _f32(outcome_probs),
            _f32(own),
            _f32(score_logits),
            _f32(score_probs),
            _f32(gamma),
            _f32(pi_logits_aux),
            _f32(q6),
            _f32(q16),
            _f32(q50),
            _f32(q6_err),
            _f32(q16_err),
            _f32(q50_err),
            _f32(q6_sc),
            _f32(q16_sc),
            _f32(q50_sc),
            _f32(q6_sc_err),
            _f32(q16_sc_err),
            _f32(q50_sc_err),
            _f32(pi_logits_soft),
            _f32(pi_logits_optimistic),
            _f32(mcts_logits),
            _f32(mcts_probs),
        )

    def compute_losses(self, predictions, targets, weights):
        """Native-torch loss is in backend_torch.losses.compute_losses (free function).
        Kept on the model only to surface a clear error for any caller that
        still expects a method here.
        """
        raise NotImplementedError(
            "Use backend_torch.losses.compute_losses (free function) instead."
        )


# ---------------------------------------------------------------------------
# Legacy trunk builder (non-generic_arch path)
# ---------------------------------------------------------------------------


def _build_legacy_trunk(
    num_blocks: int,
    channels: int,
    bottleneck_channels: int,
    bottleneck_length: int,
    conv_size: int,
    broadcast_interval: int,
    trunk_block_type: str,
    board_len: int,
) -> List[nn.Module]:
    blocks = []
    for i in range(num_blocks):
        if broadcast_interval > 0 and (i + 1) % broadcast_interval == 0:
            blocks.append(BroadcastResidualBlock(channels, board_len))
        elif trunk_block_type == "nbt":
            blocks.append(NbtResidualBlock(channels, bottleneck_channels, conv_size))
        elif trunk_block_type == "btl":
            blocks.append(
                ClassicResidualBlock(channels, channels, bottleneck_length, conv_size)
            )
        else:
            blocks.append(ClassicResidualBlock(channels, channels, 2, conv_size))
    return blocks
