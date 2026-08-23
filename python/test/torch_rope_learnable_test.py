"""Tests for learnable per-block RoPE θ (backend_torch).

Covers the three pieces of the change end to end, with no keras dependency so
they're fast and always runnable:
  1. `RoPE` exposes a learnable per-block `log_theta`; the cos/sin table is
     recomputed from it each forward and equals the old fixed-θ table at init.
  2. Backward-compat load: pre-learnable-θ checkpoints (no `log_theta`, with
     baked `_rope_cos`/`_rope_sin`/... buffers) load strict and seed θ=100.
  3. Optimizer: `log_theta` routes to AdamW with zero weight decay, and a saved
     optimizer state that predates `log_theta` remaps onto the new param set —
     preserving existing momentum and stepping without crashing.

Run:
    PYTHONPATH=python:python/test python python/test/torch_rope_learnable_test.py -v
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, "python")
sys.path.insert(0, "python/test")

import numpy as np
import torch
import torch.nn as nn

from backend_torch.model_transformer import (  # noqa: E402
    RoPE,
    ROPE_THETA,
    spiral_rope_structure,
)
from backend_torch.optimizer import (  # noqa: E402
    ConvMuon,
    build_convmuon_param_groups,
    _maybe_remap_resume_state,
    _MUON_EXCLUDE_LAYERS_TORCH,
)

POS_LEN = 3
HEAD_DIM = 8
NUM_ROT = 4
SEQ_LEN = POS_LEN * POS_LEN


class _TinyRoPEModel(nn.Module):
    """Two blocks, each a (muon) Linear + a RoPE with its own learnable θ, plus
    an adamw-path bias. Exercises muon / adamw / log_theta routing together."""

    def __init__(self, n_blocks: int = 2):
        super().__init__()
        self.pos_len = POS_LEN
        self.head_dim = HEAD_DIM
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "lin": nn.Linear(HEAD_DIM, HEAD_DIM, bias=True),
                    "rope": RoPE(
                        pos_len=POS_LEN, head_dim=HEAD_DIM, num_rotations=NUM_ROT
                    ),
                }
            )
            for _ in range(n_blocks)
        )

    def forward(self, x):  # x: (B, S, H, D)
        for blk in self.blocks:
            x = blk["rope"](x)
            x = blk["lin"](x)
        return x


def _rand_input(batch: int = 2, heads: int = 3) -> torch.Tensor:
    return torch.randn(batch, SEQ_LEN, heads, HEAD_DIM)


def _make_opt(groups, wd_factors) -> ConvMuon:
    return ConvMuon(
        groups,
        wd_factors=wd_factors,
        weight_decay=0.02,
        adam_weight_decay=0.02,
        adam_lr_ratio=1.0,
        rms_rate=0.2,
    )


# ---------------------------------------------------------------------------
# 1. RoPE learnability
# ---------------------------------------------------------------------------


class RoPELearnableTest(unittest.TestCase):
    def test_log_theta_is_learnable_and_seeded(self):
        r = RoPE(pos_len=POS_LEN, head_dim=HEAD_DIM, num_rotations=NUM_ROT)
        self.assertIsInstance(r.log_theta, torch.nn.Parameter)
        self.assertTrue(r.log_theta.requires_grad)
        self.assertAlmostEqual(r.log_theta.item(), math.log(ROPE_THETA), places=6)

    def test_per_block_independent_params(self):
        model = _TinyRoPEModel()
        thetas = [blk["rope"].log_theta for blk in model.blocks]
        # Distinct Parameter objects → each block's θ is independently learnable.
        self.assertEqual(len({id(t) for t in thetas}), len(thetas))
        ids = {id(p) for p in model.parameters()}
        self.assertTrue(all(id(t) in ids for t in thetas))

    def test_init_table_matches_fixed_theta(self):
        """Recomputed cos/sin at init must equal the documented θ=100 table."""
        r = RoPE(pos_len=POS_LEN, head_dim=HEAD_DIM, num_rotations=NUM_ROT)
        theta_idx, angle_projs = spiral_rope_structure(NUM_ROT, HEAD_DIM, POS_LEN)
        t = np.arange(HEAD_DIM // 4)
        thetas = ROPE_THETA ** (-t / (HEAD_DIM // 4))
        rot = thetas[theta_idx] * angle_projs
        cos, sin = r._cos_sin()
        np.testing.assert_allclose(cos.detach().numpy(), np.cos(rot), atol=1e-6)
        np.testing.assert_allclose(sin.detach().numpy(), np.sin(rot), atol=1e-6)

    def test_grad_flows_to_log_theta(self):
        torch.manual_seed(0)
        r = RoPE(pos_len=POS_LEN, head_dim=HEAD_DIM, num_rotations=NUM_ROT)
        x = _rand_input()
        # RoPE preserves L2 norm, so a sum-of-squares loss has an *exactly* zero
        # θ-gradient. Project onto a fixed vector (rotation-sensitive) instead.
        w = torch.randn_like(x)
        (r(x) * w).sum().backward()
        self.assertIsNotNone(r.log_theta.grad)
        self.assertTrue(torch.isfinite(r.log_theta.grad).all())
        self.assertGreater(r.log_theta.grad.abs().item(), 1e-6)

    def test_changing_theta_changes_output(self):
        torch.manual_seed(0)
        r = RoPE(pos_len=POS_LEN, head_dim=HEAD_DIM, num_rotations=NUM_ROT)
        x = _rand_input()
        out1 = r(x).clone()
        with torch.no_grad():
            r.log_theta.add_(0.5)
        self.assertFalse(torch.allclose(out1, r(x)))

    def test_derived_tables_not_persisted(self):
        """Only `log_theta` is checkpointed; derived/structural tables are not."""
        sd = RoPE(
            pos_len=POS_LEN, head_dim=HEAD_DIM, num_rotations=NUM_ROT
        ).state_dict()
        self.assertEqual(list(sd.keys()), ["log_theta"])


# ---------------------------------------------------------------------------
# 2. Backward-compatible checkpoint loading
# ---------------------------------------------------------------------------


class RoPELoadCompatTest(unittest.TestCase):
    def _legacy_state_dict(self, model: _TinyRoPEModel) -> dict:
        """Mimic a pre-learnable-θ checkpoint: drop `log_theta`, add the baked
        persistent buffers the old code saved."""
        legacy = {
            k: v.clone()
            for k, v in model.state_dict().items()
            if not k.endswith(".log_theta")
        }
        for i in range(len(model.blocks)):
            pre = f"blocks.{i}.rope."
            legacy[pre + "_rope_cos"] = torch.randn(SEQ_LEN, HEAD_DIM)
            legacy[pre + "_rope_sin"] = torch.randn(SEQ_LEN, HEAD_DIM)
            legacy[pre + "_pair_swap_indices"] = torch.zeros(HEAD_DIM, dtype=torch.long)
            legacy[pre + "_sign_cos"] = torch.ones(HEAD_DIM)
        return legacy

    def test_legacy_checkpoint_loads_strict_and_seeds_theta(self):
        src = _TinyRoPEModel()
        legacy = self._legacy_state_dict(src)
        # Tag a real weight so we can confirm non-θ weights actually load.
        legacy["blocks.0.lin.weight"].fill_(0.123)

        model = _TinyRoPEModel()
        # strict=True must succeed despite stale buffers + missing log_theta.
        model.load_state_dict(legacy, strict=True)

        for blk in model.blocks:
            self.assertAlmostEqual(
                blk["rope"].log_theta.item(), math.log(ROPE_THETA), places=5
            )
        self.assertTrue(
            torch.allclose(
                model.blocks[0]["lin"].weight,
                torch.full_like(model.blocks[0]["lin"].weight, 0.123),
            )
        )

    def test_existing_log_theta_not_overwritten(self):
        """A current-format checkpoint keeps its saved θ (hook only fills gaps)."""
        sd = _TinyRoPEModel().state_dict()
        sd["blocks.0.rope.log_theta"] = torch.tensor(1.234)
        model = _TinyRoPEModel()
        model.load_state_dict(sd, strict=True)
        self.assertAlmostEqual(
            model.blocks[0]["rope"].log_theta.item(), 1.234, places=5
        )


# ---------------------------------------------------------------------------
# 3. Optimizer routing + resume remap
# ---------------------------------------------------------------------------


class RoPEOptimizerTest(unittest.TestCase):
    def test_log_theta_routes_to_adamw_zero_wd(self):
        model = _TinyRoPEModel()
        groups, wd = build_convmuon_param_groups(
            model, exclude_layers=_MUON_EXCLUDE_LAYERS_TORCH
        )
        gname = {g["group"]: g for g in groups}
        adam_ids = {id(p) for p in gname["adamw"]["params"]}
        muon_ids = {id(p) for p in gname["muon"]["params"]}
        lt = [p for n, p in model.named_parameters() if n.endswith(".log_theta")]
        self.assertEqual(len(lt), 2)
        for p in lt:
            self.assertIn(id(p), adam_ids)
            self.assertNotIn(id(p), muon_ids)
            self.assertEqual(wd[id(p)][1], 0.0)  # zero weight-decay factor

    def _old_format_state(self, model):
        """A genuine pre-θ optimizer state: optimize the params minus log_theta,
        step twice to populate Adam m/v + Muon buffers, return its state_dict."""
        old_named = [
            (n, p) for n, p in model.named_parameters() if not n.endswith(".log_theta")
        ]
        og, owd = build_convmuon_param_groups(
            old_named, exclude_layers=_MUON_EXCLUDE_LAYERS_TORCH
        )
        old_opt = _make_opt(og, owd)
        for _, p in old_named:
            p.grad = torch.randn_like(p)
        old_opt.step()
        old_opt.step()
        return old_opt, old_named

    def test_resume_remap_preserves_state_and_steps(self):
        torch.manual_seed(0)
        model = _TinyRoPEModel()
        old_opt, old_named = self._old_format_state(model)
        saved = old_opt.state_dict()  # old-format (no log_theta entries)

        ng, nwd = build_convmuon_param_groups(
            model, exclude_layers=_MUON_EXCLUDE_LAYERS_TORCH
        )
        new_opt = _make_opt(ng, nwd)
        remapped = _maybe_remap_resume_state(new_opt, saved, model)
        new_opt.load_state_dict(remapped)

        # Existing params: optimizer state byte-identical to the old optimizer.
        for _, p in old_named:
            self.assertGreater(len(new_opt.state.get(p, {})), 0)
            for k, v in old_opt.state[p].items():
                nv = new_opt.state[p][k]
                if torch.is_tensor(v):
                    self.assertTrue(torch.equal(nv, v), msg=f"mismatch on {k}")
                else:
                    self.assertEqual(nv, v)

        lt = [p for n, p in model.named_parameters() if n.endswith(".log_theta")]
        # log_theta starts stateless (fresh), then gains state on the first step.
        for p in lt:
            self.assertEqual(len(new_opt.state.get(p, {})), 0)

        before = [p.item() for p in lt]
        model.zero_grad()
        out = model(_rand_input())
        (out * torch.randn_like(out)).sum().backward()  # rotation-sensitive loss
        new_opt.step()  # must not raise (mixed-group lazy init) and move θ
        after = [p.item() for p in lt]
        self.assertTrue(all(a != b for a, b in zip(before, after)))
        for p in lt:
            self.assertGreater(len(new_opt.state[p]), 0)

    def test_resume_is_noop_for_current_format(self):
        model = _TinyRoPEModel()
        ng, nwd = build_convmuon_param_groups(
            model, exclude_layers=_MUON_EXCLUDE_LAYERS_TORCH
        )
        opt = _make_opt(ng, nwd)
        for _, p in model.named_parameters():
            p.grad = torch.randn_like(p)
        opt.step()
        saved = opt.state_dict()  # already includes log_theta
        self.assertIs(_maybe_remap_resume_state(opt, saved, model), saved)

    def test_resume_preserves_wrapper_format(self):
        model = _TinyRoPEModel()
        old_opt, _ = self._old_format_state(model)
        wrapped = {"inner": old_opt.state_dict(), "scaler": {"sentinel": 7}}

        ng, nwd = build_convmuon_param_groups(
            model, exclude_layers=_MUON_EXCLUDE_LAYERS_TORCH
        )
        new_opt = _make_opt(ng, nwd)
        out = _maybe_remap_resume_state(new_opt, wrapped, model)

        self.assertEqual(out["scaler"], {"sentinel": 7})  # untouched
        self.assertEqual(
            sum(len(g["params"]) for g in out["inner"]["param_groups"]),
            sum(len(g["params"]) for g in new_opt.param_groups),
        )


if __name__ == "__main__":
    unittest.main()
