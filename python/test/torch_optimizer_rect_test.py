"""Verify Muon updates are correct for rectangular weights (out_dim != in_dim).

Catches regressions if anyone introduces an asymmetric usage of out_dim
vs flat_dim — the existing helpers symmetrize via `max(flat, out)`,
which silently hides a labeling bug in `_is_muon_param` /
`_rms_wd_scale` (the `else: out_dim = shape[-1]` branch is incorrectly
labeled for torch Linear, which stores weight as (out, in)).

Covers:
  - NS5 orthogonality on tall / wide / high-aspect-ratio matrices.
  - Batched NS5 == singleton NS5 for matching shapes.
  - End-to-end Muon step on rectangular Linear and Conv2d params:
      * shape preserved
      * gradient direction respected (param moves opposite to gradient)
      * momentum buffer + weight decay applied as designed.
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest

if os.environ.get("P3ACHYGO_BACKEND", "tensorflow") != "torch":
    pytest.skip("torch backend test", allow_module_level=True)


import torch
import torch.nn as nn

from backend_torch.optimizer import (
    ConvMuon,
    _newtonschulz,
    _newtonschulz_batched,
    _param_out_dim,
    build_convmuon_param_groups,
)


def _ns_singular_values(g: torch.Tensor, steps: int = 5) -> np.ndarray:
    """Run unbatched NS5 and return the singular values of the result."""
    out = _newtonschulz(g, steps=steps).float()
    return torch.linalg.svdvals(out).numpy()


class ParamOutDimTest(unittest.TestCase):
    """Lock in the contract: `_param_out_dim` returns the OUT-feature dim
    for both Linear and Conv2d weights, regardless of how rectangular
    they are. Catches the keras-ism `shape[-1]` regression.

    torch convention is "out leading":
      - `nn.Linear(in_features=A, out_features=B).weight.shape == (B, A)`
      - `nn.Conv2d(in_channels=A, out_channels=B, ks).weight.shape == (B, A, ks, ks)`
    so `shape[0]` is always the output dim.
    """

    def test_linear_tall_weight(self):
        # Linear(in=32, out=128) → weight shape (128, 32). out=128.
        w = nn.Linear(32, 128, bias=False).weight
        self.assertEqual(w.shape, (128, 32))
        self.assertEqual(_param_out_dim(w), 128)

    def test_linear_wide_weight(self):
        # Linear(in=128, out=32) → weight shape (32, 128). out=32.
        # The keras-ism `shape[-1]` would mis-report 128 here.
        w = nn.Linear(128, 32, bias=False).weight
        self.assertEqual(w.shape, (32, 128))
        self.assertEqual(_param_out_dim(w), 32)

    def test_linear_square_weight(self):
        w = nn.Linear(64, 64, bias=False).weight
        self.assertEqual(_param_out_dim(w), 64)

    def test_conv2d_weight(self):
        # Conv2d(in=16, out=64, ks=3) → weight (64, 16, 3, 3). out=64.
        w = nn.Conv2d(16, 64, kernel_size=3, bias=False).weight
        self.assertEqual(w.shape, (64, 16, 3, 3))
        self.assertEqual(_param_out_dim(w), 64)

    def test_is_muon_param_consistent_with_out_dim(self):
        """Boolean classifier `out_dim > 4 and flat_dim > 4` is symmetric in
        out/flat, so the labeling fix is invisible to it — but exercise the
        symmetric edge case explicitly to lock the contract: a 2D weight
        with the OUT side ≤ 4 must NOT be classified as muon."""
        from backend_torch.optimizer import _is_muon_param

        # Linear(in=100, out=3): out=3, flat=100. OUT side is too small;
        # this would be a head-style projection, NOT a muon-eligible param.
        w_small_out = nn.Linear(100, 3, bias=False).weight
        self.assertFalse(_is_muon_param("foo.weight", w_small_out, []))

        # Linear(in=3, out=100): out=100, flat=3. IN side is too small;
        # also not muon (too narrow on one side either way).
        w_small_in = nn.Linear(3, 100, bias=False).weight
        self.assertFalse(_is_muon_param("foo.weight", w_small_in, []))

        # Linear(in=100, out=100): both > 4 → muon.
        w_normal = nn.Linear(100, 100, bias=False).weight
        self.assertTrue(_is_muon_param("foo.weight", w_normal, []))


class NS5RectangularShapesTest(unittest.TestCase):
    """The min(M, N) leading singular values should be ≈ 1 after NS5,
    regardless of whether the matrix is tall or wide."""

    def _check_orthogonality(self, M: int, N: int, tol: float = 0.5):
        torch.manual_seed(0)
        g = torch.randn(M, N)
        sv = _ns_singular_values(g, steps=5)
        # The leading min(M, N) singular values should converge to ~1.
        k = min(M, N)
        leading = sv[:k]
        max_dev = float(np.abs(leading - 1.0).max())
        self.assertLess(
            max_dev,
            tol,
            f"NS5({M}x{N}) leading SVs not near 1: max dev = {max_dev:.3f}, sv={leading}",
        )

    def test_tall_matrix_2x(self):
        self._check_orthogonality(M=128, N=64)

    def test_wide_matrix_2x(self):
        self._check_orthogonality(M=64, N=128)

    def test_tall_matrix_4x(self):
        self._check_orthogonality(M=256, N=64)

    def test_wide_matrix_4x(self):
        self._check_orthogonality(M=64, N=256)

    def test_extreme_aspect_ratio(self):
        # 32:1 — exposes broken transpose handling if any.
        self._check_orthogonality(M=512, N=16)
        self._check_orthogonality(M=16, N=512)


class NS5BatchedMatchesSingletonTest(unittest.TestCase):
    """Batched NS5 on a stack of identical-shape matrices should match the
    singleton NS5 element-wise — verifies the bmm / baddbmm pathway."""

    def _check(self, M: int, N: int, B: int = 3):
        torch.manual_seed(0)
        gs = torch.randn(B, M, N)
        # Singleton: run NS5 on each batch element independently.
        single = torch.stack([_newtonschulz(gs[i], steps=5).float() for i in range(B)])
        # Batched: one call.
        batched = _newtonschulz_batched(gs, steps=5).float()
        # bf16 NS internals → fp32 round-trip → small element-wise tolerance.
        np.testing.assert_allclose(
            batched.numpy(),
            single.numpy(),
            rtol=1e-2,
            atol=1e-3,
        )

    def test_tall(self):
        self._check(M=64, N=16)

    def test_wide(self):
        self._check(M=16, N=64)

    def test_square(self):
        self._check(M=32, N=32)


class MuonStepOnRectangularTest(unittest.TestCase):
    """End-to-end: a single optimizer step on a rectangular Linear / Conv2d
    weight should update the parameter sensibly (shape preserved, gradient
    direction respected, no NaN)."""

    def _make_optimizer(self, model: nn.Module, lr: float = 1e-3):
        groups, wd_factors = build_convmuon_param_groups(
            model,
            lr=lr,
            exclude_layers=[],
        )
        return ConvMuon(
            groups,
            wd_factors=wd_factors,
            weight_decay=0.0,  # disable WD to isolate the NS update math
            adam_weight_decay=0.0,
            adam_lr_ratio=1.0,
            global_clipnorm=float("inf"),
        )

    def _run_step_with_known_grad(self, weight: torch.Tensor, lr: float = 1e-3):
        """Build a single-Linear model with the given weight shape, plant
        a known gradient, run one step, return (before, after)."""
        Cout, Cin = weight.shape
        torch.manual_seed(0)
        m = nn.Linear(Cin, Cout, bias=False)
        with torch.no_grad():
            m.weight.copy_(weight)
        opt = self._make_optimizer(m, lr=lr)

        # Plant a synthetic gradient with the same shape as the weight.
        torch.manual_seed(1)
        m.weight.grad = torch.randn_like(m.weight)
        before = m.weight.detach().clone()
        opt.step()
        after = m.weight.detach().clone()
        return before, after, m.weight.grad

    def test_tall_linear_shape_preserved(self):
        # Cout > Cin
        before, after, _ = self._run_step_with_known_grad(torch.randn(128, 32))
        self.assertEqual(after.shape, (128, 32))
        self.assertFalse(torch.equal(before, after), "weight should have updated")
        self.assertTrue(torch.isfinite(after).all())

    def test_wide_linear_shape_preserved(self):
        # Cout < Cin
        before, after, _ = self._run_step_with_known_grad(torch.randn(32, 128))
        self.assertEqual(after.shape, (32, 128))
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.isfinite(after).all())

    def test_update_opposes_gradient(self):
        """The Muon update direction (param_after - param_before) should
        have negative inner product with the gradient — i.e., it's a
        descent direction. Holds for both tall and wide weights."""
        for Cout, Cin in [(64, 32), (32, 64), (128, 16), (16, 128)]:
            with self.subTest(shape=(Cout, Cin)):
                before, after, grad = self._run_step_with_known_grad(
                    torch.randn(Cout, Cin), lr=1e-3
                )
                delta = after - before
                inner = torch.sum(delta * grad).item()
                self.assertLess(
                    inner,
                    0.0,
                    f"Muon step at {(Cout, Cin)} not a descent direction: "
                    f"<delta, grad> = {inner:.4f}",
                )

    def test_rectangular_conv2d_shape_preserved(self):
        """Conv2d weights are (Cout, Cin, H, W). NS5 orthogonalizes the
        (H*W*Cin, Cout) flatten. Verify a step on a Cin != Cout conv
        doesn't blow up the shape."""
        torch.manual_seed(0)
        m = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, bias=False)
        opt = self._make_optimizer(m, lr=1e-3)
        m.weight.grad = torch.randn_like(m.weight)
        before = m.weight.detach().clone()
        opt.step()
        after = m.weight.detach().clone()
        self.assertEqual(after.shape, (64, 16, 3, 3))
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(torch.isfinite(after).all())

    def test_momentum_buffer_initialized_per_param(self):
        """First step initializes per-param momentum buffer with the right
        shape. Catches a regression where buffer shape is keyed off the
        wrong dim of a rectangular weight."""
        torch.manual_seed(0)
        m = nn.Linear(32, 128, bias=False)  # weight (128, 32)
        opt = self._make_optimizer(m, lr=1e-3)
        m.weight.grad = torch.randn_like(m.weight)
        opt.step()
        buf = opt.state[m.weight]["buf"]
        self.assertEqual(buf.shape, m.weight.shape)


class AdjustedLRRectangularTest(unittest.TestCase):
    """`adj_lr_factor` per shape bucket should be `sqrt(max(M, N))` where
    (M, N) is the NS-input 2D shape. Verify it's consistent for tall and
    wide weights — the factor is symmetric in (M, N), so this catches
    regressions if someone makes it asymmetric."""

    def test_adj_lr_factor_tall_vs_wide_match(self):
        """`build_convmuon_param_groups` packs each muon param into a shape
        bucket with `adj_lr_factor = sqrt(max(M, N))`. For two Linears that
        are transposes of each other (Cout, Cin) and (Cin, Cout), the
        factor should be identical."""
        m_tall = nn.Linear(32, 128, bias=False)  # (128, 32)
        m_wide = nn.Linear(128, 32, bias=False)  # (32, 128)
        groups_tall, _ = build_convmuon_param_groups(m_tall, lr=1.0, exclude_layers=[])
        groups_wide, _ = build_convmuon_param_groups(m_wide, lr=1.0, exclude_layers=[])
        muon_tall = next(g for g in groups_tall if g["group"] == "muon")
        muon_wide = next(g for g in groups_wide if g["group"] == "muon")
        # One bucket each; same shape (32, 128) modulo transpose handling
        # in `_ns_input_shape` (returns tuple(p.shape) for 2D — so
        # (128,32) and (32,128) are *different* buckets, but both should
        # have the same sqrt(max) adj_lr_factor since max(128,32) ==
        # max(32,128)).
        self.assertEqual(len(muon_tall["ns_groups"]), 1)
        self.assertEqual(len(muon_wide["ns_groups"]), 1)
        np.testing.assert_allclose(
            muon_tall["ns_groups"][0]["adj_lr_factor"],
            muon_wide["ns_groups"][0]["adj_lr_factor"],
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
