"""Verify the torch backend's `recompute_bn_statistics` computes the EXACT
cumulative average of per-batch BN statistics (no EMA undershoot).

Mirrors `torch.optim.swa_utils.update_bn` semantics: after N batches,
`bn.running_mean` should equal `mean(per-batch means)` to fp32 precision.

    P3ACHYGO_BACKEND=torch python -m pytest python/test/recompute_bn_torch_test.py
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
from backend_torch.model_utils import recompute_bn_statistics


class _Stub(nn.Module):
    """Single-BN stub with a 2-input forward to match what
    `recompute_bn_statistics` actually calls."""

    def __init__(self, momentum: float = 0.01, eps: float = 1e-3):
        super().__init__()
        self.bn = nn.BatchNorm2d(4, momentum=momentum, eps=eps)

    def forward(self, board, _game):
        return self.bn(board)


class TorchRecomputeBnTest(unittest.TestCase):
    def test_running_mean_matches_cumulative_average(self):
        model = _Stub().eval()
        torch.manual_seed(0)

        N_BATCHES = 30
        batches = [torch.randn(8, 4, 5, 5) for _ in range(N_BATCHES)]
        game = torch.zeros(1)
        ds = [(b, game) for b in batches]

        # Per-batch unbiased variance — `nn.BatchNorm2d` stores the
        # Bessel-corrected variance for the running estimate.
        batch_means = [b.mean(dim=(0, 2, 3)).numpy() for b in batches]
        batch_vars = [
            b.permute(1, 0, 2, 3).reshape(4, -1).var(dim=1, unbiased=True).numpy()
            for b in batches
        ]
        expected_mean = np.mean(batch_means, axis=0)
        expected_var = np.mean(batch_vars, axis=0)

        # Pre-fill running stats with junk to verify the reset.
        with torch.no_grad():
            model.bn.running_mean.fill_(99.0)
            model.bn.running_var.fill_(99.0)

        recompute_bn_statistics(model, iter(ds), num_batches=N_BATCHES)

        np.testing.assert_allclose(
            model.bn.running_mean.numpy(),
            expected_mean,
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            model.bn.running_var.numpy(),
            expected_var,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_momentum_restored_after_recompute(self):
        model = _Stub(momentum=0.01).eval()
        ds = iter([(torch.randn(2, 4, 3, 3), torch.zeros(1))])
        recompute_bn_statistics(model, ds, num_batches=1)
        self.assertEqual(model.bn.momentum, 0.01)

    def test_eval_state_restored_after_recompute(self):
        model = _Stub().eval()
        ds = iter([(torch.randn(2, 4, 3, 3), torch.zeros(1))])
        recompute_bn_statistics(model, ds, num_batches=1)
        self.assertFalse(model.training, "model should be back in eval mode")


if __name__ == "__main__":
    unittest.main()
