"""Verify the keras (TF) backend's `recompute_bn_statistics` computes the
EXACT cumulative average of per-batch BN statistics (no EMA approximation).

Uses the per-step momentum trick `bn.momentum = (i-1)/i` to collapse the
EMA recurrence into a cumulative average.

    P3ACHYGO_BACKEND=tensorflow python -m pytest python/test/recompute_bn_tf_test.py
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest

if os.environ.get("P3ACHYGO_BACKEND", "tensorflow") == "torch":
    pytest.skip("TF backend test", allow_module_level=True)


import keras
from backend_tf.model_utils import recompute_bn_statistics


class _Stub(keras.Model):
    """Single-BN stub. NHWC layout; 2-input call matches what
    `recompute_bn_statistics` actually invokes."""

    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-3):
        super().__init__()
        self.bn = keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def call(self, board, game, training=False):
        del game  # unused; signature parity with the real model
        return self.bn(board, training=training)


class _DS:
    """Stub dataset object with a `take(n)` method (mirrors tf.data.Dataset)."""

    def __init__(self, items):
        self.items = items

    def take(self, n):
        return self.items[:n]


class TFRecomputeBnTest(unittest.TestCase):
    def test_running_mean_matches_cumulative_average(self):
        rng = np.random.default_rng(0)
        N_BATCHES = 30
        batches = [
            rng.standard_normal((8, 5, 5, 4)).astype(np.float32)
            for _ in range(N_BATCHES)
        ]
        game = np.zeros((1,), dtype=np.float32)

        model = _Stub()
        # Build via a dummy call so BN allocates moving_mean/moving_variance.
        _ = model(batches[0], game, training=False)

        # Pre-fill moving stats with junk to verify the reset.
        model.bn.moving_mean.assign(
            np.full(model.bn.moving_mean.shape, 99.0, dtype=np.float32)
        )
        model.bn.moving_variance.assign(
            np.full(model.bn.moving_variance.shape, 99.0, dtype=np.float32)
        )

        # keras BN uses biased (population) variance for the per-batch stat.
        batch_means = [b.mean(axis=(0, 1, 2)) for b in batches]
        batch_vars = [b.var(axis=(0, 1, 2)) for b in batches]
        expected_mean = np.mean(batch_means, axis=0)
        expected_var = np.mean(batch_vars, axis=0)

        ds = _DS([(b, game) for b in batches])
        recompute_bn_statistics(model, ds, num_batches=N_BATCHES)

        np.testing.assert_allclose(
            np.asarray(model.bn.moving_mean),
            expected_mean,
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(model.bn.moving_variance),
            expected_var,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_momentum_restored_after_recompute(self):
        model = _Stub(momentum=0.99)
        b = np.zeros((2, 3, 3, 4), dtype=np.float32)
        g = np.zeros((1,), dtype=np.float32)
        _ = model(b, g, training=False)
        recompute_bn_statistics(model, _DS([(b, g)]), num_batches=1)
        self.assertAlmostEqual(model.bn.momentum, 0.99)


if __name__ == "__main__":
    unittest.main()
