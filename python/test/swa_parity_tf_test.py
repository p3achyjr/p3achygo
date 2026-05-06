"""Verify the keras (TF) backend's `swa_avg_weights` matches a deterministic
numpy-reference EMA. Companion to `swa_parity_test.py` which covers the
backend-agnostic + torch path.

    P3ACHYGO_BACKEND=tensorflow python -m pytest python/test/swa_parity_tf_test.py
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pytest

if os.environ.get("P3ACHYGO_BACKEND", "tensorflow") == "torch":
    pytest.skip("TF backend test", allow_module_level=True)


from backend_tf.model_utils import swa_avg_weights as tf_swa


def _numpy_reference_swa(weight_lists, momentum: float):
    """Plain-numpy reference: same math the production code claims to do."""
    swa = [a.astype(np.float64).copy() for a in weight_lists[0]]
    for snap in weight_lists[1:]:
        swa = [
            s * momentum + a.astype(np.float64) * (1 - momentum)
            for s, a in zip(swa, snap)
        ]
    return swa


def _make_random_snapshots(num_snaps=21, seed=0):
    """Five tensors per snapshot — modeled on a few BN/Conv weight shapes."""
    rng = np.random.default_rng(seed)
    shapes = [(8,), (8,), (16, 8, 3, 3), (4,), (4, 16)]
    snapshots = []
    for _ in range(num_snaps):
        snap = [rng.standard_normal(s).astype(np.float32) for s in shapes]
        snapshots.append(snap)
    return snapshots


class TFSWAParityTest(unittest.TestCase):
    def test_tf_swa_matches_numpy_reference(self):
        snapshots = _make_random_snapshots(num_snaps=21)
        momentum = 0.75
        ref = _numpy_reference_swa(snapshots, momentum)

        out = tf_swa(snapshots, swa_momentum=momentum)
        for arr, ref_arr in zip(out, ref):
            np.testing.assert_allclose(
                arr,
                ref_arr.astype(np.float32),
                rtol=1e-5,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
