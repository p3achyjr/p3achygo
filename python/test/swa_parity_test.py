"""Verify the torch backend's `swa_avg_weights` matches a deterministic
numpy-reference EMA, and that the cascading EMA is numerically stable
across long snapshot lists. Backend-agnostic numpy reference tests are
also included (they run under either backend).

The TF-side parity test lives in `swa_parity_tf_test.py`.

    P3ACHYGO_BACKEND=torch python -m pytest python/test/swa_parity_test.py
"""

from __future__ import annotations

import unittest

import numpy as np


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


class SWAParityTest(unittest.TestCase):
    """Backend-independent parity against a numpy reference + finite checks."""

    def setUp(self):
        self.snapshots = _make_random_snapshots(num_snaps=21)
        self.momentum = 0.75
        self.ref = _numpy_reference_swa(self.snapshots, self.momentum)

    def test_no_nans_or_infs_in_reference(self):
        for arr in self.ref:
            self.assertFalse(np.isnan(arr).any())
            self.assertFalse(np.isinf(arr).any())

    def test_seed_weight_decays_geometrically(self):
        """Seed contribution after N updates with momentum m is m^N. With
        m=0.75, N=20, that's ~0.32% — i.e., the EMA is dominated by recent
        snapshots, which is the desired SWA behavior."""
        N = len(self.snapshots) - 1
        seed_weight = self.momentum**N
        self.assertLess(seed_weight, 0.01)
        self.assertGreater(seed_weight, 1e-4)

    def test_torch_swa_matches_numpy_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        from backend_torch.model_utils import swa_avg_weights as torch_swa

        # Convert each snapshot to a state_dict-shaped dict (torch tensors).
        # Use stable keys so the dict iteration order matches the numpy list.
        keys = [f"k{i}" for i in range(len(self.snapshots[0]))]
        torch_snapshots = [
            {k: torch.from_numpy(arr.copy()) for k, arr in zip(keys, snap)}
            for snap in self.snapshots
        ]
        out = torch_swa(torch_snapshots, swa_momentum=self.momentum)
        for k, ref_arr in zip(keys, self.ref):
            np.testing.assert_allclose(
                out[k].numpy(),
                ref_arr.astype(np.float32),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_torch_skips_integer_bn_counters(self):
        """`num_batches_tracked` (int64) must NOT be averaged — it has no
        meaningful EMA. Verify it stays at the seed value."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        from backend_torch.model_utils import swa_avg_weights as torch_swa

        snaps = []
        for i in range(5):
            snaps.append(
                {
                    "weight": torch.full((4,), float(i), dtype=torch.float32),
                    "num_batches_tracked": torch.tensor(i, dtype=torch.long),
                }
            )
        out = torch_swa(snaps, swa_momentum=0.5)
        # `weight` should be EMA'd; `num_batches_tracked` should be the seed value.
        self.assertEqual(out["num_batches_tracked"].dtype, torch.long)
        self.assertEqual(out["num_batches_tracked"].item(), 0)
        # Sanity: weight EMA != seed value
        self.assertGreater(out["weight"][0].item(), 0.0)

    def test_torch_long_cascade_is_stable(self):
        """50-snapshot cascade with momentum=0.95 — verifies no fp32 underflow
        or accumulation drift over a long EMA chain."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        from backend_torch.model_utils import swa_avg_weights as torch_swa

        snaps = _make_random_snapshots(num_snaps=50, seed=1)
        ref = _numpy_reference_swa(snaps, 0.95)
        keys = [f"k{i}" for i in range(len(snaps[0]))]
        torch_snaps = [
            {k: torch.from_numpy(a.copy()) for k, a in zip(keys, s)} for s in snaps
        ]
        out = torch_swa(torch_snaps, swa_momentum=0.95)
        for k, ref_arr in zip(keys, ref):
            np.testing.assert_allclose(
                out[k].numpy(),
                ref_arr.astype(np.float32),
                rtol=1e-4,
                atol=1e-5,
            )
            self.assertFalse(np.isnan(out[k].numpy()).any())
            self.assertFalse(np.isinf(out[k].numpy()).any())

    def test_torch_preserves_dtype_round_trip(self):
        """Output dtype matches the seed's dtype per-key, even though the
        EMA promotes to fp32 internally."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        from backend_torch.model_utils import swa_avg_weights as torch_swa

        snaps = []
        for i in range(3):
            snaps.append(
                {
                    "fp32": torch.full((4,), float(i), dtype=torch.float32),
                    "fp16": torch.full((4,), float(i), dtype=torch.float16),
                    "int": torch.tensor(i, dtype=torch.long),
                }
            )
        out = torch_swa(snaps, swa_momentum=0.5)
        self.assertEqual(out["fp32"].dtype, torch.float32)
        self.assertEqual(out["fp16"].dtype, torch.float16)
        self.assertEqual(out["int"].dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
