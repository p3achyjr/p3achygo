"""Batch Renorm exponential ramp (KataGo-style) and the step setter."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend_torch.model_layers_common import (  # noqa: E402
    enable_batch_renorm,
    set_brn_step,
)


def _clipped_r(t: int) -> float:
    """r for one training forward at ramp step t, with batch std >> running std
    so r sits at the rmax(t) clip."""
    bn = torch.nn.Sequential(torch.nn.BatchNorm2d(1, momentum=0.0, eps=1e-4))
    enable_batch_renorm(bn, rmax=3.0, dmax=5.0, ramp_batches=100, ramp_exp=True)
    set_brn_step(bn, t)
    m = bn[0]
    m.running_var.fill_(1e-6)
    x = torch.randn(64, 1, 4, 4) * 10.0
    y = bn(x).detach()
    std = (x.var(unbiased=False) + m.eps).sqrt()
    # y = (x - mean) / std * r + d with gamma 1, beta 0; recover r from the slope.
    xs = (x - x.mean()) / std
    r = float((y - y.mean()).flatten() @ xs.flatten() / (xs.flatten() @ xs.flatten()))
    return r


def test_exp_ramp_rmax():
    for t in (0, 100, 1000):
        # brn_t is incremented before use, so the forward sees t + 1.
        a = 1.0 - math.exp(-(t + 1) / 100)
        expected = 1.0 + 2.0 * a
        r = _clipped_r(t)
        assert abs(r - expected) < 1e-3, (t, r, expected)


if __name__ == "__main__":
    test_exp_ramp_rmax()
    print("ok")
