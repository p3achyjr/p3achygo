"""Batch Renorm exponential ramp (KataGo-style) and the step setter."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend_torch.model_layers_common import (  # noqa: E402
    FixScaleBatchNorm,
    FixScaleBatchRenorm,
    enable_batch_renorm,
    set_bn_eps,
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


def _fixscale_bn(eps=1e-3):
    torch.manual_seed(0)
    m = FixScaleBatchNorm(3)
    m.eps = eps
    m.set_scale(0.5)
    with torch.no_grad():
        m.beta.copy_(torch.tensor([0.1, -0.2, 0.3]))
        m.running_mean.copy_(torch.tensor([0.5, -1.0, 2.0]))
        m.running_var.copy_(torch.tensor([4.0, 0.25, 1.0]))
    return m


def test_fixscale_brn_swap_and_eval():
    m = _fixscale_bn()
    ref = _fixscale_bn()
    seq = torch.nn.Sequential(m)
    assert enable_batch_renorm(seq, 3.0, 5.0, 0) == 1
    assert type(m) is FixScaleBatchRenorm
    assert set(m.state_dict()) == set(ref.state_dict())
    m.eval(), ref.eval()
    x = torch.randn(8, 3, 5, 5)
    assert torch.allclose(m(x), ref(x), atol=1e-6)


def test_fixscale_brn_train_forward():
    """Training forward = (x - mu) / sigma_running clipped-renorm form, then
    * scale + beta; with no clipping (rmax/dmax huge) it equals the running-stat
    function, and running stats update like BatchNorm."""
    m = _fixscale_bn()
    seq = torch.nn.Sequential(m)
    enable_batch_renorm(seq, 1e6, 1e6, 0)
    m.train()
    x = torch.randn(16, 3, 5, 5) * 3.0 + 1.0
    rm, rv = m.running_mean.clone(), m.running_var.clone()
    y = m(x)
    want = (x - rm[None, :, None, None]) / (rv[None, :, None, None] + m.eps).sqrt()
    want = want * 0.5 + m.beta[None, :, None, None]
    assert torch.allclose(y, want, atol=1e-4)
    assert not torch.allclose(m.running_mean, rm)
    y.sum().backward()
    assert m.beta.grad is not None


def test_set_bn_eps_affine_free():
    m = _fixscale_bn()
    m.num_batches_tracked.zero_()
    assert set_bn_eps(torch.nn.Sequential(m), 1e-4) == 1 and m.eps == 1e-4
    m2 = _fixscale_bn()
    m2.num_batches_tracked.fill_(1)
    try:
        set_bn_eps(torch.nn.Sequential(m2), 1e-4)
        raise AssertionError("expected ValueError on a trained affine-free BN")
    except ValueError:
        pass


if __name__ == "__main__":
    test_exp_ramp_rmax()
    test_fixscale_brn_swap_and_eval()
    test_fixscale_brn_train_forward()
    test_set_bn_eps_affine_free()
    print("ok")
