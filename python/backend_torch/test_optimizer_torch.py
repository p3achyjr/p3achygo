"""
Sanity-check TorchConvMuon against a reference implementation.

torch.compile may be broken in this environment; we monkey-patch it out.
"""

import math
import sys
import os
import unittest.mock

# Patch torch.compile to a no-op before importing optimizer_torch
import torch

unittest.mock.patch("torch.compile", lambda fn, **kw: fn).start()

sys.path.insert(0, os.path.dirname(__file__))
from backend_torch.optimizer import ConvMuon, _newtonschulz

torch.manual_seed(42)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #


def check(name, a, b, atol=1e-5, rtol=1e-5):
    a, b = a.float(), b.float()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    max_err = (a - b).abs().max().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<50}  max_err={max_err:.2e}")
    return ok


# ------------------------------------------------------------------ #
# Test 1: NS singular values are in [0.5, 1.5]                       #
# ------------------------------------------------------------------ #


def test_ns_singular_values():
    print("Test 1: NS output singular values bounded in ~[0.5, 1.5]")
    for shape in [(32, 64), (64, 32), (128, 128), (3 * 3 * 32, 64)]:
        G = torch.randn(*shape)
        Q = _newtonschulz(G, steps=6).float()
        sv = torch.linalg.svdvals(Q)
        lo, hi = sv.min().item(), sv.max().item()
        ok = 0.3 < lo and hi < 1.7
        print(
            f"  [{'PASS' if ok else 'FAIL'}] shape={str(shape):<20} sv=[{lo:.3f}, {hi:.3f}]"
        )


# ------------------------------------------------------------------ #
# Test 2: Muon step matches reference formula exactly                  #
# ------------------------------------------------------------------ #


def ref_muon_step(p0, g, buf0, lr, momentum, nesterov, ns_steps, rms_rate, wd):
    """Exact reference for one Muon update. EMA momentum (matches KataGo/torch upstream)."""
    buf = buf0 * momentum + g * (1 - momentum)  # EMA
    update_g = g.lerp(buf, momentum) if nesterov else buf.clone()

    orig = update_g.shape
    g2d = update_g.reshape(-1, orig[-1]) if update_g.ndim == 4 else update_g
    flat_dim, out_dim = g2d.shape

    ortho = _newtonschulz(g2d, ns_steps).float()
    if ortho.shape != orig:
        ortho = ortho.reshape(orig)

    adj_lr = (
        lr * math.sqrt(max(flat_dim, out_dim)) * rms_rate
        if rms_rate is not None
        else lr
    )
    p1 = p0 * (1 - lr * wd) - ortho * adj_lr
    return p1, buf


def test_muon_step():
    print("\nTest 2: Single Muon step matches reference formula")
    LR, MOM, NS, RMS, WD = 1e-3, 0.95, 6, 0.2, 0.1
    all_ok = True
    for shape, tag in [((32, 64), "2D kernel"), ((3, 3, 32, 64), "4D conv kernel")]:
        p0 = torch.randn(*shape) * 0.1
        g0 = torch.randn(*shape) * 0.01

        p_ref, buf_ref = ref_muon_step(
            p0.clone(), g0, torch.zeros(*shape), LR, MOM, True, NS, RMS, WD
        )

        param = torch.nn.Parameter(p0.clone())
        opt = ConvMuon(
            [("body.kernel", param)],
            lr=LR,
            weight_decay=WD,
            momentum=MOM,
            nesterov=True,
            ns_steps=NS,
            rms_rate=RMS,
            wd_lr_exponent=None,
            global_clipnorm=None,
        )
        param.grad = g0.clone()
        opt.step()

        ok1 = check(f"param  {tag}", param.data, p_ref, atol=1e-5, rtol=1e-5)
        ok2 = check(
            f"buf    {tag}", opt.state[param]["buf"], buf_ref, atol=1e-6, rtol=1e-6
        )
        all_ok &= ok1 and ok2

    # Two-step test to verify momentum carries over
    print("  (two-step momentum carry-over)")
    p0 = torch.randn(32, 64) * 0.1
    g0 = torch.randn(32, 64) * 0.01
    g1 = torch.randn(32, 64) * 0.01

    param = torch.nn.Parameter(p0.clone())
    opt = ConvMuon(
        [("body.kernel", param)],
        lr=LR,
        weight_decay=WD,
        momentum=MOM,
        nesterov=True,
        ns_steps=NS,
        rms_rate=RMS,
        wd_lr_exponent=None,
        global_clipnorm=None,
    )
    param.grad = g0.clone()
    opt.step()
    param.grad = g1.clone()
    opt.step()

    buf0 = torch.zeros(32, 64)
    p_ref, buf_ref = ref_muon_step(p0.clone(), g0, buf0, LR, MOM, True, NS, RMS, WD)
    p_ref, buf_ref = ref_muon_step(p_ref, g1, buf_ref, LR, MOM, True, NS, RMS, WD)

    ok1 = check("param  2-step", param.data, p_ref, atol=1e-5, rtol=1e-5)
    ok2 = check("buf    2-step", opt.state[param]["buf"], buf_ref, atol=1e-6, rtol=1e-6)
    all_ok &= ok1 and ok2
    return all_ok


# ------------------------------------------------------------------ #
# Test 3: AdamW step matches reference                                #
# ------------------------------------------------------------------ #


def ref_adamw_step(p0, g, m0, v0, t, lr, beta1, beta2, eps, wd):
    m = m0 * beta1 + g * (1 - beta1)
    v = v0 * beta2 + g.square() * (1 - beta2)
    bc = math.sqrt(1 - beta2**t) / (1 - beta1**t)
    p1 = p0 * (1 - lr * wd) - lr * bc * m / (v.sqrt() + eps)
    return p1, m, v


def test_adamw_step():
    print("\nTest 3: AdamW step matches reference formula")
    LR_MAIN, RATIO = 1e-3, 0.1
    LR = LR_MAIN * RATIO
    WD_MUON, WD_ADAM = 0.1, 0.004
    B1, B2, EPS = 0.9, 0.999, 1e-8
    all_ok = True

    cases = [
        ("layer.gamma", WD_MUON * ConvMuon._WD_SCALE["gamma"], "gamma"),
        ("layer.beta", WD_MUON * ConvMuon._WD_SCALE["beta"], "beta"),
        ("body.layer.bias", WD_MUON * ConvMuon._WD_SCALE["body_bias"], "body_bias"),
        (
            "policy_head.out.bias",
            WD_MUON * ConvMuon._WD_SCALE["head_bias"],
            "head_bias",
        ),
        ("policy_head.out.kernel", WD_ADAM, "excluded head"),
    ]
    for name, expected_wd, tag in cases:
        p0 = torch.randn(64) * 0.1
        g0 = torch.randn(64) * 0.01
        p_ref, _, _ = ref_adamw_step(
            p0.clone(),
            g0,
            torch.zeros(64),
            torch.zeros(64),
            1,
            LR,
            B1,
            B2,
            EPS,
            expected_wd,
        )

        param = torch.nn.Parameter(p0.clone())
        opt = ConvMuon(
            [(name, param)],
            lr=LR_MAIN,
            weight_decay=WD_MUON,
            adam_weight_decay=WD_ADAM,
            adam_lr_ratio=RATIO,
            wd_lr_exponent=None,
            global_clipnorm=None,
            exclude_layers=[r".*policy_head.*"],
        )
        param.grad = g0.clone()
        opt.step()
        ok = check(
            f"{tag:<20} wd={expected_wd:.4f}", param.data, p_ref, atol=1e-5, rtol=1e-5
        )
        all_ok &= ok
    return all_ok


# ------------------------------------------------------------------ #
# Test 4: LR-ratio WD scaling                                         #
# ------------------------------------------------------------------ #


def test_wd_lr_scaling():
    print("\nTest 4: LR-ratio WD scaling (_lr_scale)")
    cases = [
        (1e-3, 1e-3, 0.7, 1.0),  # lr == lr_max → scale=1
        (0.5e-3, 1e-3, 0.7, 0.5**0.7),  # lr == lr_max/2
        (2e-3, 1e-3, 0.7, 1.0),  # lr > lr_max → clamped to 1
    ]
    opt = ConvMuon(
        [("p", torch.nn.Parameter(torch.zeros(1)))],
        lr=1e-3,
        wd_lr_exponent=0.7,
        wd_lr_max=1e-3,
        global_clipnorm=None,
    )
    all_ok = True
    for lr, lr_max, exp, expected in cases:
        opt._wd_lr_exponent = exp
        opt._wd_lr_max = lr_max
        actual = opt._lr_scale(lr)
        ok = abs(actual - expected) < 1e-6
        print(
            f"  [{'PASS' if ok else 'FAIL'}] lr={lr:.1e} lr_max={lr_max:.1e} exp={exp}  "
            f"actual={actual:.6f}  expected={expected:.6f}"
        )
        all_ok &= ok
    return all_ok


# ------------------------------------------------------------------ #
# Test 5: Per-category WD base values                                 #
# ------------------------------------------------------------------ #


def test_wd_categories():
    print("\nTest 5: Per-category WD base at construction")
    WD_MUON, WD_ADAM = 0.1, 0.004
    cases = [
        ("trunk.conv.kernel", (3, 3, 32, 64), WD_MUON * 1.0, "muon body"),
        (
            "trunk.attn.transformer_attention.query.kernel",
            (64, 64),
            WD_MUON * 0.5,
            "qkvo",
        ),
        ("trunk.bn.gamma", (64,), WD_MUON * 0.1, "gamma"),
        ("trunk.bn.beta", (64,), WD_MUON * 1e-3, "beta"),
        ("trunk.layer.bias", (64,), WD_MUON * 1e-2, "body_bias"),
        ("policy_head.out.bias", (64,), WD_MUON * 1e-2, "head_bias"),
        ("policy_head.out.kernel", (64, 64), WD_ADAM, "excluded head"),
    ]
    named = [
        (name, torch.nn.Parameter(torch.randn(*shape))) for name, shape, _, _ in cases
    ]
    opt = ConvMuon(
        named,
        lr=1e-3,
        weight_decay=WD_MUON,
        adam_weight_decay=WD_ADAM,
        exclude_layers=[r".*policy_head.*"],
        global_clipnorm=None,
    )
    all_ok = True
    for (name, param), (_, _, expected, tag) in zip(named, cases):
        actual = opt._wd_base[id(param)]
        ok = abs(actual - expected) < 1e-9
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {tag:<20}  actual={actual:.5f}  expected={expected:.5f}"
        )
        all_ok &= ok
    return all_ok


# ------------------------------------------------------------------ #
# Test 6: Multi-step stability (NaN/Inf check)                        #
# ------------------------------------------------------------------ #


def test_multistep():
    print("\nTest 6: Multi-step stability (20 steps)")
    named = [
        ("trunk.conv.kernel", torch.nn.Parameter(torch.randn(3, 3, 32, 64) * 0.1)),
        ("trunk.dense.kernel", torch.nn.Parameter(torch.randn(64, 128) * 0.1)),
        ("trunk.bn.gamma", torch.nn.Parameter(torch.randn(64) * 0.1)),
        ("policy_head.out.kernel", torch.nn.Parameter(torch.randn(128, 362) * 0.1)),
    ]
    opt = ConvMuon(
        named,
        lr=1e-3,
        weight_decay=0.1,
        adam_weight_decay=0.004,
        exclude_layers=[r".*policy_head.*"],
        rms_rate=0.2,
        wd_lr_exponent=0.7,
        wd_lr_max=1e-3,
        global_clipnorm=20.0,
    )
    for _ in range(20):
        for _, p in named:
            p.grad = torch.randn_like(p) * 0.01
        opt.step()
    all_ok = True
    for name, p in named:
        ok = not (torch.isnan(p.data).any() or torch.isinf(p.data).any())
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= ok
    return all_ok


if __name__ == "__main__":
    results = [
        test_ns_singular_values(),
        test_muon_step(),
        test_adamw_step(),
        test_wd_lr_scaling(),
        test_wd_categories(),
        test_multistep(),
    ]
    results = [r for r in results if r is not None]
    print(f"\n{'All targeted tests passed.' if all(results) else 'Some tests FAILED.'}")
