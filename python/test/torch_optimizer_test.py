"""Parity tests: keras ConvMuon vs torch-native ConvMuon.

Both optimizers are run on a tiny 2-layer model for N steps with identical
initial weights and identical (deterministic) batches. Loss curves must stay
within atol=5e-3 across all steps.

Run:
    KERAS_BACKEND=tensorflow PYTHONPATH=python:python/test \
        python python/test/torch_optimizer_test.py -v
"""

import os, sys, unittest

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "python")
sys.path.insert(0, "python/test")

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tiny model that exercises both Muon (conv, dense) and AdamW (bn, bias) paths
# ---------------------------------------------------------------------------


class _TinyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8, momentum=0.01, eps=1e-3)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv(x)))
        return self.fc(h.mean(dim=(2, 3)))


# ---------------------------------------------------------------------------
# Individual unit tests (no keras comparison) — fast and always runnable
# ---------------------------------------------------------------------------


class ConvMuonTorchUnitTest(unittest.TestCase):

    def _make_opt(self, model):
        from backend_torch.optimizer import ConvMuon, build_convmuon_param_groups

        groups, wd_factors = build_convmuon_param_groups(
            model,
            lr=1e-3,
            adam_lr_ratio=1.0,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
        )
        return ConvMuon(
            groups,
            wd_factors=wd_factors,
            weight_decay=0.02,
            adam_weight_decay=0.02,
            adam_lr_ratio=1.0,
            rms_rate=0.2,
        )

    def test_step_runs(self):
        model = _TinyTorch()
        opt = self._make_opt(model)
        x = torch.randn(4, 4, 5, 5)
        y = torch.randint(0, 2, (4,))
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        opt.step()

    def test_loss_decreases(self):
        """Loss should trend downward over 10 steps with a fixed batch."""
        torch.manual_seed(0)
        model = _TinyTorch()
        opt = self._make_opt(model)
        x = torch.randn(8, 4, 5, 5)
        y = torch.randint(0, 2, (8,))
        losses = []
        for _ in range(10):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(
            losses[-1],
            losses[0],
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}",
        )

    def test_lr_setter(self):
        from backend_torch.optimizer import ConvMuon, build_convmuon_param_groups

        model = _TinyTorch()
        groups, wdf = build_convmuon_param_groups(model, lr=1e-3)
        opt = ConvMuon(groups, wd_factors=wdf)
        opt.learning_rate = 2e-3
        self.assertAlmostEqual(opt.learning_rate, 2e-3)

    def test_muon_and_adamw_groups(self):
        """Conv weights go to Muon group; BN stats go to AdamW group."""
        from backend_torch.optimizer import ConvMuon, build_convmuon_param_groups

        model = _TinyTorch()
        groups, wdf = build_convmuon_param_groups(model, lr=1e-3)
        opt = ConvMuon(groups, wd_factors=wdf)
        self.assertEqual(len(opt.param_groups), 2)
        muon_group = next(g for g in opt.param_groups if g["group"] == "muon")
        adamw_group = next(g for g in opt.param_groups if g["group"] == "adamw")
        # conv.weight (4D → Muon), fc.weight (2D → Muon)
        self.assertGreater(len(muon_group["params"]), 0)
        # bn.weight, bn.bias, fc.bias → AdamW
        self.assertGreater(len(adamw_group["params"]), 0)

    def test_apply_interface(self):
        """optimizer.apply() without args should work like step()."""
        from backend_torch.optimizer import ConvMuon, build_convmuon_param_groups

        model = _TinyTorch()
        groups, wdf = build_convmuon_param_groups(model, lr=1e-3)
        opt = ConvMuon(groups, wd_factors=wdf)
        x = torch.randn(4, 4, 5, 5)
        y = torch.randint(0, 2, (4,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.apply()  # keras-compatible API

    # NS5 orthogonality on tall/wide/extreme-aspect matrices is covered by
    # `torch_optimizer_rect_test.py::NS5RectangularShapesTest`. The singleton
    # 32×32 check that used to live here was a strict subset.


# ---------------------------------------------------------------------------
# Parity test: keras ConvMuon vs torch ConvMuon on identical tiny model
# ---------------------------------------------------------------------------


class ConvMuonKerasTorchParityTest(unittest.TestCase):
    """Compare PARAM values after N optimizer steps.

    NOTE: The earlier loss-trajectory comparison drifts ~0.0017/step due to
    XLA-vs-eager fp32 fusion differences in keras `train_on_batch`'s reported
    loss (it differs from re-running the forward in eager mode by exactly
    that amount). Param parity is the meaningful metric: if the optimizer
    math agrees, params after N steps should match within fp32 noise.
    """

    N_STEPS = 10
    BATCH = 8
    ATOL = 1e-4
    RTOL = 1e-2

    def _shared_weights(self):
        """Return a dict of numpy arrays that both models will start from."""
        np.random.seed(7)
        return {
            "conv": np.random.randn(3, 3, 4, 8).astype(np.float32),  # HWIO keras
            "bn_gamma": np.ones(8, dtype=np.float32),
            "bn_beta": np.zeros(8, dtype=np.float32),
            "fc_kernel": np.random.randn(8, 2).astype(np.float32),  # (in,out) keras
            "fc_bias": np.zeros(2, dtype=np.float32),
        }

    def _keras_params(self, w):
        import keras
        import tensorflow as tf

        # Force CPU-only TF so arithmetic matches torch CPU fp32.
        tf.config.set_visible_devices([], "GPU")
        keras.mixed_precision.set_global_policy("float32")

        # Build a tiny keras model
        inp = keras.Input(shape=(5, 5, 4))
        h = keras.layers.Conv2D(
            8, 3, padding="same", use_bias=False, kernel_initializer="zeros"
        )(inp)
        h = keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(h)
        h = keras.layers.Activation("relu")(h)
        h = keras.layers.GlobalAveragePooling2D()(h)
        out = keras.layers.Dense(2)(h)
        km = keras.Model(inp, out)
        # Build
        km(np.zeros((1, 5, 5, 4), dtype=np.float32))

        # Set weights (keras NHWC convention)
        # Locate layers by type to be robust to InputLayer presence/absence.
        conv_l = next(l for l in km.layers if isinstance(l, keras.layers.Conv2D))
        bn_l = next(
            l for l in km.layers if isinstance(l, keras.layers.BatchNormalization)
        )
        dense_l = next(l for l in km.layers if isinstance(l, keras.layers.Dense))
        conv_l.kernel.assign(w["conv"])
        bn_l.gamma.assign(w["bn_gamma"])
        bn_l.beta.assign(w["bn_beta"])
        bn_l.moving_mean.assign(np.zeros(8, dtype=np.float32))
        bn_l.moving_variance.assign(np.ones(8, dtype=np.float32))
        dense_l.kernel.assign(w["fc_kernel"])
        dense_l.bias.assign(w["fc_bias"])

        from backend_tf.optimizer import ConvMuon as KConvMuon

        opt = KConvMuon(
            learning_rate=1e-3,
            weight_decay=0.02,
            adam_weight_decay=0.02,
            adam_lr_ratio=1.0,
            momentum=0.95,
            nesterov=True,
            rms_rate=0.2,
            ns_steps=5,
        )
        km.compile(
            optimizer=opt,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        np.random.seed(42)
        x = np.random.randn(self.BATCH, 5, 5, 4).astype(np.float32)
        y = np.arange(self.BATCH) % 2

        for _ in range(self.N_STEPS):
            km.train_on_batch(x, y)

        return {
            "conv": conv_l.kernel.numpy(),
            "bn_gamma": bn_l.gamma.numpy(),
            "bn_beta": bn_l.beta.numpy(),
            "fc_kernel": dense_l.kernel.numpy(),
            "fc_bias": dense_l.bias.numpy(),
        }

    def _torch_params(self, w):
        import torch
        from backend_torch.optimizer import (
            ConvMuon as TConvMuon,
            build_convmuon_param_groups,
        )

        model = _TinyTorch()
        # Set matching weights (torch NCHW convention)
        with torch.no_grad():
            # Conv: keras (H,W,I,O) → torch (O,I,H,W)
            model.conv.weight.copy_(torch.tensor(w["conv"]).permute(3, 2, 0, 1))
            model.bn.weight.copy_(torch.tensor(w["bn_gamma"]))
            model.bn.bias.copy_(torch.tensor(w["bn_beta"]))
            model.bn.running_mean.zero_()
            model.bn.running_var.fill_(1.0)
            # Dense: keras (I,O) → torch (O,I)
            model.fc.weight.copy_(torch.tensor(w["fc_kernel"]).T)
            model.fc.bias.copy_(torch.tensor(w["fc_bias"]))

        groups, wdf = build_convmuon_param_groups(
            model,
            lr=1e-3,
            adam_lr_ratio=1.0,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
        )
        opt = TConvMuon(
            groups,
            wd_factors=wdf,
            weight_decay=0.02,
            adam_weight_decay=0.02,
            adam_lr_ratio=1.0,
            rms_rate=0.2,
        )

        np.random.seed(42)
        x_np = np.random.randn(self.BATCH, 5, 5, 4).astype(np.float32)
        y_np = np.arange(self.BATCH) % 2
        x = torch.tensor(x_np).permute(0, 3, 1, 2)  # NHWC → NCHW
        y = torch.tensor(y_np, dtype=torch.long)

        for _ in range(self.N_STEPS):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step()

        return {
            # Permute torch (Cout,Cin,H,W) → keras (H,W,Cin,Cout) for direct compare.
            "conv": model.conv.weight.detach().permute(2, 3, 1, 0).numpy(),
            "bn_gamma": model.bn.weight.detach().numpy(),
            "bn_beta": model.bn.bias.detach().numpy(),
            # Linear (Cout,Cin) → keras (Cin,Cout) via .T.
            "fc_kernel": model.fc.weight.detach().T.numpy(),
            "fc_bias": model.fc.bias.detach().numpy(),
        }

    def test_param_parity(self):
        """After N steps, param values should match within atol=1e-4 / rtol=1e-2."""
        w = self._shared_weights()
        keras_p = self._keras_params(w)
        torch_p = self._torch_params(w)
        for k in keras_p:
            kp, tp = keras_p[k], torch_p[k]
            self.assertEqual(kp.shape, tp.shape, f"{k}: shape mismatch")
            abs_diff = float(np.abs(kp - tp).max())
            rel_diff = abs_diff / (float(np.abs(kp).max()) + 1e-9)
            self.assertTrue(
                abs_diff <= self.ATOL or rel_diff <= self.RTOL,
                f"{k}: abs={abs_diff:.4e}, rel={rel_diff:.4e} "
                f"(atol={self.ATOL}, rtol={self.RTOL})",
            )


if __name__ == "__main__":
    unittest.main()
