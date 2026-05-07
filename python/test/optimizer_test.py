"""Tests for the ConvMuon optimizer (python/optimizer.py)."""

import os

# Force tensorflow backend so keras.Variable etc. are TF-backed.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import unittest

import keras
import numpy as np
from keras.src import ops

from backend_tf.optimizer import ConvMuon


def _make_var(value, path):
    """Build a keras.Variable and override its path so _wd_category /
    _should_use_adamw can be exercised. `name` cannot contain '/', so we
    set _path directly."""
    v = keras.Variable(np.asarray(value, dtype=np.float32), name="v")
    v._path = path
    return v


def _to_np(x):
    return np.asarray(keras.ops.convert_to_numpy(x), dtype=np.float64)


class NewtonSchulzTest(unittest.TestCase):
    """zeropower_via_newtonschulz5 should orthogonalize: SVs → 1."""

    def setUp(self):
        self.opt = ConvMuon(
            learning_rate=1e-3,
            weight_decay=0.0,
            adam_weight_decay=0.0,
            ns_steps=5,
            exclude_layers=[],
        )

    def _check_svs_near_one(self, X_np, atol=0.4):
        # NS5 with the Keller-Jordan coefficients (3.4445, -4.7750, 2.0315)
        # is designed to push singular values toward 1 but not exactly to 1
        # — the documented range for 5 steps is roughly [0.6, 1.3].
        Y = self.opt.zeropower_via_newtonschulz5(keras.ops.convert_to_tensor(X_np), 5)
        Y_np = _to_np(Y)
        sv = np.linalg.svd(Y_np, compute_uv=False)
        self.assertTrue(
            np.all(np.abs(sv - 1.0) < atol),
            f"SVs not near 1: min={sv.min():.3f} max={sv.max():.3f}",
        )

    def test_wide_matrix(self):
        rng = np.random.default_rng(0)
        self._check_svs_near_one(rng.standard_normal((16, 64)).astype(np.float32))

    def test_tall_matrix(self):
        rng = np.random.default_rng(1)
        self._check_svs_near_one(rng.standard_normal((64, 16)).astype(np.float32))

    def test_square_matrix(self):
        rng = np.random.default_rng(2)
        self._check_svs_near_one(rng.standard_normal((32, 32)).astype(np.float32))

    def test_rectangular_4d_flattens_to_2d(self):
        # 3x3 conv with 32 in, 64 out — flattens to (9*32, 64) = (288, 64).
        rng = np.random.default_rng(3)
        G = rng.standard_normal((3, 3, 32, 64)).astype(np.float32)
        G_2d = G.reshape(-1, 64)
        self._check_svs_near_one(G_2d)


class RoutingTest(unittest.TestCase):
    """_should_use_adamw should route by shape, layer name, and exclude rules."""

    def setUp(self):
        self.opt = ConvMuon(
            learning_rate=1e-3,
            weight_decay=0.0,
            adam_weight_decay=0.0,
            exclude_layers=[r".*policy_head/.*", r".*value_head/.*"],
        )

    def _route(self, shape, path):
        v = _make_var(np.zeros(shape, dtype=np.float32), path)
        return self.opt._should_use_adamw(v)

    def test_1d_bias_routes_to_adamw(self):
        self.assertTrue(self._route((128,), "body/bias"))

    def test_small_2d_routes_to_adamw(self):
        # out_dim ≤ 4 should fall back to AdamW
        self.assertTrue(self._route((128, 4), "body/kernel"))
        self.assertTrue(self._route((4, 128), "body/kernel"))

    def test_large_2d_routes_to_muon(self):
        self.assertFalse(self._route((128, 64), "body/kernel"))

    def test_4d_conv_routes_to_muon(self):
        # Effective 2D = (3*3*32, 64) = (288, 64) — both > 4 → Muon.
        self.assertFalse(self._route((3, 3, 32, 64), "body/conv/kernel"))

    def test_embedding_routes_to_adamw(self):
        self.assertTrue(self._route((1024, 64), "embedding/embeddings"))

    def test_excluded_layer_routes_to_adamw(self):
        self.assertTrue(self._route((128, 64), "policy_head/dense/kernel"))
        self.assertTrue(self._route((128, 64), "value_head/dense/kernel"))

    def test_build_creates_velocities_only_for_adamw_vars(self):
        # Velocity slot is only allocated for AdamW path vars.
        muon_var = _make_var(np.zeros((32, 64), dtype=np.float32), "body/kernel")
        adamw_var = _make_var(np.zeros((64,), dtype=np.float32), "body/bias")
        excl_var = _make_var(
            np.zeros((32, 64), dtype=np.float32), "policy_head/dense/kernel"
        )
        opt = ConvMuon(
            learning_rate=1e-3,
            weight_decay=0.0,
            adam_weight_decay=0.0,
            exclude_layers=[r".*policy_head/.*"],
        )
        opt.build([muon_var, adamw_var, excl_var])
        self.assertIsNone(opt.adam_velocities[opt._get_variable_index(muon_var)])
        self.assertIsNotNone(opt.adam_velocities[opt._get_variable_index(adamw_var)])
        self.assertIsNotNone(opt.adam_velocities[opt._get_variable_index(excl_var)])


class WdCategoryTest(unittest.TestCase):
    """_wd_category should classify gamma/beta/biases/qkvo by path."""

    def setUp(self):
        self.opt = ConvMuon(
            learning_rate=1e-3,
            weight_decay=0.0,
            adam_weight_decay=0.0,
            exclude_layers=[],
        )

    def _cat(self, path):
        v = _make_var(np.zeros((4,), dtype=np.float32), path)
        return self.opt._wd_category(v)

    def test_gamma_paths(self):
        self.assertEqual(self._cat("body/bn/gamma"), "gamma")
        self.assertEqual(self._cat("body/rmsnorm/scale"), "gamma")

    def test_beta_paths(self):
        self.assertEqual(self._cat("body/bn/beta"), "beta")

    def test_body_bias(self):
        self.assertEqual(self._cat("body/dense/bias"), "body_bias")

    def test_head_bias(self):
        self.assertEqual(self._cat("policy_head/dense/bias"), "head_bias")
        self.assertEqual(self._cat("value_head/dense/bias"), "head_bias")

    def test_qkvo(self):
        self.assertEqual(self._cat("body/transformer_attention/query/kernel"), "qkvo")
        self.assertEqual(self._cat("body/transformer_attention/output/kernel"), "qkvo")

    def test_unclassified(self):
        self.assertIsNone(self._cat("body/conv/kernel"))


class WeightDecayTest(unittest.TestCase):
    """_apply_weight_decay should apply var ← var * (1 - lr * wd * lr_scale)."""

    LR = 1e-3

    def _build_opt(self, **kwargs):
        defaults = dict(
            learning_rate=self.LR,
            weight_decay=0.02,
            adam_weight_decay=0.01,
            exclude_layers=[],
        )
        defaults.update(kwargs)
        opt = ConvMuon(**defaults)
        return opt

    def _apply_and_get_factor(self, opt, var):
        """Run WD on a var initialized to ones; return scalar factor (1 - lr*wd)."""
        opt.build([var])
        opt._apply_weight_decay([var])
        return float(_to_np(var.value).mean())

    def test_general_muon_uses_weight_decay(self):
        opt = self._build_opt()
        v = _make_var(np.ones((32, 64), dtype=np.float32), "body/kernel")
        factor = self._apply_and_get_factor(opt, v)
        self.assertAlmostEqual(factor, 1.0 - self.LR * 0.02, places=6)

    def test_general_adamw_uses_adam_weight_decay(self):
        opt = self._build_opt()
        v = _make_var(np.ones((128,), dtype=np.float32), "body/dense/kernel_1d")
        # 1D → adamw branch. But path doesn't end in any category suffix.
        v._path = "body/dense/foo"  # not a recognized category
        factor = self._apply_and_get_factor(opt, v)
        self.assertAlmostEqual(factor, 1.0 - self.LR * 0.01, places=6)

    def test_per_category_gamma(self):
        opt = self._build_opt()
        v = _make_var(np.ones((32,), dtype=np.float32), "body/bn/gamma")
        factor = self._apply_and_get_factor(opt, v)
        # gamma factor = 0.1 → wd_value = 0.02 * 0.1 = 0.002
        self.assertAlmostEqual(factor, 1.0 - self.LR * 0.02 * 0.1, places=6)

    def test_per_category_beta(self):
        opt = self._build_opt()
        v = _make_var(np.ones((32,), dtype=np.float32), "body/bn/beta")
        factor = self._apply_and_get_factor(opt, v)
        # beta factor = 1e-3 → wd_value = 0.02 * 1e-3 = 2e-5
        self.assertAlmostEqual(factor, 1.0 - self.LR * 0.02 * 1e-3, places=7)

    def test_per_category_body_bias(self):
        opt = self._build_opt()
        v = _make_var(np.ones((32,), dtype=np.float32), "body/dense/bias")
        factor = self._apply_and_get_factor(opt, v)
        self.assertAlmostEqual(factor, 1.0 - self.LR * 0.02 * 1e-2, places=7)

    def test_per_category_head_bias(self):
        opt = self._build_opt()
        v = _make_var(np.ones((32,), dtype=np.float32), "policy_head/dense/bias")
        factor = self._apply_and_get_factor(opt, v)
        self.assertAlmostEqual(factor, 1.0 - self.LR * 0.02 * 1e-2, places=7)

    def test_per_category_qkvo(self):
        opt = self._build_opt()
        v = _make_var(
            np.ones((64, 64), dtype=np.float32),
            "body/transformer_attention/query/kernel",
        )
        factor = self._apply_and_get_factor(opt, v)
        # qkvo factor = 0.5 → wd_value = 0.02 * 0.5 = 0.01
        self.assertAlmostEqual(factor, 1.0 - self.LR * 0.02 * 0.5, places=6)

    def test_wd_lr_exponent_scaling(self):
        # At lr = max_lr/2, scale = (0.5)^0.7 ≈ 0.6155
        opt = self._build_opt(learning_rate=5e-4, wd_lr_exponent=0.7, wd_lr_max=1e-3)
        v = _make_var(np.ones((32, 64), dtype=np.float32), "body/kernel")
        opt.build([v])
        opt._apply_weight_decay([v])
        factor = float(_to_np(v.value).mean())
        expected_scale = 0.5**0.7
        expected_factor = 1.0 - 5e-4 * 0.02 * expected_scale
        self.assertAlmostEqual(factor, expected_factor, places=6)

    def test_wd_lr_exponent_clamped_above_max(self):
        # lr > max_lr should clamp to lr_ratio = 1 (factor = 1.0).
        opt = self._build_opt(learning_rate=2e-3, wd_lr_exponent=0.7, wd_lr_max=1e-3)
        v = _make_var(np.ones((32, 64), dtype=np.float32), "body/kernel")
        opt.build([v])
        opt._apply_weight_decay([v])
        factor = float(_to_np(v.value).mean())
        # No additional scaling since lr_ratio is clamped to 1.
        self.assertAlmostEqual(factor, 1.0 - 2e-3 * 0.02 * 1.0, places=6)


class MuonUpdateStepTest(unittest.TestCase):
    """_muon_update_step: EMA momentum + symmetric Nesterov + lr_adjust."""

    BETA = 0.95
    RMS = 0.2
    LR = 1e-3

    def _build_opt(self, nesterov=True, rms_rate=RMS, ns_steps=5):
        return ConvMuon(
            learning_rate=self.LR,
            weight_decay=0.0,
            adam_weight_decay=0.0,
            momentum=self.BETA,
            nesterov=nesterov,
            rms_rate=rms_rate,
            ns_steps=ns_steps,
            exclude_layers=[],
        )

    def _expected_step(self, opt, var0, m0, g, ndim_flat=None):
        """Reproduce _muon_update_step using ops directly, returning
        (m_new, var_new) numpy."""
        m_new = m0 * self.BETA + g * (1 - self.BETA)
        if opt.nesterov:
            g_eff = g * (1 - self.BETA) + m_new * self.BETA
        else:
            g_eff = m_new
        if ndim_flat is not None:
            g_2d = g_eff.reshape(-1, ndim_flat)
        else:
            g_2d = g_eff
        ns = opt.zeropower_via_newtonschulz5(
            keras.ops.convert_to_tensor(g_2d.astype(np.float32)), opt.ns_steps
        )
        ns_np = _to_np(ns)
        if opt.rms_rate is not None:
            scale = (max(ns_np.shape[-2], ns_np.shape[-1]) ** 0.5) * opt.rms_rate
        else:
            scale = 1.0
        update_2d = self.LR * ns_np * scale
        if ndim_flat is not None:
            update = update_2d.reshape(g.shape)
        else:
            update = update_2d
        return m_new, var0 - update

    def test_first_step_2d(self):
        opt = self._build_opt()
        rng = np.random.default_rng(0)
        var0 = rng.standard_normal((32, 64)).astype(np.float32) * 0.1
        g = rng.standard_normal((32, 64)).astype(np.float32) * 0.01
        v = _make_var(var0, "body/kernel")
        opt.build([v])
        m_var = opt.momentums[opt._get_variable_index(v)]
        m0 = np.zeros_like(var0)
        m_exp, var_exp = self._expected_step(opt, var0, m0, g)
        opt._muon_update_step(
            keras.ops.convert_to_tensor(g), v, opt.learning_rate, m_var
        )
        np.testing.assert_allclose(_to_np(m_var.value), m_exp, atol=1e-7)
        np.testing.assert_allclose(_to_np(v.value), var_exp, atol=1e-5)

    def test_second_step_with_existing_buf(self):
        opt = self._build_opt()
        rng = np.random.default_rng(1)
        var0 = rng.standard_normal((32, 64)).astype(np.float32) * 0.1
        m0 = rng.standard_normal((32, 64)).astype(np.float32) * 0.05
        g = rng.standard_normal((32, 64)).astype(np.float32) * 0.01
        v = _make_var(var0, "body/kernel")
        opt.build([v])
        m_var = opt.momentums[opt._get_variable_index(v)]
        m_var.assign(m0)
        m_exp, var_exp = self._expected_step(opt, var0, m0, g)
        opt._muon_update_step(
            keras.ops.convert_to_tensor(g), v, opt.learning_rate, m_var
        )
        np.testing.assert_allclose(_to_np(m_var.value), m_exp, atol=1e-7)
        np.testing.assert_allclose(_to_np(v.value), var_exp, atol=1e-5)

    def test_no_nesterov(self):
        opt = self._build_opt(nesterov=False)
        rng = np.random.default_rng(2)
        var0 = rng.standard_normal((32, 64)).astype(np.float32) * 0.1
        m0 = rng.standard_normal((32, 64)).astype(np.float32) * 0.05
        g = rng.standard_normal((32, 64)).astype(np.float32) * 0.01
        v = _make_var(var0, "body/kernel")
        opt.build([v])
        m_var = opt.momentums[opt._get_variable_index(v)]
        m_var.assign(m0)
        m_exp, var_exp = self._expected_step(opt, var0, m0, g)
        opt._muon_update_step(
            keras.ops.convert_to_tensor(g), v, opt.learning_rate, m_var
        )
        np.testing.assert_allclose(_to_np(m_var.value), m_exp, atol=1e-7)
        np.testing.assert_allclose(_to_np(v.value), var_exp, atol=1e-5)

    def test_4d_conv_flatten_roundtrip(self):
        opt = self._build_opt()
        rng = np.random.default_rng(3)
        var0 = rng.standard_normal((3, 3, 32, 64)).astype(np.float32) * 0.1
        m0 = rng.standard_normal((3, 3, 32, 64)).astype(np.float32) * 0.05
        g = rng.standard_normal((3, 3, 32, 64)).astype(np.float32) * 0.01
        v = _make_var(var0, "body/conv/kernel")
        opt.build([v])
        m_var = opt.momentums[opt._get_variable_index(v)]
        m_var.assign(m0)
        m_exp, var_exp = self._expected_step(opt, var0, m0, g, ndim_flat=64)
        opt._muon_update_step(
            keras.ops.convert_to_tensor(g), v, opt.learning_rate, m_var
        )
        np.testing.assert_allclose(_to_np(m_var.value), m_exp, atol=1e-7)
        np.testing.assert_allclose(_to_np(v.value), var_exp, atol=1e-5)

    def test_rms_rate_disabled(self):
        opt = self._build_opt(rms_rate=None)
        rng = np.random.default_rng(4)
        var0 = rng.standard_normal((32, 64)).astype(np.float32) * 0.1
        g = rng.standard_normal((32, 64)).astype(np.float32) * 0.01
        v = _make_var(var0, "body/kernel")
        opt.build([v])
        m_var = opt.momentums[opt._get_variable_index(v)]
        m0 = np.zeros_like(var0)
        m_exp, var_exp = self._expected_step(opt, var0, m0, g)
        opt._muon_update_step(
            keras.ops.convert_to_tensor(g), v, opt.learning_rate, m_var
        )
        np.testing.assert_allclose(_to_np(m_var.value), m_exp, atol=1e-7)
        np.testing.assert_allclose(_to_np(v.value), var_exp, atol=1e-5)


class AdamWUpdateStepTest(unittest.TestCase):
    """_adamw_update_step: standard Adam + bias correction."""

    LR = 1e-3
    B1 = 0.9
    B2 = 0.999
    EPS = 1e-7

    def test_first_step(self):
        opt = ConvMuon(
            learning_rate=self.LR,
            weight_decay=0.0,
            adam_weight_decay=0.0,
            adam_beta_1=self.B1,
            adam_beta_2=self.B2,
            epsilon=self.EPS,
            adam_lr_ratio=1.0,
            exclude_layers=[],
        )
        rng = np.random.default_rng(7)
        var0 = rng.standard_normal((128,)).astype(np.float32) * 0.1
        g = rng.standard_normal((128,)).astype(np.float32) * 0.01
        v = _make_var(var0, "body/dense/foo_1d")
        opt.build([v])
        idx = opt._get_variable_index(v)
        m_var = opt.momentums[idx]
        v_var = opt.adam_velocities[idx]
        # Bump iterations so local_step = 1 (matches keras Muon's local_step
        # = self.iterations + 1).
        opt._adamw_update_step(keras.ops.convert_to_tensor(g), v, self.LR, m_var, v_var)
        # Reference: m1 = (1-β1)*g, v1 = (1-β2)*g²
        m1 = (1 - self.B1) * g
        v1 = (1 - self.B2) * g * g
        # local_step = iterations+1 = 1 (iterations bumped after this call)
        b1p = self.B1**1
        b2p = self.B2**1
        alpha = self.LR * np.sqrt(1 - b2p) / (1 - b1p)
        var_exp = var0 - alpha * m1 / (np.sqrt(v1) + self.EPS)
        np.testing.assert_allclose(_to_np(m_var.value), m1, atol=1e-8)
        np.testing.assert_allclose(_to_np(v_var.value), v1, atol=1e-10)
        np.testing.assert_allclose(_to_np(v.value), var_exp, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
