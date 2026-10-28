"""Tests for the `proj_conv_size` option on `transformer_btl` blocks.

New models default to 1x1 down/up projections and record the kernel size
explicitly in their config. Checkpoints that predate the key (always 3x3) must
still load, in both backends.

Run:
    PYTHONPATH=python:python/test python python/test/btl_proj_conv_size_test.py -v
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "python")
sys.path.insert(0, "python/test")

import torch  # noqa: E402

from model_config import ModelConfig, with_btl_proj_conv_size  # noqa: E402

BOARD_LEN = 19
NUM_PLANES = 15
NUM_FEATURES = 8
REAL_CKPT = os.path.expanduser("~/p3achygo-data/v4-models/main/model_0400.pt")

_BTL = ["transformer_btl", {"output_dim": 32, "embed_dim": 16, "num_heads": 2}]
ARCH = {"trunk": [copy.deepcopy(_BTL), copy.deepcopy(_BTL)]}


def _torch_model(arch):
    from backend_torch.model_utils import new_model

    model = new_model(
        ModelConfig.from_generic_arch(arch), BOARD_LEN, NUM_PLANES, NUM_FEATURES, "t"
    )
    return model.eval()


def _torch_inputs():
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(2, BOARD_LEN, BOARD_LEN, NUM_PLANES, generator=g),
        torch.randn(2, NUM_FEATURES, generator=g),
    )


def _btl_kernels(model):
    from backend_torch.model_transformer import TransformerBottleneckBlock

    return [
        (m.conv_down.conv.kernel_size[0], m.conv_up.conv.kernel_size[0])
        for m in model.modules()
        if isinstance(m, TransformerBottleneckBlock)
    ]


class HelperTest(unittest.TestCase):
    def test_fills_only_missing_btl_entries(self):
        arch = {
            "trunk": [
                ["transformer_btl", {"embed_dim": 16, "num_heads": 2}],
                ["transformer_btl", {"embed_dim": 16, "proj_conv_size": 3}],
                ["nbt", {"output_channels": 32}],
            ],
            "c_val": 64,
        }
        out = with_btl_proj_conv_size(arch, 1)
        self.assertEqual(out["trunk"][0][1]["proj_conv_size"], 1)
        self.assertEqual(out["trunk"][1][1]["proj_conv_size"], 3)
        self.assertNotIn("proj_conv_size", out["trunk"][2][1])
        self.assertEqual(out["c_val"], 64)
        # Input is not mutated.
        self.assertNotIn("proj_conv_size", arch["trunk"][0][1])

    def test_none_passes_through(self):
        self.assertIsNone(with_btl_proj_conv_size(None, 1))


class TorchTest(unittest.TestCase):
    def test_new_model_defaults_to_1x1_and_records_it(self):
        arch = copy.deepcopy(ARCH)
        model = _torch_model(arch)
        self.assertEqual(_btl_kernels(model), [(1, 1), (1, 1)])
        for _, cfg in model.config_dict()["generic_arch"]["trunk"]:
            self.assertEqual(cfg["proj_conv_size"], 1)
        self.assertEqual(arch, ARCH)  # caller's config not mutated

    def test_explicit_3x3(self):
        arch = with_btl_proj_conv_size(ARCH, 3)
        self.assertEqual(_btl_kernels(_torch_model(arch)), [(3, 3), (3, 3)])

    def test_save_load_round_trip_keeps_1x1(self):
        from backend_torch.model_utils import load_model, save_model

        model = _torch_model(ARCH)
        x, g = _torch_inputs()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            save_model(model, path)
            loaded = load_model(path).eval()
        self.assertEqual(_btl_kernels(loaded), [(1, 1), (1, 1)])
        with torch.no_grad():
            torch.testing.assert_close(loaded(x, g)[0], model(x, g)[0])

    def test_legacy_checkpoint_without_key_loads_as_3x3(self):
        from backend_torch.model_utils import load_model, save_model

        model = _torch_model(with_btl_proj_conv_size(ARCH, 3))
        x, g = _torch_inputs()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            save_model(model, path)
            # Rewrite the saved config as a pre-`proj_conv_size` checkpoint.
            ckpt = torch.load(path, weights_only=False)
            for _, cfg in ckpt["config"]["generic_arch"]["trunk"]:
                del cfg["proj_conv_size"]
            torch.save(ckpt, path)
            loaded = load_model(path).eval()
        self.assertEqual(_btl_kernels(loaded), [(3, 3), (3, 3)])
        with torch.no_grad():
            torch.testing.assert_close(loaded(x, g)[0], model(x, g)[0])

    @unittest.skipUnless(os.path.exists(REAL_CKPT), "production checkpoint absent")
    def test_production_checkpoint_loads(self):
        from backend_torch.model_utils import load_model

        model = load_model(REAL_CKPT)
        self.assertEqual(_btl_kernels(model), 8 * [(3, 3)])


class KerasTest(unittest.TestCase):
    def setUp(self):
        try:
            from backend_tf.model import P3achyGoModel  # noqa: F401
        except Exception as e:  # keras/TF not installed in this env
            self.skipTest(f"keras backend unavailable: {e}")

    def _kernels(self, model):
        from backend_tf.model_transformer import TransformerBottleneckBlock

        return [
            (b.conv_down.conv.kernel_size[0], b.conv_up.conv.kernel_size[0])
            for b in model.blocks
            if isinstance(b, TransformerBottleneckBlock)
        ]

    def _model(self, arch):
        from backend_tf.model import P3achyGoModel

        return P3achyGoModel.create(
            ModelConfig.from_generic_arch(arch),
            BOARD_LEN,
            NUM_PLANES,
            NUM_FEATURES,
            "t",
        )

    def test_new_model_defaults_to_1x1_and_records_it(self):
        model = self._model(copy.deepcopy(ARCH))
        self.assertEqual(self._kernels(model), [(1, 1), (1, 1)])
        for _, cfg in model.get_config()["generic_arch"]["trunk"]:
            self.assertEqual(cfg["proj_conv_size"], 1)

    def test_from_config_without_key_builds_3x3(self):
        from backend_tf.model import P3achyGoModel

        config = self._model(copy.deepcopy(ARCH)).get_config()
        for _, cfg in config["generic_arch"]["trunk"]:
            del cfg["proj_conv_size"]
        self.assertEqual(
            self._kernels(P3achyGoModel.from_config(config)), [(3, 3), (3, 3)]
        )


if __name__ == "__main__":
    unittest.main()
