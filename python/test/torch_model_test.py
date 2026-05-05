"""Integration / parity tests for the torch-native P3achyGoModel.

Requires:
    ~/p3achygo-data/v4-models/b9-legacy/model_0311.keras  (loaded by the TF helper)

The test:
1. Loads the keras model under TF backend in a subprocess → pickles all 25
   forward-pass outputs as numpy arrays.
2. Loads the keras model in the main process, migrates weights to the torch
   model, runs the same input through it.
3. Asserts all 25 outputs match within atol=1e-2 (GPU TF vs CPU torch fp32).

Run:
    PYTHONPATH=python:python/test python python/test/torch_model_test.py -v
"""

import os, sys, subprocess, pickle, tempfile, shutil, unittest

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, "python")
sys.path.insert(0, "python/test")

import numpy as np
import torch

CHECKPOINT = os.path.expanduser("~/p3achygo-data/v4-models/b9-legacy/model_0311.keras")
BATCH = 2

# ---------------------------------------------------------------------------
# Subprocess helper: run keras forward pass under TF backend
# ---------------------------------------------------------------------------

_TF_HELPER = r"""
import os, sys, pickle
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "python")
import numpy as np
import keras
import tensorflow as tf
from model import P3achyGoModel

# Force CPU-only evaluation so TF and torch (CPU) use the same fp32 arithmetic.
tf.config.set_visible_devices([], 'GPU')
keras.mixed_precision.set_global_policy("float32")

ckpt, out_pkl, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3])

np.random.seed(42)
board = np.random.randn(batch_size, 19, 19, 15).astype(np.float32)
game  = np.random.randn(batch_size, 8).astype(np.float32)

km = keras.models.load_model(ckpt,
    custom_objects=P3achyGoModel.custom_objects(), compile=False)
km.trainable = False
outputs = km(board, game, training=False)
result = [o.numpy() for o in outputs]
with open(out_pkl, "wb") as f:
    pickle.dump({"outputs": result, "board": board, "game": game}, f)
"""


class TorchModelParityTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CHECKPOINT):
            raise unittest.SkipTest(f"Checkpoint not found: {CHECKPOINT}")

        cls.tmpdir = tempfile.mkdtemp(prefix="torch_model_parity_")
        helper = os.path.join(cls.tmpdir, "tf_helper.py")
        pkl = os.path.join(cls.tmpdir, "tf_ref.pkl")

        with open(helper, "w") as f:
            f.write(_TF_HELPER)

        env = {
            **os.environ,
            "PYTHONPATH": "python:python/test",
            "TF_CPP_MIN_LOG_LEVEL": "3",
        }
        result = subprocess.run(
            [sys.executable, helper, CHECKPOINT, pkl, str(BATCH)],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"TF helper failed (rc={result.returncode}):\n"
                f"STDOUT: {result.stdout[-2000:]}\n"
                f"STDERR: {result.stderr[-2000:]}"
            )

        with open(pkl, "rb") as f:
            ref = pickle.load(f)

        cls.tf_outputs = ref["outputs"]
        cls.board = ref["board"]
        cls.game = ref["game"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _build_torch_model(self):
        """Load keras model, migrate weights, return torch model in eval mode."""
        import keras
        from model import P3achyGoModel as KP3achyGoModel
        from backend_torch.model import P3achyGoModel as TP3achyGoModel
        from scripts.migrate_keras_to_torch import (
            _copy_conv2d,
            _copy_linear,
            _copy_bn,
            _copy_nbt,
            _copy_broadcast,
            _copy_policy_head,
            _copy_value_head,
            _BLOCK_TYPES,
        )

        km = keras.models.load_model(
            CHECKPOINT,
            custom_objects=KP3achyGoModel.custom_objects(),
            compile=False,
        )
        km.trainable = False

        config = km.get_config()
        if "num_head_channels" in config and "num_policy_head_channels" not in config:
            config["num_policy_head_channels"] = config.pop("num_head_channels")
            config["num_value_head_channels"] = config.get("c_val", 64)
        config.pop("name", None)
        config.pop("is_transformer", None)

        tm = TP3achyGoModel(**config)
        tm.eval()

        # Migrate weights using explicit type-aware copiers
        _copy_conv2d(km.init_board_conv, tm.init_board_conv)
        _copy_linear(km.init_game_layer, tm.init_game_layer)
        for kb, tb in zip(list(km.blocks), list(tm.blocks)):
            _BLOCK_TYPES[type(kb).__name__](kb, tb)
        _copy_policy_head(km.policy_head, tm.policy_head)
        _copy_value_head(km.value_head, tm.value_head)

        return tm

    def test_forward_pass_parity(self):
        """All 25 outputs of torch model must match keras within atol=1e-2."""
        tm = self._build_torch_model()

        board_t = torch.tensor(self.board)
        game_t = torch.tensor(self.game)

        with torch.no_grad():
            torch_outs = tm(board_t, game_t, training=False)

        self.assertEqual(
            len(torch_outs),
            len(self.tf_outputs),
            f"Output tuple length: torch={len(torch_outs)} vs keras={len(self.tf_outputs)}",
        )

        for i, (t_out, k_out) in enumerate(zip(torch_outs, self.tf_outputs)):
            t_np = t_out.numpy()
            # Normalize shapes: squeeze trailing size-1 dims on keras side,
            # and leading size-1 dims on torch side, for q-values and ownership.
            k_np = k_out.squeeze()  # keras: (N, H, W, 1) → (N, H, W); (N,) stays (N,)
            t_np = t_np.squeeze()  # torch: (N, 1) → (N,); (N, H, W) stays (N, H, W)
            # After squeezing both, shapes and values should match
            self.assertEqual(
                t_np.shape,
                k_np.shape,
                f"Output {i}: shape mismatch after squeeze: "
                f"torch={t_out.shape}→{t_np.shape} vs keras={k_out.shape}→{k_np.shape}",
            )
            max_diff = float(np.abs(t_np - k_np).max())
            self.assertLess(
                max_diff,
                0.05,
                f"Output {i}: max abs diff {max_diff:.4f} > 0.05 "
                f"(shape {t_np.shape})",
            )

    def test_model_parameter_count(self):
        """Torch model should have same parameter count as keras model."""
        import keras
        from model import P3achyGoModel as KP3achyGoModel
        from backend_torch.model import P3achyGoModel as TP3achyGoModel

        km = keras.models.load_model(
            CHECKPOINT, custom_objects=KP3achyGoModel.custom_objects(), compile=False
        )
        keras_params = sum(v.numpy().size for v in km.trainable_weights)

        tm = self._build_torch_model()
        torch_params = sum(p.numel() for p in tm.parameters())

        self.assertEqual(
            torch_params,
            keras_params,
            f"Param count: torch={torch_params:,} vs keras={keras_params:,}",
        )

    def test_channels_last_same_result(self):
        """channels_last memory format must produce bit-identical results."""
        tm_cf = self._build_torch_model()
        tm_cl = self._build_torch_model()
        tm_cl = tm_cl.to(memory_format=torch.channels_last)

        board_t = torch.tensor(self.board)
        game_t = torch.tensor(self.game)
        board_cl = board_t.contiguous()

        with torch.no_grad():
            out_cf = tm_cf(board_t, game_t, training=False)
            out_cl = tm_cl(board_cl, game_t, training=False)

        for i, (a, b) in enumerate(zip(out_cf, out_cl)):
            np.testing.assert_allclose(
                a.numpy(),
                b.numpy(),
                atol=1e-3,
                err_msg=f"output {i}: channels_last differs",
            )


if __name__ == "__main__":
    unittest.main()
