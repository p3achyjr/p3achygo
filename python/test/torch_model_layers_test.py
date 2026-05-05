"""Unit tests for backend_torch/model_layers_common.py.

Tests run under KERAS_BACKEND=tensorflow (keras not needed in backend_torch
layers). Each test:
  1. Builds a small keras layer with the same config.
  2. Builds the torch equivalent.
  3. Copies weights.
  4. Runs both on identical input.
  5. Asserts outputs match within atol=1e-4.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import unittest
import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.insert(0, "python")


# ---------------------------------------------------------------------------
# Weight-copy helper: keras ConvBlock → torch ConvBlock
# ---------------------------------------------------------------------------


def _copy_keras_conv_block(keras_block, torch_block):
    """Copy weights from a keras ConvPreActivation into a torch ConvBlock.

    keras layout (via direct attribute access on the layer):
      conv.kernel  : (H, W, Cin, Cout)
      norm_layer.gamma / beta / moving_mean / moving_variance : (Cin,)
    torch layout:
      conv.weight  : (Cout, Cin, H, W)
      bn.weight / bias / running_mean / running_var : (Cin,)
    """
    conv_kernel = keras_block.conv.kernel.numpy()  # (H,W,Cin,Cout)
    bn_gamma = keras_block.norm_layer.gamma.numpy()
    bn_beta = keras_block.norm_layer.beta.numpy()
    moving_mean = keras_block.norm_layer.moving_mean.numpy()
    moving_var = keras_block.norm_layer.moving_variance.numpy()

    with torch.no_grad():
        torch_block.conv.weight.copy_(torch.tensor(conv_kernel).permute(3, 2, 0, 1))
        torch_block.bn.weight.copy_(torch.tensor(bn_gamma))
        torch_block.bn.bias.copy_(torch.tensor(bn_beta))
        torch_block.bn.running_mean.copy_(torch.tensor(moving_mean))
        torch_block.bn.running_var.copy_(torch.tensor(moving_var))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class MishTest(unittest.TestCase):

    def test_mish_matches_keras(self):
        import keras
        from backend_torch.model_layers_common import mish

        x_np = np.random.randn(4, 8).astype(np.float32)
        x_t = torch.tensor(x_np)
        keras_out = keras.activations.mish(x_np).numpy()
        torch_out = mish(x_t).detach().numpy()
        np.testing.assert_allclose(
            torch_out, keras_out, atol=1e-5, err_msg="mish output mismatch"
        )

    def test_mish_shape_preserved(self):
        from backend_torch.model_layers_common import mish

        x = torch.randn(3, 7, 5, 5)
        self.assertEqual(mish(x).shape, x.shape)


class ConvPreActivationTest(unittest.TestCase):

    def setUp(self):
        import keras
        from model_layers_common import make_conv_block as keras_make_conv_block

        self.in_ch = 16
        self.out_ch = 32
        self.ks = 3
        self.batch = 2
        self.h = self.w = 7

        # Build and call keras block (must call to create variables).
        # keras's make_conv_block(output_channels, conv_size) makes a ConvPreActivation.
        self.keras_block = keras_make_conv_block(self.out_ch, self.ks)
        # Build variables by running a forward pass.
        dummy = np.zeros((self.batch, self.h, self.w, self.in_ch), dtype=np.float32)
        self.keras_block(dummy, training=False)

    def _make_torch_block(self):
        from backend_torch.model_layers_common import ConvPreActivation

        block = ConvPreActivation(self.in_ch, self.out_ch, self.ks)
        block.eval()
        return block

    def test_output_shape(self):
        block = self._make_torch_block()
        x = torch.zeros(self.batch, self.in_ch, self.h, self.w)
        with torch.no_grad():
            y = block(x)
        self.assertEqual(y.shape, (self.batch, self.out_ch, self.h, self.w))

    def test_parity_with_keras(self):
        import keras

        block = self._make_torch_block()
        _copy_keras_conv_block(self.keras_block, block)
        block.eval()

        # NHWC input for keras, NCHW for torch
        x_nhwc = np.random.randn(self.batch, self.h, self.w, self.in_ch).astype(
            np.float32
        )
        x_nchw = torch.tensor(x_nhwc).permute(0, 3, 1, 2).contiguous()

        keras_out = self.keras_block(x_nhwc, training=False).numpy()
        with torch.no_grad():
            torch_out = block(x_nchw).permute(0, 2, 3, 1).numpy()

        # GPU (TF) vs CPU (torch) float32 arithmetic introduces ~1e-3 divergence.
        np.testing.assert_allclose(
            torch_out,
            keras_out,
            atol=1e-3,
            err_msg="ConvPreActivation output mismatch vs keras",
        )

    def test_channels_last_same_result(self):
        """channels_last memory format must produce identical values to channels_first."""
        from backend_torch.model_layers_common import ConvPreActivation

        block_cf = ConvPreActivation(self.in_ch, self.out_ch, self.ks).eval()
        block_cl = ConvPreActivation(self.in_ch, self.out_ch, self.ks).eval()
        # Copy identical weights
        with torch.no_grad():
            for p_cl, p_cf in zip(block_cl.parameters(), block_cf.parameters()):
                p_cl.copy_(p_cf)

        block_cl = block_cl.to(memory_format=torch.channels_last)

        x = torch.randn(self.batch, self.in_ch, self.h, self.w)
        x_cl = x.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            out_cf = block_cf(x)
            out_cl = block_cl(x_cl)

        np.testing.assert_allclose(
            out_cf.numpy(),
            out_cl.numpy(),
            atol=1e-6,
            err_msg="channels_last and channels_first results differ",
        )


class ConvPostActivationTest(unittest.TestCase):

    def test_output_shape(self):
        from backend_torch.model_layers_common import ConvPostActivation

        block = ConvPostActivation(16, 32, 3).eval()
        x = torch.zeros(2, 16, 5, 5)
        with torch.no_grad():
            y = block(x)
        self.assertEqual(y.shape, (2, 32, 5, 5))


class MakeConvTest(unittest.TestCase):

    def test_same_padding(self):
        from backend_torch.model_layers_common import make_conv

        for ks in (1, 3, 5):
            conv = make_conv(8, 16, ks)
            x = torch.zeros(1, 8, 19, 19)
            self.assertEqual(
                conv(x).shape,
                (1, 16, 19, 19),
                f"'same' padding failed for kernel_size={ks}",
            )

    def test_no_bias_by_default(self):
        from backend_torch.model_layers_common import make_conv

        conv = make_conv(8, 16, 3)
        self.assertIsNone(conv.bias)

    def test_bias_when_requested(self):
        from backend_torch.model_layers_common import make_conv

        conv = make_conv(8, 16, 3, use_bias=True)
        self.assertIsNotNone(conv.bias)


class MakeDenseTest(unittest.TestCase):

    def test_shape(self):
        from backend_torch.model_layers_common import make_dense

        fc = make_dense(8, 32)
        x = torch.zeros(4, 8)
        self.assertEqual(fc(x).shape, (4, 32))


class MakeBnTest(unittest.TestCase):

    def test_params(self):
        from backend_torch.model_layers_common import make_bn

        bn = make_bn(64)
        self.assertAlmostEqual(bn.momentum, 0.01)
        self.assertAlmostEqual(bn.eps, 1e-3)


if __name__ == "__main__":
    unittest.main()
