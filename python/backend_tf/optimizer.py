import re

import keras
from keras.src import ops


@keras.saving.register_keras_serializable(package="p3achygo")
class ConvMuon(keras.optimizers.Muon):
    """Muon variant that applies Newton-Schulz to conv weights by flattening
    them to 2D ([out_channels, in_channels*H*W]) before the NS iterations,
    then restoring the original shape.

    The stock Muon routes all non-2D variables to AdamW. This subclass instead
    allows any variable whose effective 2D dims (after flattening) are both > 4
    to use the Muon update path, which includes conv weights.

    Momentum follows the EMA convention used by torch upstream and the
    Keller-Jordan reference (`m = m*β + g*(1-β)`, with symmetric Nesterov
    `update = g*(1-β) + m*β`), not Keras stock Muon's plain accumulation
    (`m = m*β + g`).

    Args:
        wd_lr_exponent: If set, scales the Muon weight decay as
            wd × (lr / wd_lr_max)^wd_lr_exponent each step, matching KataGo
            upstream's sublinear WD decay (exponent=0.70). Defaults to None
            (constant WD).
        wd_lr_max: The reference LR at which `weight_decay` is calibrated
            (typically the peak/starting LR). Required when wd_lr_exponent
            is set. Defaults to None.

    Per-category WD scale factors (hardcoded in `_WD_SCALE_FACTORS`):
    norm scale params (BN γ + RMSNorm scale), norm shift params (BN β),
    body biases, head biases, and attention Q/K/V/O projection kernels
    each receive `factor × weight_decay` instead of going through the
    AdamW/Muon paths. lr-ratio scaling (`wd_lr_exponent`) is applied.
    """

    _GAMMA_SUFFIXES = ("/gamma", ".gamma", "/scale", ".scale")
    _BETA_SUFFIXES = ("/beta", ".beta")
    _BIAS_SUFFIXES = ("/bias", ".bias")
    _QKVO_SUFFIXES = (
        "/query/kernel",
        "/key/kernel",
        "/value/kernel",
        "/output/kernel",
    )

    _WD_SCALE_FACTORS = {
        "gamma": 0.1,
        "beta": 1e-3,
        "body_bias": 1e-2,
        "head_bias": 1e-2,
        "qkvo": 0.5,
    }

    def __init__(
        self,
        *args,
        wd_lr_exponent=None,
        wd_lr_max=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wd_lr_exponent = wd_lr_exponent
        self.wd_lr_max = wd_lr_max

    def _should_use_adamw(self, variable):
        shape = variable.shape
        if len(shape) < 2:
            return True
        # Keras conv kernels are [H, W, in_channels, out_channels] — out is last.
        # Flatten to [H*W*in, out] and treat those as the effective 2D dims.
        out_dim = shape[-1]
        flat_dim = 1
        for d in shape[:-1]:
            flat_dim *= d
        # NS iterations are meaningless on tiny matrices
        if out_dim <= 4 or flat_dim <= 4:
            return True
        if self.exclude_embeddings and "embedding" in variable.path.lower():
            return True
        for keyword in self.exclude_layers:
            if re.search(keyword, variable.path):
                return True
        return False

    def _muon_update_step(self, gradient, variable, lr, m):
        # EMA momentum: m = m*β + g*(1-β)  (matches torch.optim.Muon and
        # Keller-Jordan reference; differs from Keras stock plain accumulation).
        one_minus_beta = 1 - self.momentum
        self.assign_add(m, ops.multiply(ops.subtract(gradient, m), one_minus_beta))
        # Symmetric Nesterov: update = g*(1-β) + m*β
        if self.nesterov:
            g = ops.add(
                ops.multiply(gradient, one_minus_beta),
                ops.multiply(m, self.momentum),
            )
        else:
            g = m

        original_shape = g.shape
        needs_flatten = len(original_shape) > 2
        if needs_flatten:
            # Keras conv layout: [H, W, in, out] → flatten to [H*W*in, out].
            # NS handles the tall-matrix case (H*W*in > out) with an internal
            # transpose, so this is equivalent to [out, H*W*in] row-wise.
            g_2d = ops.reshape(g, [-1, original_shape[-1]])
        else:
            g_2d = g

        update_2d = self.zeropower_via_newtonschulz5(g_2d, self.ns_steps)

        # Apply lr_adjust while still 2D so shape[0]/shape[1] index correctly
        # into [H*W*in, out] rather than the original 4D shape.
        scaled_2d = self.lr_adjust(lr * update_2d)

        if needs_flatten:
            scaled = ops.reshape(scaled_2d, original_shape)
        else:
            scaled = scaled_2d

        self.assign_sub(variable, scaled)

    def _wd_category(self, variable):
        """Classify variable for `_WD_SCALE_FACTORS` lookup.

        Returns one of "gamma", "beta", "body_bias", "head_bias", "qkvo",
        or None if the variable is not in any scaled category.
        """
        p = variable.path.lower()
        if p.endswith(self._GAMMA_SUFFIXES):
            return "gamma"
        if p.endswith(self._BETA_SUFFIXES):
            return "beta"
        if p.endswith(self._BIAS_SUFFIXES):
            is_head = "policy_head" in p or "value_head" in p
            return "head_bias" if is_head else "body_bias"
        if "transformer_attention" in p and p.endswith(self._QKVO_SUFFIXES):
            return "qkvo"
        return None

    def _lr_scale(self):
        if self.wd_lr_exponent is not None and self.wd_lr_max is not None:
            lr = ops.cast(self.learning_rate, "float32")
            lr_ratio = lr / ops.cast(self.wd_lr_max, "float32")
            # Clamp to (0, 1] — WD should not exceed the base value.
            lr_ratio = ops.minimum(lr_ratio, 1.0)
            return ops.power(lr_ratio, self.wd_lr_exponent)
        else:
            return 1.0

    def _apply_weight_decay(self, variables):
        for variable in variables:
            if not self._use_weight_decay(variable):
                continue
            category = self._wd_category(variable)
            scale_factor = self._WD_SCALE_FACTORS.get(category) if category else None
            if scale_factor is not None and self.weight_decay is not None:
                wd_value = self.weight_decay * scale_factor
            elif self._should_use_adamw(variable):
                wd_value = self.adam_weight_decay
            else:
                wd_value = self.weight_decay
            if wd_value is None:
                continue
            wd = ops.cast(wd_value, variable.dtype)
            lr = ops.cast(self.learning_rate, variable.dtype)
            lr_scale_factor = ops.cast(self._lr_scale(), variable.dtype)
            variable.assign(variable - variable * wd * lr * lr_scale_factor)

    def get_config(self):
        config = super().get_config()
        config["wd_lr_exponent"] = self.wd_lr_exponent
        config["wd_lr_max"] = self.wd_lr_max
        return config
