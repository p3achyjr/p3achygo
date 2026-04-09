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

    Args:
        scale_weight_decay_by_rms: If True, the Muon weight-decay step is
            multiplied by the same RMS scale factor applied to the gradient
            step (sqrt(max(flat_dim, out_dim)) * rms_rate). This makes the
            per-element grad/WD ratio identical across all body layer shapes,
            matching KataGo upstream's behavior. Defaults to False to preserve
            backward compatibility with older checkpoints.
        wd_lr_exponent: If set, scales the Muon weight decay as
            wd × (lr / wd_lr_max)^wd_lr_exponent each step, matching KataGo
            upstream's sublinear WD decay (exponent=0.70). Defaults to None
            (constant WD). Has no effect on AdamW variables.
        wd_lr_max: The reference LR at which `weight_decay` is calibrated
            (typically the peak/starting LR). Required when wd_lr_exponent
            is set. Defaults to None.
    """

    def __init__(
        self,
        *args,
        scale_weight_decay_by_rms=False,
        wd_lr_exponent=None,
        wd_lr_max=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_weight_decay_by_rms = scale_weight_decay_by_rms
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
        self.assign_add(m, ops.add(gradient, m * (self.momentum - 1)))
        g = ops.add(gradient, self.momentum * m) if self.nesterov else m

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

    def _wd_rms_scale(self, variable):
        """RMS scale factor used by lr_adjust for a given variable's 2D shape.

        Matches the scaling applied to the gradient step in _muon_update_step,
        using the same [H*W*in, out] flattening convention. Returns 1.0 if
        rms_rate is None (Moonlight scaling disabled).
        """
        if self.rms_rate is None:
            return 1.0
        shape = variable.shape
        out_dim = shape[-1]
        flat_dim = 1
        for d in shape[:-1]:
            flat_dim *= d
        return float(max(flat_dim, out_dim)) ** 0.5 * self.rms_rate

    def _apply_weight_decay(self, variables):
        for variable in variables:
            if not self._use_weight_decay(variable):
                continue
            if self._should_use_adamw(variable):
                wd_value = self.adam_weight_decay
                rms_scale = 1.0
                lr_scale_factor = 1.0
            else:
                wd_value = self.weight_decay
                rms_scale = (
                    self._wd_rms_scale(variable)
                    if self.scale_weight_decay_by_rms
                    else 1.0
                )
                if self.wd_lr_exponent is not None and self.wd_lr_max is not None:
                    lr = ops.cast(self.learning_rate, "float32")
                    lr_ratio = lr / ops.cast(self.wd_lr_max, "float32")
                    # Clamp to (0, 1] — WD should not exceed the base value.
                    lr_ratio = ops.minimum(lr_ratio, 1.0)
                    lr_scale_factor = ops.power(lr_ratio, self.wd_lr_exponent)
                else:
                    lr_scale_factor = 1.0
            if wd_value is None:
                continue
            wd = ops.cast(wd_value, variable.dtype)
            lr = ops.cast(self.learning_rate, variable.dtype)
            lr_scale_factor = ops.cast(lr_scale_factor, variable.dtype)
            variable.assign(variable - variable * wd * lr * rms_scale * lr_scale_factor)

    def get_config(self):
        config = super().get_config()
        config["scale_weight_decay_by_rms"] = self.scale_weight_decay_by_rms
        config["wd_lr_exponent"] = self.wd_lr_exponent
        config["wd_lr_max"] = self.wd_lr_max
        return config
