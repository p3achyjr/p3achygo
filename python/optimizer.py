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
    """

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
