"""Torch-side runtime helpers: GPU config + TensorBoard scalar logging."""

import sys
import types

# tensorboard.compat treats the *existence* of `tensorboard.compat.notf` as an
# explicit opt-out from real TensorFlow: when found, tensorboard uses its
# bundled `tensorflow_stub` (which implements `tf.io.gfile` etc. without
# importing TF). Registering the marker here, before importing
# `torch.utils.tensorboard`, keeps TF out of the process and avoids the
# `tensorflow.BytesList` proto-descriptor collision with `tfrecord.example_pb2`
# that we use on the data path.
sys.modules.setdefault(
    "tensorboard.compat.notf",
    types.ModuleType("tensorboard.compat.notf"),
)

import torch
import keras
from torch.utils.tensorboard import SummaryWriter as _TBWriter


def configure_gpu(mixed_precision_policy: str = "mixed_float16"):
    assert torch.cuda.is_available(), "No GPUs detected."
    # torch's caching allocator already grows lazily — no memory_growth knob.
    keras.mixed_precision.set_global_policy(mixed_precision_policy)


class SummaryWriter:
    """torch.utils.tensorboard-backed scalar writer with the agnostic API."""

    def __init__(self, log_dir: str):
        self._impl = _TBWriter(log_dir)

    def scalar(self, name: str, value, step: int):
        self._impl.add_scalar(name, float(value), global_step=int(step))

    def close(self):
        if hasattr(self._impl, "close"):
            self._impl.close()
