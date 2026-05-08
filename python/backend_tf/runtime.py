"""TF-side runtime helpers: GPU config + TensorBoard scalar logging."""

import tensorflow as tf
import keras


def configure_gpu(mixed_precision_policy: str = "mixed_float16"):
    physical_gpus = tf.config.list_physical_devices("GPU")
    assert physical_gpus, "No GPUs detected."
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.mixed_precision.set_global_policy(mixed_precision_policy)


def gpu_count() -> int:
    """Number of visible GPUs (after CUDA_VISIBLE_DEVICES masking)."""
    return len(tf.config.list_physical_devices("GPU"))


class SummaryWriter:
    """tf.summary-backed scalar writer with the agnostic API."""

    def __init__(self, log_dir: str):
        self._impl = tf.summary.create_file_writer(log_dir)

    def scalar(self, name: str, value, step: int):
        with self._impl.as_default():
            tf.summary.scalar(name, value, step=step)

    def close(self):
        if hasattr(self._impl, "close"):
            self._impl.close()
