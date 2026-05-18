"""TF-native ChunkDataset.

Reads a single .tfrecord.zz chunk via tf.data.TFRecordDataset, parses
each record with tf.train.Example, and runs the shared numpy expand
(transforms.expand) under tf.numpy_function. Yields batched 20-tuples
of TF tensors. Front-to-back, no shuffling.
"""

from __future__ import annotations

import tensorflow as tf

import transforms
from constants import BOARD_LEN, NUM_MOVES, SCORE_RANGE, NUM_V_BUCKETS


_DTYPES = (
    tf.float32,
    tf.float32,
    tf.int32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.int32,
    tf.float32,
    tf.bool,
    tf.int32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.float32,
    tf.int32,
    tf.bool,
)
_SHAPES = (
    (BOARD_LEN, BOARD_LEN, 15),
    (8,),
    (),
    (),
    (),
    (SCORE_RANGE,),
    (NUM_MOVES,),
    (),
    (NUM_MOVES,),
    (),
    (BOARD_LEN, BOARD_LEN),
    (),
    (),
    (),
    (),
    (),
    (),
    (2,),
    (NUM_V_BUCKETS,),
    (),
)


def _expand_inner(serialized_bytes):
    """Called via tf.numpy_function — receives a Python `bytes` object."""
    ex = tf.train.Example()
    ex.ParseFromString(serialized_bytes)
    return transforms.expand(ex.features.feature)


def _expand_tf(serialized):
    out = tf.numpy_function(_expand_inner, [serialized], _DTYPES)
    for tensor, shape in zip(out, _SHAPES):
        tensor.set_shape(shape)
    return tuple(out)


def _count_records(path):
    ds = tf.data.TFRecordDataset(path, compression_type="ZLIB")
    return int(ds.reduce(tf.constant(0, dtype=tf.int64), lambda c, _: c + 1))


def iter_records(path):
    """Yield per-record 20-tuples (numpy) without batching.

    Convenience for utility scripts that process records one at a time.
    """
    for raw in tf.data.TFRecordDataset(path, compression_type="ZLIB"):
        yield _expand_inner(raw.numpy())


class ChunkDataset:
    """Iterable yielding batched 20-tuples of TF tensors from a chunk.

    Front-to-back. Final batch may be partial. `len(ds)` returns batch count.
    """

    def __init__(self, path: str, batch_size: int):
        self._path = path
        self._batch_size = batch_size
        ds = tf.data.TFRecordDataset(path, compression_type="ZLIB")
        # num_parallel_calls=1: tf.numpy_function holds the GIL inside the
        # Python callable. Using AUTOTUNE (≈ #CPUs) creates lock contention
        # and slows things down — empirically 1 is the best knob value.
        ds = ds.map(_expand_tf, num_parallel_calls=1)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        self._impl = ds
        self._num_records: int | None = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __iter__(self):
        return iter(self._impl)

    def __len__(self) -> int:
        if self._num_records is None:
            self._num_records = _count_records(self._path)
        n, b = self._num_records, self._batch_size
        return (n + b - 1) // b
