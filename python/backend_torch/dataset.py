"""Torch-native ChunkDataset.

Reads a single .tfrecord.zz chunk via the patched tfrecord.tfrecord_iterator
(zlib support — see backend_torch/zlib_tfrecord.py), parses each record via
tfrecord.example_pb2, runs the shared numpy expand (transforms.expand_bytes)
inside an IterableDataset. Yields batched 20-tuples of torch tensors via
DataLoader's default collate.

Read-once guarantee: with `num_workers > 0`, records are sharded round-robin
across workers (worker N owns indices {N, N+W, N+2W, …}). Each record is
expanded by exactly one worker. Batch composition / inter-batch order is
NOT deterministic across worker counts; only the multiset of records read
is invariant.

For deterministic order, use `num_workers=0`.
"""

from __future__ import annotations

import torch
import torch.utils.data

import backend_torch.zlib_tfrecord  # noqa: F401  (applies zlib patch)
from tfrecord import reader as _tfreader

import transforms


class _ChunkIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            worker_id, n_workers = 0, 1
        else:
            worker_id, n_workers = info.id, info.num_workers
        # Round-robin sharding ensures every record is owned by exactly one
        # worker. Each worker still walks the full zlib stream (no random
        # access in zlib), but only expands its share — and `expand` is the
        # dominant cost.
        for i, view in enumerate(
            _tfreader.tfrecord_iterator(self._path, compression_type="zlib")
        ):
            if i % n_workers == worker_id:
                yield transforms.expand_bytes(bytes(view))


def _count_records(path: str) -> int:
    return sum(1 for _ in _tfreader.tfrecord_iterator(path, compression_type="zlib"))


def iter_records(path: str):
    """Yield per-record 20-tuples (numpy) without batching."""
    for view in _tfreader.tfrecord_iterator(path, compression_type="zlib"):
        yield transforms.expand_bytes(bytes(view))


class ChunkDataset:
    """Iterable yielding batched 20-tuples of torch tensors from a chunk.

    Front-to-back. Final batch may be partial. `len(ds)` returns batch count.
    With `num_workers > 0`, each record is read once but batch composition is
    not order-preserving relative to `num_workers=0`.

    Tensors are moved to `device` (default: cuda if available, else cpu)
    before being yielded, mirroring TF's auto-placement behavior so consumers
    can stay device-agnostic.
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        *,
        num_workers: int = 0,
        device=None,
    ):
        self._path = path
        self._batch_size = batch_size
        self._device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._impl = torch.utils.data.DataLoader(
            _ChunkIterableDataset(path),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self._num_records: int | None = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __iter__(self):
        device = self._device
        for batch in self._impl:
            yield tuple(t.to(device) if hasattr(t, "to") else t for t in batch)

    def __len__(self) -> int:
        if self._num_records is None:
            self._num_records = _count_records(self._path)
        n, b = self._num_records, self._batch_size
        return (n + b - 1) // b
