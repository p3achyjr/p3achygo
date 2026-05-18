"""Backend-aware ChunkDataset dispatcher.

Selects backend_tf.dataset or backend_torch.dataset based on KERAS_BACKEND.
Both backends expose the same `ChunkDataset(path, batch_size)` interface,
yielding batched 20-tuples in the framework's native tensor type. Reads
the chunk in file order (no shuffling).
"""

from backend import BACKEND as _backend

if _backend == "torch":
    from backend_torch.dataset import ChunkDataset, iter_records
elif _backend == "tensorflow":
    from backend_tf.dataset import ChunkDataset, iter_records
else:
    raise ValueError(f"unsupported backend {_backend!r}")

__all__ = ["ChunkDataset", "iter_records"]
