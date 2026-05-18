"""Parity test: dataset.ChunkDataset(tf) vs dataset.ChunkDataset(torch).

Both backends must yield bit-equal numpy values for the same chunk when
randomness inside transforms.expand is stubbed deterministically.

TF and tfrecord proto descriptors collide if loaded in the same process,
so the TF backend reads the chunk in a subprocess and dumps batches to
a pickle. The parent process reads the same chunk via the torch backend
and compares element-wise.
"""

import os
import pickle
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

import numpy as np
from tfrecord import example_pb2 as pb
from tfrecord.writer import TFRecordWriter

import symmetry as sym
import transforms
from constants import (
    BOARD_LEN,
    NUM_MOVES,
    NUM_V_BUCKETS,
    BLACK,
    WHITE,
)
from backend_torch.dataset import ChunkDataset as TorchChunkDataset


# Subprocess script: imports TF + backend_tf.dataset, reads a chunk with
# deterministic randomness, pickles batches. Never imports tfrecord.
_HELPER_SCRIPT = r"""
import os, pickle, sys

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np

import symmetry as sym
# Deterministic randomness: IDENTITY symmetry, mask off (uniform=0.5).
sym.get_random_symmetry = lambda: sym.IDENTITY
np.random.uniform = lambda *a, **k: 0.5

from backend_tf.dataset import ChunkDataset

path, out_path, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = ChunkDataset(path, batch_size)
batches = []
for batch in ds:
    batches.append(tuple(t.numpy() for t in batch))
with open(out_path, "wb") as f:
    pickle.dump({"batches": batches, "len": len(ds)}, f)
"""


def _bytes_feat(value: bytes) -> pb.Feature:
    return pb.Feature(bytes_list=pb.BytesList(value=[value]))


def _float_feat(value: float) -> pb.Feature:
    return pb.Feature(float_list=pb.FloatList(value=[value]))


def _make_record(idx: int, schema_new: bool) -> bytes:
    board = np.zeros(BOARD_LEN * BOARD_LEN, dtype=np.int8)
    board[idx * BOARD_LEN + idx] = BLACK if idx % 2 else WHITE
    last_moves = np.array([idx, idx + 1, -1, -1, -1], dtype=np.int16)
    pi = np.zeros(NUM_MOVES, dtype=np.float32)
    pi[idx] = 1.0
    zg = np.zeros(BOARD_LEN * BOARD_LEN, dtype=np.int8)

    feature = {
        "bsize": _bytes_feat(np.array([BOARD_LEN], dtype=np.uint8).tobytes()),
        "board": _bytes_feat(board.tobytes()),
        "last_moves": _bytes_feat(last_moves.tobytes()),
        "stones_atari": _bytes_feat(zg.tobytes()),
        "stones_two_liberties": _bytes_feat(zg.tobytes()),
        "stones_three_liberties": _bytes_feat(zg.tobytes()),
        "stones_in_ladder": _bytes_feat(zg.tobytes()),
        "color": _bytes_feat(
            np.array([BLACK if idx % 2 else WHITE], dtype=np.int8).tobytes()
        ),
        "komi": _float_feat(0.5 * idx),
        "own": _bytes_feat(zg.tobytes()),
        "pi": _bytes_feat(pi.tobytes()),
        "pi_aux": _bytes_feat(np.array([idx], dtype=np.int16).tobytes()),
        "score_margin": _float_feat(float(idx - 2.0)),
        "q6": _float_feat(0.1 * idx),
        "q16": _float_feat(0.2 * idx),
        "q50": _float_feat(0.3 * idx),
        "q6_score": _float_feat(0.4 * idx),
        "q16_score": _float_feat(0.5 * idx),
        "q50_score": _float_feat(0.6 * idx),
    }
    if schema_new:
        pi_aux_dist = np.zeros(NUM_MOVES, dtype=np.float32)
        pi_aux_dist[idx] = 1.0
        vdist = np.zeros(NUM_V_BUCKETS, dtype=np.uint32)
        vdist[idx % NUM_V_BUCKETS] = idx + 1
        feature["pi_aux_dist"] = _bytes_feat(pi_aux_dist.tobytes())
        feature["mcts_value_dist"] = _bytes_feat(vdist.tobytes())
    return pb.Example(features=pb.Features(feature=feature)).SerializeToString()


def _write_zlib_chunk(path, records):
    framing = bytearray()
    for record in records:
        length_bytes = struct.pack("<Q", len(record))
        framing += length_bytes
        framing += TFRecordWriter.masked_crc(length_bytes)
        framing += record
        framing += TFRecordWriter.masked_crc(record)
    with open(path, "wb") as f:
        f.write(zlib.compress(bytes(framing)))


class DatasetParityTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="chunk_dataset_parity_")
        cls.path = os.path.join(cls.tmpdir, "test.tfrecord.zz")
        cls.tf_pkl = os.path.join(cls.tmpdir, "tf.pkl")
        cls.helper = os.path.join(cls.tmpdir, "tf_helper.py")
        cls.batch_size = 2

        # 3 old + 4 new schema records → 7 records → 4 batches at bs=2.
        records = [_make_record(i, False) for i in range(3)] + [
            _make_record(i + 3, True) for i in range(4)
        ]
        _write_zlib_chunk(cls.path, records)

        with open(cls.helper, "w") as f:
            f.write(_HELPER_SCRIPT)

        env = {
            **os.environ,
            "PYTHONPATH": "python:python/test",
            "TF_CPP_MIN_LOG_LEVEL": "3",
        }
        result = subprocess.run(
            [sys.executable, cls.helper, cls.path, cls.tf_pkl, str(cls.batch_size)],
            check=True,
            env=env,
            capture_output=True,
        )
        with open(cls.tf_pkl, "rb") as f:
            cls.tf_ref = pickle.load(f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _torch_batches(self):
        # Same deterministic stubs as the TF subprocess.
        saved_sym = sym.get_random_symmetry
        saved_uniform = np.random.uniform
        sym.get_random_symmetry = lambda: sym.IDENTITY
        np.random.uniform = lambda *a, **k: 0.5
        try:
            ds = TorchChunkDataset(self.path, self.batch_size)
            return list(ds), len(ds)
        finally:
            sym.get_random_symmetry = saved_sym
            np.random.uniform = saved_uniform

    def test_batch_count(self):
        _, torch_len = self._torch_batches()
        self.assertEqual(torch_len, 4)
        self.assertEqual(self.tf_ref["len"], 4)

    def test_batch_parity(self):
        torch_batches, _ = self._torch_batches()
        tf_batches = self.tf_ref["batches"]
        self.assertEqual(len(torch_batches), len(tf_batches))

        for bi, (tb, fb) in enumerate(zip(torch_batches, tf_batches)):
            self.assertEqual(len(tb), len(fb))
            for ei, (t_elem, f_elem) in enumerate(zip(tb, fb)):
                t_np = np.asarray(t_elem)
                f_np = np.asarray(f_elem)
                self.assertEqual(
                    t_np.shape,
                    f_np.shape,
                    f"batch {bi} elem {ei}: shape mismatch "
                    f"{t_np.shape} vs {f_np.shape}",
                )
                self.assertEqual(
                    t_np.dtype.kind,
                    f_np.dtype.kind,
                    f"batch {bi} elem {ei}: dtype kind mismatch "
                    f"{t_np.dtype} vs {f_np.dtype}",
                )
                if t_np.dtype.kind == "f":
                    np.testing.assert_allclose(
                        t_np,
                        f_np,
                        atol=1e-5,
                        rtol=1e-5,
                        err_msg=f"batch {bi} elem {ei}",
                    )
                else:
                    np.testing.assert_array_equal(
                        t_np, f_np, err_msg=f"batch {bi} elem {ei}"
                    )


if __name__ == "__main__":
    unittest.main()
