"""Parity tests for the zlib tfrecord patch (backend_torch/zlib_tfrecord.py).

TF and `tfrecord` register conflicting proto descriptors, so we cannot
import both in the same process. The test orchestrates a subprocess that
uses TF to write a synthetic `.tfrecord.zz` and dump reference data
(raw record bytes + parsed features); the parent process imports
`tfrecord` and asserts byte- and feature-level parity.
"""

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest


# ---------- TF subprocess: write + read reference ---------- #

_TF_HELPER_SCRIPT = r"""
import pickle
import sys

import numpy as np
import tensorflow as tf

out_path, ref_path = sys.argv[1], sys.argv[2]

# Write a few records with mixed feature types and varied sizes.
records_to_write = []
for i in range(5):
    feature = {
        "f_int":   tf.train.Feature(int64_list=tf.train.Int64List(value=[i, i*2, i*3])),
        "f_float": tf.train.Feature(float_list=tf.train.FloatList(value=[0.1*i, 0.2*i])),
        "f_bytes": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"x" * (10 + i*7)])),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    records_to_write.append(ex.SerializeToString())

opts = tf.io.TFRecordOptions(compression_type="ZLIB")
with tf.io.TFRecordWriter(out_path, options=opts) as w:
    for r in records_to_write:
        w.write(r)

# Read back via TF and dump (a) raw record bytes (b) parsed feature dicts.
tf_raw = []
tf_parsed = []
for raw in tf.data.TFRecordDataset(out_path, compression_type="ZLIB"):
    raw_bytes = raw.numpy()
    tf_raw.append(raw_bytes)
    ex = tf.train.Example()
    ex.ParseFromString(raw_bytes)
    parsed = {}
    for key, feat in ex.features.feature.items():
        kind = feat.WhichOneof("kind")
        if kind == "int64_list":
            parsed[key] = ("int", list(feat.int64_list.value))
        elif kind == "float_list":
            parsed[key] = ("float", list(feat.float_list.value))
        elif kind == "bytes_list":
            parsed[key] = ("bytes", list(feat.bytes_list.value))
    tf_parsed.append(parsed)

with open(ref_path, "wb") as f:
    pickle.dump({"raw": tf_raw, "parsed": tf_parsed}, f)
"""


# ---------- main process: tfrecord side ---------- #

# Importing this module monkey-patches tfrecord.reader.tfrecord_iterator.
import backend_torch.zlib_tfrecord  # noqa: F401
from tfrecord import reader as tfreader
from tfrecord.torch.dataset import TFRecordDataset


class ZlibTFRecordParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="zlib_tfrecord_test_")
        cls.data_path = os.path.join(cls.tmpdir, "test.tfrecord.zz")
        cls.ref_path = os.path.join(cls.tmpdir, "ref.pkl")
        cls.helper_path = os.path.join(cls.tmpdir, "tf_helper.py")
        with open(cls.helper_path, "w") as f:
            f.write(_TF_HELPER_SCRIPT)
        subprocess.run(
            [sys.executable, cls.helper_path, cls.data_path, cls.ref_path],
            check=True,
            capture_output=True,
        )
        with open(cls.ref_path, "rb") as f:
            cls.ref = pickle.load(f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_raw_bytes_parity(self):
        ours = [
            bytes(view)
            for view in tfreader.tfrecord_iterator(
                self.data_path, compression_type="zlib"
            )
        ]
        ref = self.ref["raw"]
        self.assertEqual(len(ours), len(ref))
        for i, (a, b) in enumerate(zip(ours, ref)):
            self.assertEqual(a, b, f"record {i} byte mismatch")

    def test_feature_parity_via_dataset(self):
        ds = TFRecordDataset(
            self.data_path,
            index_path=None,
            compression_type="zlib",
            description=None,
        )
        ours = list(ds)
        ref = self.ref["parsed"]
        self.assertEqual(len(ours), len(ref))
        for i, (parsed, ref_dict) in enumerate(zip(ours, ref)):
            self.assertEqual(
                set(parsed.keys()),
                set(ref_dict.keys()),
                f"record {i} feature keys differ",
            )
            for key, (kind, ref_val) in ref_dict.items():
                got = parsed[key]
                if kind == "int":
                    self.assertEqual(list(got), ref_val, f"r{i} {key}")
                elif kind == "float":
                    # FloatList round-trips bit-for-bit through the same
                    # protobuf encoding both paths use.
                    self.assertEqual(list(got), ref_val, f"r{i} {key}")
                elif kind == "bytes":
                    # Single-element bytes_list comes back as scalar bytes
                    # in tfrecord; ref_val is a list of bytes.
                    if len(ref_val) == 1:
                        self.assertEqual(got, ref_val[0], f"r{i} {key}")
                    else:
                        self.assertEqual(list(got), ref_val, f"r{i} {key}")


if __name__ == "__main__":
    unittest.main()
