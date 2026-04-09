"""Tests for transforms.py expand function."""

import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import transforms
from constants import *

# Index of each field in the tuple returned by expand().
_IDX_INPUT = 0
_IDX_INPUT_GLOBAL_STATE = 1
_IDX_COLOR = 2
_IDX_KOMI = 3
_IDX_SCORE = 4
_IDX_SCORE_ONE_HOT = 5
_IDX_POLICY = 6
_IDX_POLICY_AUX = 7
_IDX_POLICY_AUX_DIST = 8
_IDX_HAS_PI_AUX_DIST = 9
_IDX_OWN = 10
_IDX_Q6 = 11
_IDX_Q16 = 12
_IDX_Q50 = 13
_IDX_Q6_SCORE = 14
_IDX_Q16_SCORE = 15
_IDX_Q50_SCORE = 16
_IDX_GAME_OUTCOME = 17
_IDX_MCTS_VALUE_DIST = 18
_IDX_HAS_MCTS_VALUE_DIST = 19

_NUM_OUTPUTS = 20


def _make_example(
    *,
    pi_aux_dist=None,  # float32 ndarray of shape (NUM_MOVES,), or None for old schema
    mcts_value_dist=None,  # uint32 ndarray of shape (NUM_V_BUCKETS,), or None
    pi_aux_index=0,
    score_margin=0.0,
):
    """Build a serialized tf.train.Example. Omit optional fields when None."""
    bsize = BOARD_LEN
    last_moves = np.array([-1, -1, -1, -1, -1], dtype=np.int16)

    feature = {
        "bsize": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.array([bsize], dtype=np.uint8).tobytes()]
            )
        ),
        "board": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.zeros(bsize * bsize, dtype=np.int8).tobytes()]
            )
        ),
        "last_moves": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[last_moves.tobytes()])
        ),
        "stones_atari": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.zeros(bsize * bsize, dtype=np.int8).tobytes()]
            )
        ),
        "stones_two_liberties": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.zeros(bsize * bsize, dtype=np.int8).tobytes()]
            )
        ),
        "stones_three_liberties": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.zeros(bsize * bsize, dtype=np.int8).tobytes()]
            )
        ),
        "stones_in_ladder": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.zeros(bsize * bsize, dtype=np.int8).tobytes()]
            )
        ),
        "color": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.array([BLACK], dtype=np.int8).tobytes()]
            )
        ),
        "komi": tf.train.Feature(float_list=tf.train.FloatList(value=[6.5])),
        "own": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.zeros(bsize * bsize, dtype=np.int8).tobytes()]
            )
        ),
        "pi": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.zeros(bsize * bsize + 1, dtype=np.float32).tobytes()]
            )
        ),
        "pi_aux": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[np.array([pi_aux_index], dtype=np.int16).tobytes()]
            )
        ),
        "score_margin": tf.train.Feature(
            float_list=tf.train.FloatList(value=[score_margin])
        ),
        "q6": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
        "q16": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
        "q50": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
        "q6_score": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
        "q16_score": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
        "q50_score": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
    }

    if pi_aux_dist is not None:
        feature["pi_aux_dist"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[pi_aux_dist.astype(np.float32).tobytes()]
            )
        )
    if mcts_value_dist is not None:
        feature["mcts_value_dist"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[mcts_value_dist.astype(np.uint32).tobytes()]
            )
        )

    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


def _write_tfrecord(path, serialized_examples):
    with tf.io.TFRecordWriter(path) as w:
        for ex in serialized_examples:
            w.write(ex)


class TransformsTest(unittest.TestCase):

    def _old_schema_example(self, **kwargs):
        return _make_example(**kwargs)

    def _new_schema_example(self, **kwargs):
        pi_aux_dist = kwargs.pop("pi_aux_dist", np.zeros(NUM_MOVES, dtype=np.float32))
        mcts_value_dist = kwargs.pop(
            "mcts_value_dist", np.zeros(NUM_V_BUCKETS, dtype=np.uint32)
        )
        return _make_example(
            pi_aux_dist=pi_aux_dist, mcts_value_dist=mcts_value_dist, **kwargs
        )

    # ------------------------------------------------------------------
    # Basic shape / structure tests
    # ------------------------------------------------------------------

    def test_expand_output_count(self):
        result = transforms.expand(self._old_schema_example())
        self.assertEqual(len(result), _NUM_OUTPUTS)

    def test_expand_input_shape(self):
        result = transforms.expand(self._old_schema_example())
        self.assertEqual(result[_IDX_INPUT].shape, (BOARD_LEN, BOARD_LEN, 15))

    def test_expand_global_state_shape(self):
        result = transforms.expand(self._old_schema_example())
        self.assertEqual(result[_IDX_INPUT_GLOBAL_STATE].shape, (num_input_features(),))

    def test_expand_policy_shape(self):
        result = transforms.expand(self._old_schema_example())
        self.assertEqual(result[_IDX_POLICY].shape, (NUM_MOVES,))

    def test_expand_score_one_hot_shape(self):
        result = transforms.expand(self._old_schema_example())
        self.assertEqual(result[_IDX_SCORE_ONE_HOT].shape, (SCORE_RANGE,))

    def test_expand_q_score_shapes(self):
        result = transforms.expand(self._old_schema_example())
        for idx in (_IDX_Q6_SCORE, _IDX_Q16_SCORE, _IDX_Q50_SCORE):
            self.assertEqual(result[idx].shape, ())

    # ------------------------------------------------------------------
    # Old schema: optional fields absent → sentinels + flags=False
    # ------------------------------------------------------------------

    def test_old_schema_has_flags_false(self):
        result = transforms.expand(self._old_schema_example())
        self.assertFalse(result[_IDX_HAS_PI_AUX_DIST].numpy())
        self.assertFalse(result[_IDX_HAS_MCTS_VALUE_DIST].numpy())

    def test_old_schema_policy_aux_dist_is_zeros(self):
        result = transforms.expand(self._old_schema_example())
        dist = result[_IDX_POLICY_AUX_DIST].numpy()
        self.assertEqual(dist.shape, (NUM_MOVES,))
        np.testing.assert_array_equal(dist, np.zeros(NUM_MOVES, dtype=np.float32))

    def test_old_schema_mcts_value_dist_is_zeros(self):
        result = transforms.expand(self._old_schema_example())
        dist = result[_IDX_MCTS_VALUE_DIST].numpy()
        self.assertEqual(dist.shape, (NUM_V_BUCKETS,))
        np.testing.assert_array_equal(dist, np.zeros(NUM_V_BUCKETS, dtype=np.uint32))

    # ------------------------------------------------------------------
    # New schema: optional fields present → correct values + flags=True
    # ------------------------------------------------------------------

    def test_new_schema_has_flags_true(self):
        result = transforms.expand(self._new_schema_example())
        self.assertTrue(result[_IDX_HAS_PI_AUX_DIST].numpy())
        self.assertTrue(result[_IDX_HAS_MCTS_VALUE_DIST].numpy())

    def test_new_schema_policy_aux_dist_shape(self):
        result = transforms.expand(self._new_schema_example())
        self.assertEqual(result[_IDX_POLICY_AUX_DIST].shape, (NUM_MOVES,))

    def test_new_schema_mcts_value_dist_values(self):
        dist = np.zeros(NUM_V_BUCKETS, dtype=np.uint32)
        dist[10] = 3
        dist[40] = 7
        result = transforms.expand(self._new_schema_example(mcts_value_dist=dist))
        decoded = result[_IDX_MCTS_VALUE_DIST].numpy()
        # Decoded as int32 (same bit layout as uint32 for values < 2^31).
        self.assertEqual(decoded[10], 3)
        self.assertEqual(decoded[40], 7)
        for i in range(NUM_V_BUCKETS):
            if i not in (10, 40):
                self.assertEqual(decoded[i], 0)

    # ------------------------------------------------------------------
    # Mixed-schema TFRecord round-trip
    # ------------------------------------------------------------------

    def test_mixed_schema_tfrecord_roundtrip(self):
        """Write a chunk with old- and new-schema records, verify both parse."""
        # New-schema record: pi_aux_dist with a spike at index 42, value dist
        # with 5 counts at bucket 25.
        new_pi_aux = np.zeros(NUM_MOVES, dtype=np.float32)
        new_pi_aux[42] = 1.0
        new_vdist = np.zeros(NUM_V_BUCKETS, dtype=np.uint32)
        new_vdist[25] = 5

        old_rec = self._old_schema_example(score_margin=1.0)
        new_rec = self._new_schema_example(
            pi_aux_dist=new_pi_aux,
            mcts_value_dist=new_vdist,
            score_margin=-1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mixed.tfrecord")
            _write_tfrecord(path, [old_rec, new_rec])

            results = list(tf.data.TFRecordDataset(path).map(transforms.expand))

        self.assertEqual(len(results), 2)

        # Unpack tuples.
        old_r = [t.numpy() for t in results[0]]
        new_r = [t.numpy() for t in results[1]]

        # Old record: flags False, sentinels.
        self.assertFalse(old_r[_IDX_HAS_PI_AUX_DIST])
        self.assertFalse(old_r[_IDX_HAS_MCTS_VALUE_DIST])
        np.testing.assert_array_equal(
            old_r[_IDX_POLICY_AUX_DIST], np.zeros(NUM_MOVES, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            old_r[_IDX_MCTS_VALUE_DIST], np.zeros(NUM_V_BUCKETS, dtype=np.int32)
        )

        # New record: flags True, values present.
        self.assertTrue(new_r[_IDX_HAS_PI_AUX_DIST])
        self.assertTrue(new_r[_IDX_HAS_MCTS_VALUE_DIST])
        # pi_aux_dist[42] should be 1 (symmetry may permute it, but sum is preserved).
        self.assertAlmostEqual(new_r[_IDX_POLICY_AUX_DIST].sum(), 1.0, places=5)
        self.assertEqual(new_r[_IDX_MCTS_VALUE_DIST][25], 5)
        self.assertEqual(new_r[_IDX_MCTS_VALUE_DIST].sum(), 5)


if __name__ == "__main__":
    unittest.main()
