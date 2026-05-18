"""Tests for transforms.py expand function (framework-agnostic)."""

import contextlib
import os
import struct
import tempfile
import unittest

import numpy as np
from tfrecord import example_pb2 as pb
from tfrecord import reader as tfreader
from tfrecord.writer import TFRecordWriter

import symmetry as sym
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


@contextlib.contextmanager
def _deterministic(symmetry=sym.IDENTITY, mask_active=False):
    """Stub the two random sources inside transforms.expand.

    - sym.get_random_symmetry returns the requested symmetry constant.
    - np.random.uniform returns a fixed scalar so the mask_last_moves
      branch (`np.random.uniform() < 0.05`) fires iff `mask_active`.
    """
    saved_sym = sym.get_random_symmetry
    saved_uniform = np.random.uniform
    sym.get_random_symmetry = lambda: symmetry
    np.random.uniform = lambda *args, **kwargs: 0.0 if mask_active else 0.5
    try:
        yield
    finally:
        sym.get_random_symmetry = saved_sym
        np.random.uniform = saved_uniform


def _flat_grid(positions, bsize=BOARD_LEN, dtype=np.int8):
    arr = np.zeros(bsize * bsize, dtype=dtype)
    for r, c, v in positions:
        arr[r * bsize + c] = v
    return arr


def _bytes_feature(value: bytes) -> pb.Feature:
    return pb.Feature(bytes_list=pb.BytesList(value=[value]))


def _float_feature(value: float) -> pb.Feature:
    return pb.Feature(float_list=pb.FloatList(value=[value]))


def _make_example(
    *,
    bsize=BOARD_LEN,
    color=BLACK,
    komi=6.5,
    board=None,
    last_moves=None,
    stones_atari=None,
    stones_two_liberties=None,
    stones_three_liberties=None,
    stones_in_ladder=None,
    own=None,
    pi=None,
    pi_aux_index=0,
    pi_aux_dist=None,
    mcts_value_dist=None,
    score_margin=0.0,
    q6=0.0,
    q16=0.0,
    q50=0.0,
    q6_score=0.0,
    q16_score=0.0,
    q50_score=0.0,
):
    """Build a serialized Example proto. Optional fields omitted when None."""

    def _zg():
        return np.zeros(bsize * bsize, dtype=np.int8)

    if board is None:
        board = _zg()
    if stones_atari is None:
        stones_atari = _zg()
    if stones_two_liberties is None:
        stones_two_liberties = _zg()
    if stones_three_liberties is None:
        stones_three_liberties = _zg()
    if stones_in_ladder is None:
        stones_in_ladder = _zg()
    if own is None:
        own = _zg()
    if last_moves is None:
        last_moves = np.array([-1, -1, -1, -1, -1], dtype=np.int16)
    else:
        last_moves = np.asarray(last_moves, dtype=np.int16)
    if pi is None:
        pi = np.zeros(NUM_MOVES, dtype=np.float32)

    feature = {
        "bsize": _bytes_feature(np.array([bsize], dtype=np.uint8).tobytes()),
        "board": _bytes_feature(board.astype(np.int8).tobytes()),
        "last_moves": _bytes_feature(last_moves.tobytes()),
        "stones_atari": _bytes_feature(stones_atari.astype(np.int8).tobytes()),
        "stones_two_liberties": _bytes_feature(
            stones_two_liberties.astype(np.int8).tobytes()
        ),
        "stones_three_liberties": _bytes_feature(
            stones_three_liberties.astype(np.int8).tobytes()
        ),
        "stones_in_ladder": _bytes_feature(stones_in_ladder.astype(np.int8).tobytes()),
        "color": _bytes_feature(np.array([color], dtype=np.int8).tobytes()),
        "komi": _float_feature(komi),
        "own": _bytes_feature(own.astype(np.int8).tobytes()),
        "pi": _bytes_feature(pi.astype(np.float32).tobytes()),
        "pi_aux": _bytes_feature(np.array([pi_aux_index], dtype=np.int16).tobytes()),
        "score_margin": _float_feature(score_margin),
        "q6": _float_feature(q6),
        "q16": _float_feature(q16),
        "q50": _float_feature(q50),
        "q6_score": _float_feature(q6_score),
        "q16_score": _float_feature(q16_score),
        "q50_score": _float_feature(q50_score),
    }

    if pi_aux_dist is not None:
        feature["pi_aux_dist"] = _bytes_feature(
            pi_aux_dist.astype(np.float32).tobytes()
        )
    if mcts_value_dist is not None:
        feature["mcts_value_dist"] = _bytes_feature(
            mcts_value_dist.astype(np.uint32).tobytes()
        )

    return pb.Example(features=pb.Features(feature=feature)).SerializeToString()


def _write_tfrecord(path, serialized_examples):
    """Write uncompressed tfrecord framing manually."""
    with open(path, "wb") as f:
        for record in serialized_examples:
            length_bytes = struct.pack("<Q", len(record))
            f.write(length_bytes)
            f.write(TFRecordWriter.masked_crc(length_bytes))
            f.write(record)
            f.write(TFRecordWriter.masked_crc(record))


# ======================================================================
# Existing structural tests (kept).
# ======================================================================


class TransformsStructuralTest(unittest.TestCase):

    def _old(self, **kwargs):
        return _make_example(**kwargs)

    def _new(self, **kwargs):
        pi_aux_dist = kwargs.pop("pi_aux_dist", np.zeros(NUM_MOVES, dtype=np.float32))
        mcts_value_dist = kwargs.pop(
            "mcts_value_dist", np.zeros(NUM_V_BUCKETS, dtype=np.uint32)
        )
        return _make_example(
            pi_aux_dist=pi_aux_dist, mcts_value_dist=mcts_value_dist, **kwargs
        )

    def test_expand_output_count(self):
        result = transforms.expand_bytes(self._old())
        self.assertEqual(len(result), _NUM_OUTPUTS)

    def test_expand_input_shape(self):
        result = transforms.expand_bytes(self._old())
        self.assertEqual(result[_IDX_INPUT].shape, (BOARD_LEN, BOARD_LEN, 15))

    def test_expand_global_state_shape(self):
        result = transforms.expand_bytes(self._old())
        self.assertEqual(result[_IDX_INPUT_GLOBAL_STATE].shape, (num_input_features(),))

    def test_expand_policy_shape(self):
        result = transforms.expand_bytes(self._old())
        self.assertEqual(result[_IDX_POLICY].shape, (NUM_MOVES,))

    def test_expand_score_one_hot_shape(self):
        result = transforms.expand_bytes(self._old())
        self.assertEqual(result[_IDX_SCORE_ONE_HOT].shape, (SCORE_RANGE,))

    def test_expand_q_score_shapes(self):
        result = transforms.expand_bytes(self._old())
        for idx in (_IDX_Q6_SCORE, _IDX_Q16_SCORE, _IDX_Q50_SCORE):
            # q-fields are now Python floats (scalars); shape concept maps to ().
            self.assertTrue(np.isscalar(result[idx]))

    def test_old_schema_has_flags_false(self):
        result = transforms.expand_bytes(self._old())
        self.assertFalse(result[_IDX_HAS_PI_AUX_DIST])
        self.assertFalse(result[_IDX_HAS_MCTS_VALUE_DIST])

    def test_old_schema_policy_aux_dist_is_zeros(self):
        result = transforms.expand_bytes(self._old())
        dist = result[_IDX_POLICY_AUX_DIST]
        self.assertEqual(dist.shape, (NUM_MOVES,))
        np.testing.assert_array_equal(dist, np.zeros(NUM_MOVES, dtype=np.float32))

    def test_old_schema_mcts_value_dist_is_zeros(self):
        result = transforms.expand_bytes(self._old())
        dist = result[_IDX_MCTS_VALUE_DIST]
        self.assertEqual(dist.shape, (NUM_V_BUCKETS,))
        np.testing.assert_array_equal(dist, np.zeros(NUM_V_BUCKETS, dtype=np.int32))

    def test_new_schema_has_flags_true(self):
        result = transforms.expand_bytes(self._new())
        self.assertTrue(result[_IDX_HAS_PI_AUX_DIST])
        self.assertTrue(result[_IDX_HAS_MCTS_VALUE_DIST])

    def test_new_schema_policy_aux_dist_shape(self):
        result = transforms.expand_bytes(self._new())
        self.assertEqual(result[_IDX_POLICY_AUX_DIST].shape, (NUM_MOVES,))

    def test_new_schema_mcts_value_dist_values(self):
        dist = np.zeros(NUM_V_BUCKETS, dtype=np.uint32)
        dist[10] = 3
        dist[40] = 7
        result = transforms.expand_bytes(self._new(mcts_value_dist=dist))
        decoded = result[_IDX_MCTS_VALUE_DIST]
        self.assertEqual(decoded[10], 3)
        self.assertEqual(decoded[40], 7)
        for i in range(NUM_V_BUCKETS):
            if i not in (10, 40):
                self.assertEqual(decoded[i], 0)

    def test_mixed_schema_tfrecord_roundtrip(self):
        new_pi_aux = np.zeros(NUM_MOVES, dtype=np.float32)
        new_pi_aux[42] = 1.0
        new_vdist = np.zeros(NUM_V_BUCKETS, dtype=np.uint32)
        new_vdist[25] = 5

        old_rec = _make_example(score_margin=1.0)
        new_rec = _make_example(
            pi_aux_dist=new_pi_aux,
            mcts_value_dist=new_vdist,
            score_margin=-1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mixed.tfrecord")
            _write_tfrecord(path, [old_rec, new_rec])
            results = [
                transforms.expand_bytes(bytes(view))
                for view in tfreader.tfrecord_iterator(path, compression_type=None)
            ]

        self.assertEqual(len(results), 2)
        old_r = results[0]
        new_r = results[1]

        self.assertFalse(old_r[_IDX_HAS_PI_AUX_DIST])
        self.assertFalse(old_r[_IDX_HAS_MCTS_VALUE_DIST])
        np.testing.assert_array_equal(
            old_r[_IDX_POLICY_AUX_DIST], np.zeros(NUM_MOVES, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            old_r[_IDX_MCTS_VALUE_DIST], np.zeros(NUM_V_BUCKETS, dtype=np.int32)
        )

        self.assertTrue(new_r[_IDX_HAS_PI_AUX_DIST])
        self.assertTrue(new_r[_IDX_HAS_MCTS_VALUE_DIST])
        self.assertAlmostEqual(new_r[_IDX_POLICY_AUX_DIST].sum(), 1.0, places=5)
        self.assertEqual(new_r[_IDX_MCTS_VALUE_DIST][25], 5)
        self.assertEqual(new_r[_IDX_MCTS_VALUE_DIST].sum(), 5)


# ======================================================================
# Content correctness tests (deterministic symmetry).
# ======================================================================


class TransformsBlackPathTest(unittest.TestCase):

    def test_get_color_extracts_black_and_white_planes(self):
        board = _flat_grid([(3, 5, BLACK), (7, 11, WHITE)])
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(color=BLACK, board=board))
        planes = r[_IDX_INPUT]
        self.assertEqual(planes[3, 5, 0], 1.0)
        self.assertEqual(planes[7, 11, 0], 0.0)
        self.assertEqual(planes[3, 5, 1], 0.0)
        self.assertEqual(planes[7, 11, 1], 1.0)

    def test_own_unchanged_for_black(self):
        own = _flat_grid([(3, 5, 1), (7, 11, -1)])
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(color=BLACK, own=own))
        own_out = r[_IDX_OWN]
        self.assertEqual(own_out[3, 5], 1)
        self.assertEqual(own_out[7, 11], -1)

    def test_komi_passthrough(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(komi=7.5))
        self.assertAlmostEqual(float(r[_IDX_KOMI]), 7.5, places=5)


class TransformsWhitePathTest(unittest.TestCase):

    def test_our_stones_are_white_when_color_white(self):
        board = _flat_grid([(3, 5, BLACK), (7, 11, WHITE)])
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(color=WHITE, board=board))
        planes = r[_IDX_INPUT]
        self.assertEqual(planes[7, 11, 0], 1.0)
        self.assertEqual(planes[3, 5, 0], 0.0)
        self.assertEqual(planes[3, 5, 1], 1.0)
        self.assertEqual(planes[7, 11, 1], 0.0)

    def test_own_negated_for_white(self):
        own = _flat_grid([(3, 5, 1), (7, 11, -1)])
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(color=WHITE, own=own))
        own_out = r[_IDX_OWN]
        self.assertEqual(own_out[3, 5], -1)
        self.assertEqual(own_out[7, 11], 1)


class TransformsLastMovesTest(unittest.TestCase):

    def test_last_moves_one_hot_planes(self):
        last_moves = np.array([62, -1, PASS_MOVE_ENCODING, 0, 360], dtype=np.int16)
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(last_moves=last_moves))
        planes = r[_IDX_INPUT]
        self.assertEqual(planes[3, 5, 2], 1.0)
        self.assertEqual(planes[:, :, 2].sum(), 1.0)
        self.assertTrue(np.all(planes[:, :, 3] == 0.0))  # NON_MOVE
        self.assertTrue(np.all(planes[:, :, 4] == 0.0))  # PASS
        self.assertEqual(planes[0, 0, 5], 1.0)
        self.assertEqual(planes[18, 18, 6], 1.0)

    def test_mask_active_zeros_last_moves(self):
        last_moves = np.array([0, 1, 2, 3, 4], dtype=np.int16)
        with _deterministic(sym.IDENTITY, mask_active=True):
            r = transforms.expand_bytes(_make_example(last_moves=last_moves))
        planes = r[_IDX_INPUT]
        for i in range(2, 7):
            self.assertTrue(np.all(planes[:, :, i] == 0.0), f"plane {i} not zeroed")


class TransformsSupportPlanesTest(unittest.TestCase):

    def test_atari_two_three_lib_in_ladder_planes(self):
        atari = _flat_grid([(2, 4, BLACK), (5, 9, WHITE)])
        twolib = _flat_grid([(0, 0, BLACK)])
        threelib = _flat_grid([(0, 1, WHITE)])
        ladder = _flat_grid([(18, 18, BLACK)])
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(
                _make_example(
                    color=BLACK,
                    stones_atari=atari,
                    stones_two_liberties=twolib,
                    stones_three_liberties=threelib,
                    stones_in_ladder=ladder,
                )
            )
        planes = r[_IDX_INPUT]
        self.assertEqual(planes[2, 4, 7], 1.0)
        self.assertEqual(planes[5, 9, 8], 1.0)
        self.assertEqual(planes[0, 0, 9], 1.0)
        self.assertEqual(planes[0, 1, 12], 1.0)
        self.assertEqual(planes[18, 18, 13], 1.0)


class TransformsScoreOneHotTest(unittest.TestCase):

    def test_score_zero(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(score_margin=0.0))
        oh = r[_IDX_SCORE_ONE_HOT]
        self.assertEqual(oh[SCORE_RANGE_MIDPOINT], 1.0)
        self.assertEqual(oh.sum(), 1.0)

    def test_score_positive_floor(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(score_margin=5.7))
        oh = r[_IDX_SCORE_ONE_HOT]
        self.assertEqual(oh[SCORE_RANGE_MIDPOINT + 5], 1.0)

    def test_score_negative_floor(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(score_margin=-3.2))
        oh = r[_IDX_SCORE_ONE_HOT]
        self.assertEqual(oh[SCORE_RANGE_MIDPOINT - 4], 1.0)

    def test_score_clamp_high(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(score_margin=500.0))
        oh = r[_IDX_SCORE_ONE_HOT]
        self.assertEqual(oh[SCORE_RANGE - 1], 1.0)

    def test_score_clamp_low(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(score_margin=-500.0))
        oh = r[_IDX_SCORE_ONE_HOT]
        self.assertEqual(oh[0], 1.0)


class TransformsGameOutcomeTest(unittest.TestCase):

    def test_score_positive_is_win(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(score_margin=10.0))
        np.testing.assert_array_equal(r[_IDX_GAME_OUTCOME], [0.0, 1.0])

    def test_score_negative_is_loss(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(score_margin=-10.0))
        np.testing.assert_array_equal(r[_IDX_GAME_OUTCOME], [1.0, 0.0])

    def test_score_zero_is_draw(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(score_margin=0.0))
        np.testing.assert_array_equal(r[_IDX_GAME_OUTCOME], [0.5, 0.5])


class TransformsGlobalStateTest(unittest.TestCase):

    def test_black_color_indicator_and_komi_sign(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(color=BLACK, komi=7.5))
        gs = r[_IDX_INPUT_GLOBAL_STATE]
        self.assertEqual(gs[0], 1.0)
        self.assertEqual(gs[1], 0.0)
        self.assertAlmostEqual(gs[7], -7.5 / 15.0, places=5)

    def test_white_color_indicator_and_komi_sign(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(color=WHITE, komi=7.5))
        gs = r[_IDX_INPUT_GLOBAL_STATE]
        self.assertEqual(gs[0], 0.0)
        self.assertEqual(gs[1], 1.0)
        self.assertAlmostEqual(gs[7], 7.5 / 15.0, places=5)

    def test_pass_detection(self):
        last_moves = np.array([PASS_MOVE_ENCODING, 0, -1, -1, -1], dtype=np.int16)
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(last_moves=last_moves))
        gs = r[_IDX_INPUT_GLOBAL_STATE]
        self.assertEqual(gs[2], 1.0)  # move 0 was pass
        for i in range(3, 7):
            self.assertEqual(gs[i], 0.0)


class TransformsSymmetryTest(unittest.TestCase):

    def test_identity_preserves_board(self):
        board = _flat_grid([(3, 5, BLACK)])
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(_make_example(color=BLACK, board=board))
        planes = r[_IDX_INPUT]
        self.assertEqual(planes[3, 5, 0], 1.0)
        self.assertEqual(planes[:, :, 0].sum(), 1.0)

    def test_rot90_rotates_board(self):
        # rotate_loc((1, 2), k=1, n=19) = [2, 17]; expect stone at (2, 17)
        board = _flat_grid([(1, 2, BLACK)])
        with _deterministic(sym.ROT90):
            r = transforms.expand_bytes(_make_example(color=BLACK, board=board))
        planes = r[_IDX_INPUT]
        self.assertEqual(planes[2, BOARD_LEN - 2, 0], 1.0)
        self.assertEqual(planes[:, :, 0].sum(), 1.0)

    def test_policy_spike_under_rot90(self):
        # Spike at (2, 4) → flat 42; under ROT90 → (4, 16) → flat 4*19+16 = 92.
        pi = np.zeros(NUM_MOVES, dtype=np.float32)
        pi[2 * BOARD_LEN + 4] = 1.0
        with _deterministic(sym.ROT90):
            r = transforms.expand_bytes(_make_example(pi=pi))
        policy = r[_IDX_POLICY]
        self.assertAlmostEqual(policy[4 * BOARD_LEN + 16], 1.0, places=5)
        self.assertEqual(policy[NUM_MOVES - 1], 0.0)
        self.assertAlmostEqual(policy.sum(), 1.0, places=5)

    def test_pass_slot_preserved_under_rot90(self):
        pi = np.zeros(NUM_MOVES, dtype=np.float32)
        pi[NUM_MOVES - 1] = 0.7
        pi[2 * BOARD_LEN + 4] = 0.3
        with _deterministic(sym.ROT90):
            r = transforms.expand_bytes(_make_example(pi=pi))
        policy = r[_IDX_POLICY]
        self.assertAlmostEqual(policy[NUM_MOVES - 1], 0.7, places=5)
        self.assertAlmostEqual(policy[4 * BOARD_LEN + 16], 0.3, places=5)


class TransformsPolicyAuxTest(unittest.TestCase):

    def test_pi_aux_pass_unchanged_under_rot90(self):
        with _deterministic(sym.ROT90):
            r = transforms.expand_bytes(_make_example(pi_aux_index=PASS_MOVE_ENCODING))
        self.assertEqual(int(r[_IDX_POLICY_AUX]), PASS_MOVE_ENCODING)

    def test_pi_aux_board_move_under_rot90(self):
        with _deterministic(sym.ROT90):
            r = transforms.expand_bytes(_make_example(pi_aux_index=42))
        self.assertEqual(int(r[_IDX_POLICY_AUX]), 4 * BOARD_LEN + 16)


class TransformsPiAuxDistTest(unittest.TestCase):

    def test_pi_aux_dist_spike_under_rot90(self):
        d = np.zeros(NUM_MOVES, dtype=np.float32)
        d[2 * BOARD_LEN + 4] = 1.0
        with _deterministic(sym.ROT90):
            r = transforms.expand_bytes(
                _make_example(
                    pi_aux_dist=d,
                    mcts_value_dist=np.zeros(NUM_V_BUCKETS, dtype=np.uint32),
                )
            )
        out = r[_IDX_POLICY_AUX_DIST]
        self.assertAlmostEqual(out[4 * BOARD_LEN + 16], 1.0, places=5)
        self.assertEqual(out[NUM_MOVES - 1], 0.0)
        self.assertAlmostEqual(out.sum(), 1.0, places=5)


class TransformsQPassThroughTest(unittest.TestCase):

    def test_q_values_passthrough(self):
        with _deterministic(sym.IDENTITY):
            r = transforms.expand_bytes(
                _make_example(
                    q6=0.1,
                    q16=0.2,
                    q50=0.3,
                    q6_score=1.5,
                    q16_score=2.5,
                    q50_score=3.5,
                )
            )
        self.assertAlmostEqual(float(r[_IDX_Q6]), 0.1, places=5)
        self.assertAlmostEqual(float(r[_IDX_Q16]), 0.2, places=5)
        self.assertAlmostEqual(float(r[_IDX_Q50]), 0.3, places=5)
        self.assertAlmostEqual(float(r[_IDX_Q6_SCORE]), 1.5, places=5)
        self.assertAlmostEqual(float(r[_IDX_Q16_SCORE]), 2.5, places=5)
        self.assertAlmostEqual(float(r[_IDX_Q50_SCORE]), 3.5, places=5)


if __name__ == "__main__":
    unittest.main()
