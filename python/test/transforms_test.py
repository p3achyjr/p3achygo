"""Tests for transforms.py v0/v1 expand functions."""

import unittest
import tensorflow as tf
import numpy as np

import transforms
from constants import *


class TransformsTest(unittest.TestCase):

    def _create_mock_v0_example(self, use_valid_last_moves=False):
        """Create a mock v0 tfrecord example."""
        bsize = BOARD_LEN

        # Use valid board positions for last_moves when needed for full expand tests
        # -1 means "no move" but causes issues with symmetry transforms in eager mode
        if use_valid_last_moves:
            # Use center position (9,9) which transforms safely under all symmetries
            last_moves_data = np.array([180, 180, 180, 180, 180], dtype=np.int16)  # 9*19+9 = 180
        else:
            last_moves_data = np.array([-1, -1, -1, -1, -1], dtype=np.int16)

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
                bytes_list=tf.train.BytesList(
                    value=[last_moves_data.tobytes()]
                )
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
                    value=[np.array([0], dtype=np.int16).tobytes()]
                )
            ),
            "score_margin": tf.train.Feature(
                float_list=tf.train.FloatList(value=[0.0])
            ),
            "q30": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
            "q100": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
            "q200": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def _create_mock_v1_example(self, use_valid_last_moves=False):
        """Create a mock v1 tfrecord example."""
        bsize = BOARD_LEN

        # Use valid board positions for last_moves when needed for full expand tests
        if use_valid_last_moves:
            last_moves_data = np.array([180, 180, 180, 180, 180], dtype=np.int16)  # 9*19+9 = 180
        else:
            last_moves_data = np.array([-1, -1, -1, -1, -1], dtype=np.int16)

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
                bytes_list=tf.train.BytesList(
                    value=[last_moves_data.tobytes()]
                )
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
                    value=[np.array([0], dtype=np.int16).tobytes()]
                )
            ),
            "score_margin": tf.train.Feature(
                float_list=tf.train.FloatList(value=[0.0])
            ),
            "q6": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
            "q16": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
            "q50": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
            "q6_score": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
            "q16_score": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
            "q50_score": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def test_expand_v0_output_count(self):
        """Test that expand_v0 returns 12 outputs."""
        example = self._create_mock_v0_example(use_valid_last_moves=False)
        result = transforms.expand_v0(example)
        self.assertEqual(len(result), 12)

    def test_expand_v1_output_count(self):
        """Test that expand_v1 returns 15 outputs (12 + 3 score predictions)."""
        example = self._create_mock_v1_example(use_valid_last_moves=False)
        result = transforms.expand_v1(example)
        self.assertEqual(len(result), 15)

    def test_expand_v0_input_shape(self):
        """Test that expand_v0 returns input with 13 planes."""
        example = self._create_mock_v0_example(use_valid_last_moves=False)
        result = transforms.expand_v0(example)
        input_tensor = result[0]
        # Shape should be (BOARD_LEN, BOARD_LEN, 13)
        self.assertEqual(input_tensor.shape, (BOARD_LEN, BOARD_LEN, 13))

    def test_expand_v1_input_shape(self):
        """Test that expand_v1 returns input with 15 planes (13 + 2 ladder)."""
        example = self._create_mock_v1_example(use_valid_last_moves=False)
        result = transforms.expand_v1(example)
        input_tensor = result[0]
        # Shape should be (BOARD_LEN, BOARD_LEN, 15)
        self.assertEqual(input_tensor.shape, (BOARD_LEN, BOARD_LEN, 15))

    def test_expand_v0_global_state_shape(self):
        """Test that global state has shape (7,)."""
        example = self._create_mock_v0_example(use_valid_last_moves=False)
        result = transforms.expand_v0(example)
        global_state = result[1]
        self.assertEqual(global_state.shape, (NUM_INPUT_FEATURES,))

    def test_expand_v1_global_state_shape(self):
        """Test that v1 global state has shape (7,)."""
        example = self._create_mock_v1_example(use_valid_last_moves=False)
        result = transforms.expand_v1(example)
        global_state = result[1]
        self.assertEqual(global_state.shape, (NUM_INPUT_FEATURES,))

    def test_expand_v0_policy_shape(self):
        """Test that policy has shape (362,)."""
        example = self._create_mock_v0_example(use_valid_last_moves=False)
        result = transforms.expand_v0(example)
        policy = result[6]
        self.assertEqual(policy.shape, (BOARD_LEN * BOARD_LEN + 1,))

    def test_expand_v1_policy_shape(self):
        """Test that v1 policy has shape (362,)."""
        example = self._create_mock_v1_example(use_valid_last_moves=False)
        result = transforms.expand_v1(example)
        policy = result[6]
        self.assertEqual(policy.shape, (BOARD_LEN * BOARD_LEN + 1,))

    def test_expand_v0_score_one_hot_shape(self):
        """Test that score_one_hot has shape (SCORE_RANGE,)."""
        example = self._create_mock_v0_example(use_valid_last_moves=False)
        result = transforms.expand_v0(example)
        score_one_hot = result[5]
        self.assertEqual(score_one_hot.shape, (SCORE_RANGE,))

    def test_expand_v1_has_q_score_outputs(self):
        """Test that expand_v1 includes q6_score, q16_score, q50_score."""
        example = self._create_mock_v1_example(use_valid_last_moves=False)
        result = transforms.expand_v1(example)
        # Last 3 outputs should be q_score values
        q6_score = result[12]
        q16_score = result[13]
        q50_score = result[14]
        # They should be scalars
        self.assertEqual(q6_score.shape, ())
        self.assertEqual(q16_score.shape, ())
        self.assertEqual(q50_score.shape, ())

    def test_parse_example_v0(self):
        """Test _parse_example returns correct keys for v0."""
        example = self._create_mock_v0_example()
        parsed = transforms._parse_example(example, version=0)

        expected_keys = {
            "bsize",
            "board",
            "last_moves",
            "stones_atari",
            "stones_two_liberties",
            "stones_three_liberties",
            "color",
            "komi",
            "own",
            "policy",
            "policy_aux",
            "score",
            "q6",
            "q16",
            "q50",
        }
        self.assertEqual(set(parsed.keys()), expected_keys)

    def test_parse_example_v1(self):
        """Test _parse_example returns correct keys for v1."""
        example = self._create_mock_v1_example()
        parsed = transforms._parse_example(example, version=1)

        expected_keys = {
            "bsize",
            "board",
            "last_moves",
            "stones_atari",
            "stones_two_liberties",
            "stones_three_liberties",
            "stones_in_ladder",
            "color",
            "komi",
            "own",
            "policy",
            "policy_aux",
            "score",
            "q6",
            "q16",
            "q50",
            "q6_score",
            "q16_score",
            "q50_score",
        }
        self.assertEqual(set(parsed.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
