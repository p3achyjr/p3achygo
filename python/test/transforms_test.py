"""Tests for transforms.py expand function."""

import unittest
import tensorflow as tf
import numpy as np

import transforms
from constants import *


class TransformsTest(unittest.TestCase):

    def _create_mock_example(self, use_valid_last_moves=False):
        """Create a mock tfrecord example."""
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

    def test_expand_output_count(self):
        """Test that expand returns 15 outputs."""
        example = self._create_mock_example(use_valid_last_moves=False)
        result = transforms.expand(example)
        self.assertEqual(len(result), 15)

    def test_expand_input_shape(self):
        """Test that expand returns input with 15 planes (13 + 2 ladder)."""
        example = self._create_mock_example(use_valid_last_moves=False)
        result = transforms.expand(example)
        input_tensor = result[0]
        # Shape should be (BOARD_LEN, BOARD_LEN, 15)
        self.assertEqual(input_tensor.shape, (BOARD_LEN, BOARD_LEN, 15))

    def test_expand_global_state_shape(self):
        """Test that global state has shape (7,)."""
        example = self._create_mock_example(use_valid_last_moves=False)
        result = transforms.expand(example)
        global_state = result[1]
        self.assertEqual(global_state.shape, (NUM_INPUT_FEATURES,))

    def test_expand_policy_shape(self):
        """Test that policy has shape (362,)."""
        example = self._create_mock_example(use_valid_last_moves=False)
        result = transforms.expand(example)
        policy = result[6]
        self.assertEqual(policy.shape, (BOARD_LEN * BOARD_LEN + 1,))

    def test_expand_score_one_hot_shape(self):
        """Test that score_one_hot has shape (SCORE_RANGE,)."""
        example = self._create_mock_example(use_valid_last_moves=False)
        result = transforms.expand(example)
        score_one_hot = result[5]
        self.assertEqual(score_one_hot.shape, (SCORE_RANGE,))

    def test_expand_has_q_score_outputs(self):
        """Test that expand includes q6_score, q16_score, q50_score."""
        example = self._create_mock_example(use_valid_last_moves=False)
        result = transforms.expand(example)
        # Last 3 outputs should be q_score values
        q6_score = result[12]
        q16_score = result[13]
        q50_score = result[14]
        # They should be scalars
        self.assertEqual(q6_score.shape, ())
        self.assertEqual(q16_score.shape, ())
        self.assertEqual(q50_score.shape, ())

    def test_parse_example(self):
        """Test _parse_example returns correct keys."""
        example = self._create_mock_example()
        parsed = transforms._parse_example(example)

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
