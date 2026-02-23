"""Tests for model.py v0/v1 model shapes and forward pass."""

import unittest
import tensorflow as tf
import numpy as np

import sys
sys.path.insert(0, '/app/python')

from model import P3achyGoModel
from model_config import ModelConfig
from constants import *


class ModelV1Test(unittest.TestCase):

    def _create_tiny_v0_model(self):
        """Create a tiny v0 model for testing."""
        config = ModelConfig.tiny()
        config.kVersion = 0
        return P3achyGoModel.create(
            config,
            board_len=BOARD_LEN,
            num_input_planes=NUM_INPUT_PLANES,
            num_input_features=NUM_INPUT_FEATURES,
            name="tiny_v0",
        )

    def _create_tiny_v1_model(self):
        """Create a tiny v1 model for testing."""
        config = ModelConfig.tiny()
        config.kVersion = 1
        return P3achyGoModel.create(
            config,
            board_len=BOARD_LEN,
            num_input_planes=NUM_INPUT_PLANES + 2,  # v1 has 2 extra ladder planes
            num_input_features=NUM_INPUT_FEATURES,
            name="tiny_v1",
        )

    def _create_mock_input(self, num_planes):
        """Create mock input tensors for testing."""
        batch_size = 2
        board_state = tf.random.uniform(
            (batch_size, BOARD_LEN, BOARD_LEN, num_planes),
            dtype=tf.float32
        )
        game_state = tf.random.uniform(
            (batch_size, NUM_INPUT_FEATURES),
            dtype=tf.float32
        )
        return board_state, game_state

    def test_v0_model_version(self):
        """Test that v0 model has version 0."""
        model = self._create_tiny_v0_model()
        self.assertEqual(model.version, 0)

    def test_v1_model_version(self):
        """Test that v1 model has version 1."""
        model = self._create_tiny_v1_model()
        self.assertEqual(model.version, 1)

    def test_v0_model_input_planes(self):
        """Test that v0 model expects 13 input planes."""
        model = self._create_tiny_v0_model()
        self.assertEqual(model.num_input_planes, NUM_INPUT_PLANES)

    def test_v1_model_input_planes(self):
        """Test that v1 model expects 15 input planes."""
        model = self._create_tiny_v1_model()
        self.assertEqual(model.num_input_planes, NUM_INPUT_PLANES + 2)

    def test_v0_forward_pass_output_count(self):
        """Test that v0 model forward pass returns 12 outputs."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        outputs = model(board_state, game_state, training=False)
        self.assertEqual(len(outputs), 12)

    def test_v1_forward_pass_output_count(self):
        """Test that v1 model forward pass returns 23 outputs."""
        model = self._create_tiny_v1_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES + 2)
        outputs = model(board_state, game_state, training=False)
        # v1 returns: 12 base + 3 q_err + 3 q_score + 3 q_score_err + 2 soft/optimistic = 23
        self.assertEqual(len(outputs), 23)

    def test_v0_policy_shape(self):
        """Test that v0 policy output has correct shape."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        outputs = model(board_state, game_state, training=False)

        pi_logits = outputs[0]
        pi_probs = outputs[1]

        expected_policy_size = BOARD_LEN * BOARD_LEN + 1  # 362
        self.assertEqual(pi_logits.shape, (2, expected_policy_size))
        self.assertEqual(pi_probs.shape, (2, expected_policy_size))

    def test_v1_policy_shape(self):
        """Test that v1 policy output has correct shape."""
        model = self._create_tiny_v1_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES + 2)
        outputs = model(board_state, game_state, training=False)

        pi_logits = outputs[0]
        pi_probs = outputs[1]

        expected_policy_size = BOARD_LEN * BOARD_LEN + 1  # 362
        self.assertEqual(pi_logits.shape, (2, expected_policy_size))
        self.assertEqual(pi_probs.shape, (2, expected_policy_size))

    def test_v0_outcome_shape(self):
        """Test that v0 outcome output has correct shape."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        outputs = model(board_state, game_state, training=False)

        outcome_logits = outputs[2]
        outcome_probs = outputs[3]

        self.assertEqual(outcome_logits.shape, (2, 2))  # win/loss
        self.assertEqual(outcome_probs.shape, (2, 2))

    def test_v0_ownership_shape(self):
        """Test that v0 ownership output has correct shape."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        outputs = model(board_state, game_state, training=False)

        ownership = outputs[4]

        self.assertEqual(ownership.shape, (2, BOARD_LEN, BOARD_LEN, 1))

    def test_v0_score_shape(self):
        """Test that v0 score output has correct shape."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        outputs = model(board_state, game_state, training=False)

        score_logits = outputs[5]
        score_probs = outputs[6]

        self.assertEqual(score_logits.shape, (2, SCORE_RANGE))
        self.assertEqual(score_probs.shape, (2, SCORE_RANGE))

    def test_v0_q_values_shape(self):
        """Test that v0 Q-value outputs have correct shape."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        outputs = model(board_state, game_state, training=False)

        q6 = outputs[9]
        q16 = outputs[10]
        q50 = outputs[11]

        self.assertEqual(q6.shape, (2,))
        self.assertEqual(q16.shape, (2,))
        self.assertEqual(q50.shape, (2,))

    def test_v1_q_error_shape(self):
        """Test that v1 Q-error outputs have correct shape."""
        model = self._create_tiny_v1_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES + 2)
        outputs = model(board_state, game_state, training=False)

        q6_err = outputs[12]
        q16_err = outputs[13]
        q50_err = outputs[14]

        self.assertEqual(q6_err.shape, (2,))
        self.assertEqual(q16_err.shape, (2,))
        self.assertEqual(q50_err.shape, (2,))

    def test_v1_q_score_shape(self):
        """Test that v1 Q-score outputs have correct shape."""
        model = self._create_tiny_v1_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES + 2)
        outputs = model(board_state, game_state, training=False)

        q6_score = outputs[15]
        q16_score = outputs[16]
        q50_score = outputs[17]

        self.assertEqual(q6_score.shape, (2,))
        self.assertEqual(q16_score.shape, (2,))
        self.assertEqual(q50_score.shape, (2,))

    def test_v1_soft_optimistic_policy_shape(self):
        """Test that v1 soft/optimistic policy outputs have correct shape."""
        model = self._create_tiny_v1_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES + 2)
        outputs = model(board_state, game_state, training=False)

        pi_soft = outputs[21]
        pi_optimistic = outputs[22]

        expected_policy_size = BOARD_LEN * BOARD_LEN + 1  # 362
        self.assertEqual(pi_soft.shape, (2, expected_policy_size))
        self.assertEqual(pi_optimistic.shape, (2, expected_policy_size))

    def test_v0_training_mode(self):
        """Test that v0 model runs in training mode."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        # Should not raise any errors
        outputs = model(board_state, game_state, training=True)
        self.assertEqual(len(outputs), 12)

    def test_v1_training_mode(self):
        """Test that v1 model runs in training mode."""
        model = self._create_tiny_v1_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES + 2)
        # Should not raise any errors
        outputs = model(board_state, game_state, training=True)
        self.assertEqual(len(outputs), 23)

    def test_v0_policy_sums_to_one(self):
        """Test that v0 policy probabilities sum to 1."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        outputs = model(board_state, game_state, training=False)

        pi_probs = outputs[1]
        sums = tf.reduce_sum(pi_probs, axis=1)

        np.testing.assert_allclose(sums.numpy(), [1.0, 1.0], rtol=1e-5)

    def test_v1_policy_sums_to_one(self):
        """Test that v1 policy probabilities sum to 1."""
        model = self._create_tiny_v1_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES + 2)
        outputs = model(board_state, game_state, training=False)

        pi_probs = outputs[1]
        sums = tf.reduce_sum(pi_probs, axis=1)

        np.testing.assert_allclose(sums.numpy(), [1.0, 1.0], rtol=1e-5)

    def test_v0_outcome_probs_sum_to_one(self):
        """Test that v0 outcome probabilities sum to 1."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        outputs = model(board_state, game_state, training=False)

        outcome_probs = outputs[3]
        sums = tf.reduce_sum(outcome_probs, axis=1)

        np.testing.assert_allclose(sums.numpy(), [1.0, 1.0], rtol=1e-5)

    def test_v0_q_values_in_range(self):
        """Test that v0 Q-values are in [-1, 1] range (tanh output)."""
        model = self._create_tiny_v0_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES)
        outputs = model(board_state, game_state, training=False)

        q6 = outputs[9].numpy()
        q16 = outputs[10].numpy()
        q50 = outputs[11].numpy()

        self.assertTrue(np.all(q6 >= -1) and np.all(q6 <= 1))
        self.assertTrue(np.all(q16 >= -1) and np.all(q16 <= 1))
        self.assertTrue(np.all(q50 >= -1) and np.all(q50 <= 1))

    def test_v1_q_values_in_range(self):
        """Test that v1 Q-values are in [-1, 1] range (tanh output)."""
        model = self._create_tiny_v1_model()
        board_state, game_state = self._create_mock_input(NUM_INPUT_PLANES + 2)
        outputs = model(board_state, game_state, training=False)

        q6 = outputs[9].numpy()
        q16 = outputs[10].numpy()
        q50 = outputs[11].numpy()

        self.assertTrue(np.all(q6 >= -1) and np.all(q6 <= 1))
        self.assertTrue(np.all(q16 >= -1) and np.all(q16 <= 1))
        self.assertTrue(np.all(q50 >= -1) and np.all(q50 <= 1))


if __name__ == "__main__":
    unittest.main()
