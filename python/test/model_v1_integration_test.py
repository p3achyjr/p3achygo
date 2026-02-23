"""Integration tests for model.py v0/v1 training step."""

import unittest
import tensorflow as tf
import numpy as np

import sys
sys.path.insert(0, '/app/python')

from model import P3achyGoModel
from model_config import ModelConfig
from constants import *


def compute_loss(model, **kwargs):
    """
    Call the model's compute_losses method.

    Note: In Keras 3, keras.Model has a 'loss' property that would shadow
    a method named 'loss', so we renamed our method to 'compute_losses'.
    """
    return model.compute_losses(**kwargs)


class ModelV1IntegrationTest(unittest.TestCase):

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

    def _create_mock_v0_batch(self, batch_size=4):
        """Create mock training batch for v0."""
        board_state = tf.random.uniform(
            (batch_size, BOARD_LEN, BOARD_LEN, NUM_INPUT_PLANES),
            dtype=tf.float32
        )
        game_state = tf.random.uniform(
            (batch_size, NUM_INPUT_FEATURES),
            dtype=tf.float32
        )

        # Ground truth labels
        policy = tf.nn.softmax(tf.random.uniform((batch_size, BOARD_LEN * BOARD_LEN + 1)))
        policy_aux = tf.random.uniform((batch_size,), minval=0, maxval=361, dtype=tf.int32)
        score = tf.random.uniform((batch_size,), minval=-50, maxval=50, dtype=tf.float32)
        score_one_hot = tf.one_hot(
            tf.cast(score + SCORE_RANGE_MIDPOINT, tf.int32),
            depth=SCORE_RANGE
        )
        own = tf.random.uniform((batch_size, BOARD_LEN, BOARD_LEN), minval=-1, maxval=1)
        q6 = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        q16 = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        q50 = tf.random.uniform((batch_size,), minval=-1, maxval=1)

        return {
            "board_state": board_state,
            "game_state": game_state,
            "policy": policy,
            "policy_aux": policy_aux,
            "score": score,
            "score_one_hot": score_one_hot,
            "own": own,
            "q6": q6,
            "q16": q16,
            "q50": q50,
        }

    def _create_mock_v1_batch(self, batch_size=4):
        """Create mock training batch for v1."""
        board_state = tf.random.uniform(
            (batch_size, BOARD_LEN, BOARD_LEN, NUM_INPUT_PLANES + 2),
            dtype=tf.float32
        )
        game_state = tf.random.uniform(
            (batch_size, NUM_INPUT_FEATURES),
            dtype=tf.float32
        )

        # Ground truth labels
        policy = tf.nn.softmax(tf.random.uniform((batch_size, BOARD_LEN * BOARD_LEN + 1)))
        policy_aux = tf.random.uniform((batch_size,), minval=0, maxval=361, dtype=tf.int32)
        score = tf.random.uniform((batch_size,), minval=-50, maxval=50, dtype=tf.float32)
        score_one_hot = tf.one_hot(
            tf.cast(score + SCORE_RANGE_MIDPOINT, tf.int32),
            depth=SCORE_RANGE
        )
        own = tf.random.uniform((batch_size, BOARD_LEN, BOARD_LEN), minval=-1, maxval=1)
        q6 = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        q16 = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        q50 = tf.random.uniform((batch_size,), minval=-1, maxval=1)

        # v1-specific labels
        q6_score = tf.random.uniform((batch_size,), minval=-50, maxval=50)
        q16_score = tf.random.uniform((batch_size,), minval=-50, maxval=50)
        q50_score = tf.random.uniform((batch_size,), minval=-50, maxval=50)

        return {
            "board_state": board_state,
            "game_state": game_state,
            "policy": policy,
            "policy_aux": policy_aux,
            "score": score,
            "score_one_hot": score_one_hot,
            "own": own,
            "q6": q6,
            "q16": q16,
            "q50": q50,
            "q6_score": q6_score,
            "q16_score": q16_score,
            "q50_score": q50_score,
        }

    def test_v0_loss_computation(self):
        """Test that v0 loss computation works correctly."""
        model = self._create_tiny_v0_model()
        batch = self._create_mock_v0_batch()

        outputs = model(batch["board_state"], batch["game_state"], training=True)

        loss_result = compute_loss(model,
            pi_logits=outputs[0],
            pi_logits_aux=outputs[8],
            game_outcome=outputs[2],
            score_logits=outputs[5],
            own_pred=outputs[4],
            q6_pred=outputs[9],
            q16_pred=outputs[10],
            q50_pred=outputs[11],
            gamma=outputs[7],
            policy=batch["policy"],
            policy_aux=batch["policy_aux"],
            score=batch["score"],
            score_one_hot=batch["score_one_hot"],
            own=batch["own"],
            q6=batch["q6"],
            q16=batch["q16"],
            q50=batch["q50"],
            w_pi=1.0,
            w_pi_aux=0.15,
            w_val=1.0,
            w_outcome=1.0,
            w_score=0.02,
            w_own=0.15,
            w_q6=0.05,
            w_q16=0.15,
            w_q50=0.3,
            w_gamma=0.0005,
        )

        # v0 loss returns 14 values
        self.assertEqual(len(loss_result), 14)

        # Check that loss is a scalar and finite
        total_loss = loss_result[0]
        self.assertEqual(total_loss.shape, ())
        self.assertTrue(tf.math.is_finite(total_loss))

    def test_v1_loss_computation(self):
        """Test that v1 loss computation works correctly."""
        model = self._create_tiny_v1_model()
        batch = self._create_mock_v1_batch()

        outputs = model(batch["board_state"], batch["game_state"], training=True)

        loss_result = compute_loss(model,
            pi_logits=outputs[0],
            pi_logits_aux=outputs[8],
            game_outcome=outputs[2],
            score_logits=outputs[5],
            own_pred=outputs[4],
            q6_pred=outputs[9],
            q16_pred=outputs[10],
            q50_pred=outputs[11],
            gamma=outputs[7],
            policy=batch["policy"],
            policy_aux=batch["policy_aux"],
            score=batch["score"],
            score_one_hot=batch["score_one_hot"],
            own=batch["own"],
            q6=batch["q6"],
            q16=batch["q16"],
            q50=batch["q50"],
            w_pi=1.0,
            w_pi_aux=0.15,
            w_val=1.0,
            w_outcome=1.0,
            w_score=0.02,
            w_own=0.15,
            w_q6=0.05,
            w_q16=0.15,
            w_q50=0.3,
            w_gamma=0.0005,
            # v1 parameters
            q6_err_pred=outputs[12],
            q16_err_pred=outputs[13],
            q50_err_pred=outputs[14],
            q6_score_pred=outputs[15],
            q16_score_pred=outputs[16],
            q50_score_pred=outputs[17],
            q6_score_err_pred=outputs[18],
            q16_score_err_pred=outputs[19],
            q50_score_err_pred=outputs[20],
            pi_logits_soft=outputs[21],
            pi_logits_optimistic=outputs[22],
            # v1 ground truth
            q6_score=batch["q6_score"],
            q16_score=batch["q16_score"],
            q50_score=batch["q50_score"],
            # v1 weights
            w_q_err=0.1,
            w_q_score=0.1,
            w_q_score_err=0.1,
            w_pi_soft=0.1,
            w_pi_optimistic=0.1,
        )

        # v1 loss returns 14 values (same as v0)
        self.assertEqual(len(loss_result), 14)

        # Check that loss is a scalar and finite
        total_loss = loss_result[0]
        self.assertEqual(total_loss.shape, ())
        self.assertTrue(tf.math.is_finite(total_loss))

    def test_v0_gradient_step(self):
        """Test that v0 gradient step works correctly."""
        model = self._create_tiny_v0_model()
        batch = self._create_mock_v0_batch()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        with tf.GradientTape() as tape:
            outputs = model(batch["board_state"], batch["game_state"], training=True)

            loss_result = compute_loss(model,
                pi_logits=outputs[0],
                pi_logits_aux=outputs[8],
                game_outcome=outputs[2],
                score_logits=outputs[5],
                own_pred=outputs[4],
                q6_pred=outputs[9],
                q16_pred=outputs[10],
                q50_pred=outputs[11],
                gamma=outputs[7],
                policy=batch["policy"],
                policy_aux=batch["policy_aux"],
                score=batch["score"],
                score_one_hot=batch["score_one_hot"],
                own=batch["own"],
                q6=batch["q6"],
                q16=batch["q16"],
                q50=batch["q50"],
                w_pi=1.0,
                w_pi_aux=0.15,
                w_val=1.0,
                w_outcome=1.0,
                w_score=0.02,
                w_own=0.15,
                w_q6=0.05,
                w_q16=0.15,
                w_q50=0.3,
                w_gamma=0.0005,
            )
            total_loss = loss_result[0]

        gradients = tape.gradient(total_loss, model.trainable_variables)

        # Check that gradients are computed
        self.assertGreater(len(gradients), 0)

        # Check that at least some gradients are non-None
        non_none_grads = [g for g in gradients if g is not None]
        self.assertGreater(len(non_none_grads), 0)

        # Apply gradients (should not raise)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def test_v1_gradient_step(self):
        """Test that v1 gradient step works correctly."""
        model = self._create_tiny_v1_model()
        batch = self._create_mock_v1_batch()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        with tf.GradientTape() as tape:
            outputs = model(batch["board_state"], batch["game_state"], training=True)

            loss_result = compute_loss(model,
                pi_logits=outputs[0],
                pi_logits_aux=outputs[8],
                game_outcome=outputs[2],
                score_logits=outputs[5],
                own_pred=outputs[4],
                q6_pred=outputs[9],
                q16_pred=outputs[10],
                q50_pred=outputs[11],
                gamma=outputs[7],
                policy=batch["policy"],
                policy_aux=batch["policy_aux"],
                score=batch["score"],
                score_one_hot=batch["score_one_hot"],
                own=batch["own"],
                q6=batch["q6"],
                q16=batch["q16"],
                q50=batch["q50"],
                w_pi=1.0,
                w_pi_aux=0.15,
                w_val=1.0,
                w_outcome=1.0,
                w_score=0.02,
                w_own=0.15,
                w_q6=0.05,
                w_q16=0.15,
                w_q50=0.3,
                w_gamma=0.0005,
                # v1 parameters
                q6_err_pred=outputs[12],
                q16_err_pred=outputs[13],
                q50_err_pred=outputs[14],
                q6_score_pred=outputs[15],
                q16_score_pred=outputs[16],
                q50_score_pred=outputs[17],
                q6_score_err_pred=outputs[18],
                q16_score_err_pred=outputs[19],
                q50_score_err_pred=outputs[20],
                pi_logits_soft=outputs[21],
                pi_logits_optimistic=outputs[22],
                # v1 ground truth
                q6_score=batch["q6_score"],
                q16_score=batch["q16_score"],
                q50_score=batch["q50_score"],
                # v1 weights
                w_q_err=0.1,
                w_q_score=0.1,
                w_q_score_err=0.1,
                w_pi_soft=0.1,
                w_pi_optimistic=0.1,
            )
            total_loss = loss_result[0]

        gradients = tape.gradient(total_loss, model.trainable_variables)

        # Check that gradients are computed
        self.assertGreater(len(gradients), 0)

        # Check that at least some gradients are non-None
        non_none_grads = [g for g in gradients if g is not None]
        self.assertGreater(len(non_none_grads), 0)

        # Apply gradients (should not raise)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def test_v0_loss_decreases(self):
        """Test that v0 loss decreases over multiple training steps."""
        model = self._create_tiny_v0_model()
        batch = self._create_mock_v0_batch()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        initial_loss = None
        final_loss = None

        for step in range(10):
            with tf.GradientTape() as tape:
                outputs = model(batch["board_state"], batch["game_state"], training=True)

                loss_result = compute_loss(model,
                    pi_logits=outputs[0],
                    pi_logits_aux=outputs[8],
                    game_outcome=outputs[2],
                    score_logits=outputs[5],
                    own_pred=outputs[4],
                    q6_pred=outputs[9],
                    q16_pred=outputs[10],
                    q50_pred=outputs[11],
                    gamma=outputs[7],
                    policy=batch["policy"],
                    policy_aux=batch["policy_aux"],
                    score=batch["score"],
                    score_one_hot=batch["score_one_hot"],
                    own=batch["own"],
                    q6=batch["q6"],
                    q16=batch["q16"],
                    q50=batch["q50"],
                    w_pi=1.0,
                    w_pi_aux=0.15,
                    w_val=1.0,
                    w_outcome=1.0,
                    w_score=0.02,
                    w_own=0.15,
                    w_q6=0.05,
                    w_q16=0.15,
                    w_q50=0.3,
                    w_gamma=0.0005,
                )
                total_loss = loss_result[0]

            if step == 0:
                initial_loss = total_loss.numpy()
            if step == 9:
                final_loss = total_loss.numpy()

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Loss should decrease (or at least not increase significantly)
        self.assertLess(final_loss, initial_loss * 1.1)

    def test_v1_loss_decreases(self):
        """Test that v1 loss decreases over multiple training steps."""
        model = self._create_tiny_v1_model()
        batch = self._create_mock_v1_batch()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        initial_loss = None
        final_loss = None

        for step in range(10):
            with tf.GradientTape() as tape:
                outputs = model(batch["board_state"], batch["game_state"], training=True)

                loss_result = compute_loss(model,
                    pi_logits=outputs[0],
                    pi_logits_aux=outputs[8],
                    game_outcome=outputs[2],
                    score_logits=outputs[5],
                    own_pred=outputs[4],
                    q6_pred=outputs[9],
                    q16_pred=outputs[10],
                    q50_pred=outputs[11],
                    gamma=outputs[7],
                    policy=batch["policy"],
                    policy_aux=batch["policy_aux"],
                    score=batch["score"],
                    score_one_hot=batch["score_one_hot"],
                    own=batch["own"],
                    q6=batch["q6"],
                    q16=batch["q16"],
                    q50=batch["q50"],
                    w_pi=1.0,
                    w_pi_aux=0.15,
                    w_val=1.0,
                    w_outcome=1.0,
                    w_score=0.02,
                    w_own=0.15,
                    w_q6=0.05,
                    w_q16=0.15,
                    w_q50=0.3,
                    w_gamma=0.0005,
                    # v1 parameters
                    q6_err_pred=outputs[12],
                    q16_err_pred=outputs[13],
                    q50_err_pred=outputs[14],
                    q6_score_pred=outputs[15],
                    q16_score_pred=outputs[16],
                    q50_score_pred=outputs[17],
                    q6_score_err_pred=outputs[18],
                    q16_score_err_pred=outputs[19],
                    q50_score_err_pred=outputs[20],
                    pi_logits_soft=outputs[21],
                    pi_logits_optimistic=outputs[22],
                    # v1 ground truth
                    q6_score=batch["q6_score"],
                    q16_score=batch["q16_score"],
                    q50_score=batch["q50_score"],
                    # v1 weights
                    w_q_err=0.1,
                    w_q_score=0.1,
                    w_q_score_err=0.1,
                    w_pi_soft=0.1,
                    w_pi_optimistic=0.1,
                )
                total_loss = loss_result[0]

            if step == 0:
                initial_loss = total_loss.numpy()
            if step == 9:
                final_loss = total_loss.numpy()

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Loss should decrease (or at least not increase significantly)
        self.assertLess(final_loss, initial_loss * 1.1)

    def test_v1_v0_backward_compatible(self):
        """Test that v1 model can run v0-style loss (without v1 params)."""
        model = self._create_tiny_v1_model()
        batch = self._create_mock_v1_batch()

        outputs = model(batch["board_state"], batch["game_state"], training=True)

        # Call loss without v1-specific parameters
        loss_result = compute_loss(model,
            pi_logits=outputs[0],
            pi_logits_aux=outputs[8],
            game_outcome=outputs[2],
            score_logits=outputs[5],
            own_pred=outputs[4],
            q6_pred=outputs[9],
            q16_pred=outputs[10],
            q50_pred=outputs[11],
            gamma=outputs[7],
            policy=batch["policy"],
            policy_aux=batch["policy_aux"],
            score=batch["score"],
            score_one_hot=batch["score_one_hot"],
            own=batch["own"],
            q6=batch["q6"],
            q16=batch["q16"],
            q50=batch["q50"],
            w_pi=1.0,
            w_pi_aux=0.15,
            w_val=1.0,
            w_outcome=1.0,
            w_score=0.02,
            w_own=0.15,
            w_q6=0.05,
            w_q16=0.15,
            w_q50=0.3,
            w_gamma=0.0005,
            # Note: not passing v1 parameters - should still work
        )

        # Should still return valid loss
        total_loss = loss_result[0]
        self.assertEqual(total_loss.shape, ())
        self.assertTrue(tf.math.is_finite(total_loss))


if __name__ == "__main__":
    unittest.main()
