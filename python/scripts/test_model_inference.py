#!/usr/bin/env python3
"""
Test script to verify model loading and inference on training examples.

This is a simpler test version before running the full KataGo comparison.

Usage:
    python test_model_inference.py \
        --model /path/to/p3achygo/model \
        --examples /path/to/examples.tfrecord.zz \
        --num_examples 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import P3achyGoModel
import transforms
import constants


def load_model(model_path: str) -> tf.keras.Model:
    """Load p3achygo model from checkpoint."""
    print(f"Loading p3achygo model from {model_path}...")
    model = tf.keras.models.load_model(
        model_path, custom_objects=P3achyGoModel.custom_objects()
    )
    print(f"Model loaded successfully!")
    return model


def load_examples(tfrecord_path: str, num_examples: int) -> tf.data.Dataset:
    """Load training examples from TFRecord file."""
    print(f"Loading examples from {tfrecord_path}...")

    ds = tf.data.TFRecordDataset(tfrecord_path, compression_type="ZLIB")
    ds = ds.map(transforms.expand)
    ds = ds.take(num_examples)

    print(f"Dataset loaded successfully!")
    return ds


def predict_p3achy(model: tf.keras.Model, example: tuple) -> dict:
    """Run p3achygo model inference on an example."""
    # Extract tensors from example tuple
    # expand() returns: (input, input_global_state, color, komi, score, score_one_hot,
    #                    policy, policy_aux, own, q30, q100, q200)
    board_state = example[0]  # Shape (19, 19, 13)
    game_state = example[1]  # Shape (7,)

    # Add batch dimension
    board_state = tf.expand_dims(board_state, 0)
    game_state = tf.expand_dims(game_state, 0)

    # Run inference
    outputs = model(board_state, game_state, training=False)

    # Extract outputs
    # outputs = (pi_logits, pi, outcome_logits, outcome_probs, ownership,
    #            score_logits, score_probs, gamma, pi_logits_aux, q30, q100, q200)
    pi_probs = outputs[1][0].numpy()  # Shape (362,)
    outcome_probs = outputs[3][0].numpy()  # Shape (2,)
    ownership = outputs[4][0].numpy()  # Shape (19, 19)
    score_probs = outputs[6][0].numpy()  # Shape (800,)

    # Compute value (win probability for current player)
    value = float(outcome_probs[1])

    # Compute expected score from score distribution
    score_range = np.arange(
        -constants.SCORE_RANGE_MIDPOINT, constants.SCORE_RANGE_MIDPOINT
    )
    score_mean = float(np.sum(score_probs * score_range))

    return {
        "policy": pi_probs,
        "value": value,
        "score_mean": score_mean,
        "ownership": ownership,
    }


def display_prediction(idx: int, example: tuple, pred: dict):
    """Display prediction results."""
    print(f"\n{'='*60}")
    print(f"Example {idx}")
    print(f"{'='*60}")

    # Extract from tuple: (input, input_global_state, color, komi, ...)
    board_state = example[0].numpy()  # (19, 19, 13)
    color = example[2].numpy()
    komi = example[3].numpy()

    # Board state has our stones in channel 0, opp stones in channel 1
    our_stones = np.sum(board_state[:, :, 0])
    opp_stones = np.sum(board_state[:, :, 1])

    print(f"Board: {our_stones:.0f} our stones, {opp_stones:.0f} opp stones")
    print(f"Current player: {'Black' if color == constants.BLACK else 'White'}")
    print(f"Komi: {komi:.1f}")

    # Show predictions
    print(f"\nPredictions:")
    print(f"  Value (win prob): {pred['value']:.4f}")
    print(f"  Score mean: {pred['score_mean']:.2f}")

    # Show top 5 moves by policy
    top_5_indices = np.argsort(pred["policy"])[-5:][::-1]
    print(f"\n  Top 5 moves:")
    for rank, idx in enumerate(top_5_indices, 1):
        if idx == constants.PASS_MOVE_ENCODING:
            move_str = "PASS"
        else:
            row = idx // 19
            col = idx % 19
            move_str = f"({row}, {col})"
        prob = pred["policy"][idx]
        print(f"    {rank}. {move_str}: {prob:.4f}")

    # Show ownership stats
    own = pred["ownership"]
    print(f"\n  Ownership:")
    print(f"    Black territory: {np.sum(np.maximum(own, 0)):.1f} points")
    print(f"    White territory: {np.sum(np.abs(np.minimum(own, 0))):.1f} points")


def main():
    parser = argparse.ArgumentParser(
        description="Test p3achygo model inference on training examples"
    )
    parser.add_argument(
        "--model", required=True, help="Path to p3achygo model checkpoint"
    )
    parser.add_argument(
        "--examples",
        required=True,
        help="Path to training examples TFRecord file (.tfrecord.zz)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to process (default: 10)",
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Load examples
    examples = load_examples(args.examples, args.num_examples)

    # Process examples
    print(f"\nProcessing {args.num_examples} examples...\n")

    for idx, example in enumerate(examples):
        pred = predict_p3achy(model, example)
        display_prediction(idx, example, pred)

    print(f"\n{'='*60}")
    print(f"Test completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
