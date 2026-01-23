#!/usr/bin/env python3
"""
Compare p3achygo model predictions with KataGo on training examples.

This script evaluates both models on a buffer of training examples and identifies
the top 50 most divergent examples across policy, value, score, and ownership.

Usage:
    python compare_with_katago.py \
        --p3achy_model /path/to/p3achygo/model \
        --katago_model /path/to/katago/model.bin.gz \
        --katago_config /path/to/katago/config.cfg \
        --examples /path/to/examples.tfrecord \
        --output divergent_examples.json \
        --num_examples 1000 \
        --top_k 50
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import P3achyGoModel
import transforms
import constants


@dataclass
class ModelPredictions:
    """Predictions from a Go model."""

    policy: np.ndarray  # Shape (362,) - move probabilities
    value: float  # Win probability for current player
    score_mean: float  # Expected score margin
    ownership: np.ndarray  # Shape (19, 19) - ownership predictions


@dataclass
class DivergenceMetrics:
    """Divergence metrics between two model predictions."""

    kl_divergence_policy: float  # KL(p3achy || katago) for policy
    js_divergence_policy: float  # JS divergence (symmetric)
    value_diff: float  # Absolute difference in value
    score_diff: float  # Absolute difference in score
    ownership_mse: float  # MSE of ownership predictions
    ownership_max_diff: float  # Maximum pointwise ownership difference
    combined_score: float  # Weighted combination for ranking


@dataclass
class ExampleComparison:
    """Comparison of models on a single example."""

    example_id: int
    sgf_moves: str  # SGF representation of position
    p3achy_preds: ModelPredictions
    katago_preds: ModelPredictions
    metrics: DivergenceMetrics


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL(P || Q) divergence."""
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence (symmetric)."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def load_p3achy_model(model_path: str) -> tf.keras.Model:
    """Load p3achygo model from checkpoint."""
    print(f"Loading p3achygo model from {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects=P3achyGoModel.custom_objects()
    )
    return model


def create_sgf_from_example(example: tuple) -> str:
    """Create SGF representation of position for KataGo analysis."""
    # Extract from tuple: (input, input_global_state, color, komi, ...)
    board_state = example[0].numpy()  # Shape (19, 19, 13)
    color = example[2].numpy()
    komi = example[3].numpy()

    # Reconstruct board from feature planes
    # Channel 0 = our stones, Channel 1 = opp stones
    our_stones = board_state[:, :, 0]
    opp_stones = board_state[:, :, 1]

    # Start SGF
    sgf_lines = [
        "(;FF[4]GM[1]SZ[19]",
        f"KM[{komi:.1f}]",
        "PL[{}]".format("B" if color == constants.BLACK else "W"),
    ]

    # Add setup stones (current board position)
    black_stones = []
    white_stones = []

    # Determine which color is which based on current player
    if color == constants.BLACK:
        # We are black, opponent is white
        black_locs = our_stones
        white_locs = opp_stones
    else:
        # We are white, opponent is black
        black_locs = opp_stones
        white_locs = our_stones

    for i in range(19):
        for j in range(19):
            pos = chr(ord("a") + j) + chr(ord("a") + i)
            if black_locs[i, j] > 0:
                black_stones.append(pos)
            elif white_locs[i, j] > 0:
                white_stones.append(pos)

    if black_stones:
        sgf_lines.append(f"AB[{']['.join(black_stones)}]")
    if white_stones:
        sgf_lines.append(f"AW[{']['.join(white_stones)}]")

    sgf_lines.append(")")
    return "".join(sgf_lines)


def query_katago_gtp(
    katago_binary: str, model_path: str, config_path: str, sgf_content: str
) -> ModelPredictions:
    """Query KataGo via GTP for predictions on a position."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sgf", delete=False) as f:
        f.write(sgf_content)
        sgf_path = f.name

    try:
        # Start KataGo GTP process
        process = subprocess.Popen(
            [katago_binary, "gtp", "-model", model_path, "-config", config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Load SGF
        commands = [
            f"loadsgf {sgf_path}",
            "kata-get-ownership",
            "kata-get-policy",
            "kata-analyze interval 1 maxvisits 1",
            "quit",
        ]

        stdout, stderr = process.communicate("\n".join(commands))

        # Parse KataGo output
        # This is a simplified parser - you may need to adjust based on actual output format
        lines = stdout.strip().split("\n")

        policy = np.zeros(362)
        ownership = np.zeros((19, 19))
        value = 0.5
        score_mean = 0.0

        # Parse kata-analyze output for value and score
        for line in lines:
            if line.startswith("info"):
                # Parse analysis output
                # Format: info move <move> visits <n> winrate <wr> scoreMean <sm> ...
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "winrate" and i + 1 < len(parts):
                        value = float(parts[i + 1])
                    elif part == "scoreMean" and i + 1 < len(parts):
                        score_mean = float(parts[i + 1])
            elif line.startswith("kata-get-policy"):
                # Parse policy output
                # You'll need to implement this based on KataGo's output format
                pass
            elif line.startswith("kata-get-ownership"):
                # Parse ownership output
                # You'll need to implement this based on KataGo's output format
                pass

        return ModelPredictions(
            policy=policy, value=value, score_mean=score_mean, ownership=ownership
        )

    finally:
        os.unlink(sgf_path)


def query_katago_analysis(
    katago_binary: str, model_path: str, config_path: str, sgf_content: str
) -> ModelPredictions:
    """Query KataGo via analysis engine for predictions."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sgf", delete=False) as f:
        f.write(sgf_content)
        sgf_path = f.name

    # Create analysis query
    query = {
        "id": "query",
        "moves": [],  # Position is in SGF
        "initialStones": [],  # Will be loaded from SGF
        "rules": "chinese",
        "komi": 7.5,
        "boardXSize": 19,
        "boardYSize": 19,
        "includeOwnership": True,
        "includePolicy": True,
        "maxVisits": 1,  # Single evaluation, no search
    }

    query_json = json.dumps(query)

    try:
        # Run KataGo analysis
        process = subprocess.Popen(
            [katago_binary, "analysis", "-model", model_path, "-config", config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(query_json + "\n")

        # Parse response
        response = json.loads(stdout.strip())

        # Extract predictions
        policy_array = np.array(response.get("policy", [0.0] * 362))
        if len(policy_array) < 362:
            policy_array = np.pad(policy_array, (0, 362 - len(policy_array)))

        value = response.get("rootInfo", {}).get("winrate", 0.5)
        score_mean = response.get("rootInfo", {}).get("scoreMean", 0.0)

        ownership_flat = response.get("ownership", [0.0] * 361)
        ownership = np.array(ownership_flat).reshape((19, 19))

        return ModelPredictions(
            policy=policy_array, value=value, score_mean=score_mean, ownership=ownership
        )

    finally:
        os.unlink(sgf_path)


def predict_p3achy(model: tf.keras.Model, example: tuple) -> ModelPredictions:
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
    pi_logits = outputs[0][0].numpy()  # Shape (362,)
    pi_probs = outputs[1][0].numpy()  # Shape (362,)
    outcome_probs = outputs[3][0].numpy()  # Shape (2,)
    ownership = outputs[4][0].numpy()  # Shape (19, 19)
    score_probs = outputs[6][0].numpy()  # Shape (800,)

    # Compute value (win probability for current player)
    # outcome_probs[0] is loss, outcome_probs[1] is win
    value = outcome_probs[1]

    # Compute expected score from score distribution
    score_range = np.arange(
        -constants.SCORE_RANGE_MIDPOINT, constants.SCORE_RANGE_MIDPOINT
    )
    score_mean = np.sum(score_probs * score_range)

    return ModelPredictions(
        policy=pi_probs,
        value=float(value),
        score_mean=float(score_mean),
        ownership=ownership,
    )


def compute_divergence(
    p3achy: ModelPredictions, katago: ModelPredictions, weights: Dict[str, float] = None
) -> DivergenceMetrics:
    """Compute divergence metrics between predictions."""
    if weights is None:
        weights = {"policy": 1.0, "value": 2.0, "score": 1.0, "ownership": 1.5}

    # Policy divergence
    kl_div = kl_divergence(p3achy.policy, katago.policy)
    js_div = js_divergence(p3achy.policy, katago.policy)

    # Value difference
    value_diff = abs(p3achy.value - katago.value)

    # Score difference
    score_diff = abs(p3achy.score_mean - katago.score_mean)

    # Ownership metrics
    ownership_mse = np.mean((p3achy.ownership - katago.ownership) ** 2)
    ownership_max_diff = np.max(np.abs(p3achy.ownership - katago.ownership))

    # Combined score for ranking
    combined_score = (
        weights["policy"] * js_div
        + weights["value"] * value_diff
        + weights["score"] * score_diff / 50.0  # Normalize score to ~[0, 1]
        + weights["ownership"] * ownership_mse
    )

    return DivergenceMetrics(
        kl_divergence_policy=float(kl_div),
        js_divergence_policy=float(js_div),
        value_diff=float(value_diff),
        score_diff=float(score_diff),
        ownership_mse=float(ownership_mse),
        ownership_max_diff=float(ownership_max_diff),
        combined_score=float(combined_score),
    )


def load_examples(tfrecord_path: str, num_examples: int) -> tf.data.Dataset:
    """Load training examples from TFRecord file."""
    print(f"Loading examples from {tfrecord_path}")

    ds = tf.data.TFRecordDataset(tfrecord_path, compression_type="ZLIB")
    ds = ds.map(transforms.expand_v0)
    ds = ds.take(num_examples)

    return ds


def compare_models(
    p3achy_model: tf.keras.Model,
    katago_binary: str,
    katago_model: str,
    katago_config: str,
    examples: tf.data.Dataset,
    use_analysis_engine: bool = True,
) -> List[ExampleComparison]:
    """Compare both models on all examples."""
    comparisons = []

    query_fn = query_katago_analysis if use_analysis_engine else query_katago_gtp

    for idx, example in enumerate(examples):
        print(f"Processing example {idx + 1}...", end="\r")

        # Get p3achygo predictions
        p3achy_preds = predict_p3achy(p3achy_model, example)

        # Create SGF for KataGo
        sgf_content = create_sgf_from_example(example)

        # Get KataGo predictions
        try:
            katago_preds = query_fn(
                katago_binary, katago_model, katago_config, sgf_content
            )
        except Exception as e:
            print(f"\nError querying KataGo for example {idx}: {e}")
            continue

        # Compute divergence
        metrics = compute_divergence(p3achy_preds, katago_preds)

        # Store comparison
        comparisons.append(
            ExampleComparison(
                example_id=idx,
                sgf_moves=sgf_content,
                p3achy_preds=p3achy_preds,
                katago_preds=katago_preds,
                metrics=metrics,
            )
        )

    print()  # New line after progress
    return comparisons


def find_top_divergent(
    comparisons: List[ExampleComparison], top_k: int = 50, category: str = "combined"
) -> List[ExampleComparison]:
    """Find top-k most divergent examples by category."""
    if category == "combined":
        key_fn = lambda x: x.metrics.combined_score
    elif category == "policy":
        key_fn = lambda x: x.metrics.js_divergence_policy
    elif category == "value":
        key_fn = lambda x: x.metrics.value_diff
    elif category == "score":
        key_fn = lambda x: x.metrics.score_diff
    elif category == "ownership":
        key_fn = lambda x: x.metrics.ownership_mse
    else:
        raise ValueError(f"Unknown category: {category}")

    sorted_comparisons = sorted(comparisons, key=key_fn, reverse=True)
    return sorted_comparisons[:top_k]


def serialize_comparison(comp: ExampleComparison) -> Dict[str, Any]:
    """Serialize comparison to JSON-compatible dict."""
    return {
        "example_id": comp.example_id,
        "sgf_moves": comp.sgf_moves,
        "p3achy_preds": {
            "policy": comp.p3achy_preds.policy.tolist(),
            "value": comp.p3achy_preds.value,
            "score_mean": comp.p3achy_preds.score_mean,
            "ownership": comp.p3achy_preds.ownership.tolist(),
        },
        "katago_preds": {
            "policy": comp.katago_preds.policy.tolist(),
            "value": comp.katago_preds.value,
            "score_mean": comp.katago_preds.score_mean,
            "ownership": comp.katago_preds.ownership.tolist(),
        },
        "metrics": asdict(comp.metrics),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare p3achygo and KataGo model predictions"
    )
    parser.add_argument(
        "--p3achy_model", required=True, help="Path to p3achygo model checkpoint"
    )
    parser.add_argument(
        "--katago_binary",
        default="katago",
        help="Path to KataGo binary (default: katago)",
    )
    parser.add_argument(
        "--katago_model", required=True, help="Path to KataGo model file (.bin.gz)"
    )
    parser.add_argument(
        "--katago_config", required=True, help="Path to KataGo config file (.cfg)"
    )
    parser.add_argument(
        "--examples", required=True, help="Path to training examples TFRecord file"
    )
    parser.add_argument(
        "--output",
        default="divergent_examples.json",
        help="Output JSON file (default: divergent_examples.json)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of examples to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of most divergent examples to output (default: 50)",
    )
    parser.add_argument(
        "--use_gtp",
        action="store_true",
        help="Use GTP interface instead of analysis engine",
    )

    args = parser.parse_args()

    # Load p3achygo model
    p3achy_model = load_p3achy_model(args.p3achy_model)

    # Load examples
    examples = load_examples(args.examples, args.num_examples)

    # Compare models
    print("Comparing models...")
    comparisons = compare_models(
        p3achy_model,
        args.katago_binary,
        args.katago_model,
        args.katago_config,
        examples,
        use_analysis_engine=not args.use_gtp,
    )

    print(f"Processed {len(comparisons)} examples")

    # Find top divergent examples for each category
    results = {
        "combined": find_top_divergent(comparisons, args.top_k, "combined"),
        "policy": find_top_divergent(comparisons, args.top_k, "policy"),
        "value": find_top_divergent(comparisons, args.top_k, "value"),
        "score": find_top_divergent(comparisons, args.top_k, "score"),
        "ownership": find_top_divergent(comparisons, args.top_k, "ownership"),
    }

    # Serialize and save
    output_data = {
        category: [serialize_comparison(comp) for comp in comps]
        for category, comps in results.items()
    }

    output_data["summary"] = {
        "total_examples": len(comparisons),
        "top_k": args.top_k,
        "avg_policy_js_div": np.mean(
            [c.metrics.js_divergence_policy for c in comparisons]
        ),
        "avg_value_diff": np.mean([c.metrics.value_diff for c in comparisons]),
        "avg_score_diff": np.mean([c.metrics.score_diff for c in comparisons]),
        "avg_ownership_mse": np.mean([c.metrics.ownership_mse for c in comparisons]),
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print("\nSummary statistics:")
    print(
        f"  Average policy JS divergence: {output_data['summary']['avg_policy_js_div']:.4f}"
    )
    print(f"  Average value difference: {output_data['summary']['avg_value_diff']:.4f}")
    print(f"  Average score difference: {output_data['summary']['avg_score_diff']:.2f}")
    print(f"  Average ownership MSE: {output_data['summary']['avg_ownership_mse']:.4f}")


if __name__ == "__main__":
    main()
