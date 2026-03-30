#!/usr/bin/env python3
"""
Compare p3achygo and KataGo with visual output.

This version includes:
- Board text art visualization
- Top 5 moves for both models
- Value, score, and ownership predictions
- GPU support for both models
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import tensorflow as tf

# Force GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import P3achyGoModel
import transforms
import constants
from board import GoBoard, char_at

# Define classes locally
from dataclasses import dataclass


@dataclass
class ModelPredictions:
    """Predictions from a Go model."""

    policy: np.ndarray  # Shape (362,)
    value: float
    score_mean: float
    ownership: np.ndarray  # Shape (19, 19)


@dataclass
class DivergenceMetrics:
    """Divergence metrics between two model predictions."""

    kl_divergence_policy: float
    js_divergence_policy: float
    value_diff: float
    score_diff: float
    ownership_mse: float
    ownership_max_diff: float
    combined_score: float


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
    model = tf.keras.models.load_model(model_path)
    return model


def load_examples(tfrecord_path: str, num_examples: int):
    """Load training examples from TFRecord file."""
    print(f"Loading examples from {tfrecord_path}")
    ds = tf.data.TFRecordDataset(tfrecord_path, compression_type="ZLIB")
    ds = ds.map(transforms.expand)
    ds = ds.take(num_examples)
    print(f"Dataset loaded successfully!")
    return ds


def compute_divergence(
    p3achy: ModelPredictions, katago: ModelPredictions, weights: dict = None
) -> DivergenceMetrics:
    """Compute divergence metrics between predictions."""
    if weights is None:
        weights = {"policy": 1.0, "value": 2.0, "score": 1.0, "ownership": 1.5}

    kl_div = kl_divergence(p3achy.policy, katago.policy)
    js_div = js_divergence(p3achy.policy, katago.policy)
    value_diff = abs(p3achy.value - katago.value)
    score_diff = abs(p3achy.score_mean - katago.score_mean)
    ownership_mse = np.mean((p3achy.ownership - katago.ownership) ** 2)
    ownership_max_diff = np.max(np.abs(p3achy.ownership - katago.ownership))

    combined_score = (
        weights["policy"] * js_div
        + weights["value"] * value_diff
        + weights["score"] * score_diff / 50.0
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


def board_from_features(board_state: np.ndarray, color: int) -> np.ndarray:
    """Reconstruct board array from feature planes."""
    # Channel 0 = our stones, Channel 1 = opp stones
    board = np.zeros((19, 19), dtype=np.int8)

    our_stones = board_state[:, :, 0]
    opp_stones = board_state[:, :, 1]

    if color == constants.BLACK:
        board[our_stones > 0] = constants.BLACK
        board[opp_stones > 0] = constants.WHITE
    else:
        board[our_stones > 0] = constants.WHITE
        board[opp_stones > 0] = constants.BLACK

    return board


def move_to_string(move_idx: int) -> str:
    """Convert move index to coordinate string."""
    if move_idx == constants.PASS_MOVE_ENCODING:
        return "PASS"
    row = move_idx // 19
    col = move_idx % 19
    col_letter = chr(ord("A") + col + (1 if col >= 8 else 0))  # Skip 'I'
    return f"{col_letter}{19 - row}"


def ownership_to_string(ownership: np.ndarray) -> str:
    """Convert ownership grid to visual string representation.

    Ownership convention: positive = black territory, negative = white territory
    ● = Strong white (<-0.7), ◯ = Weak white (-0.7 to -0.3), ⋅ = Neutral (-0.3 to 0.3)
    ◆ = Weak black (0.3 to 0.7), ○ = Strong black (>0.7)
    """

    def char_at(own_pred, i, j):
        x = own_pred[i, j]
        # Positive values = black territory, negative = white territory
        bounds = [-1.0, -0.5, 0, 0.5, 1.0]
        chars = ["●", "◯", "⋅", "◆", "○"]  # positive=black=○, negative=white=●
        deltas = [abs(x - b) for b in bounds]
        return chars[np.argmin(deltas)]

    s = []
    for i in range(19):
        s.append(
            f"{19 - i:2d} " + " ".join([char_at(ownership, i, j) for j in range(19)])
        )

    s.append("   " + " ".join(list("ABCDEFGHJKLMNOPQRST")))
    return "\n".join(s)


def extract_ground_truth(example: tuple) -> ModelPredictions:
    """Extract ground truth targets from training example."""
    # Example format from transforms.expand():
    # (input, input_global_state, color, komi, score, score_one_hot, policy, policy_aux, own, q30, q100, q200)
    # Indices:    0,       1,                2,     3,    4,     5,             6,      7,          8,   9,   10,   11
    policy_target = example[6].numpy()  # Already 362 values
    score_target = example[4].numpy()  # Scalar score
    ownership_target = example[8].numpy()  # 19x19 grid

    # Value target: use q100 (value at 100 visits)
    # q100 should be 1 for win, -1 for loss from current player's perspective
    value_target = example[10].numpy()

    return ModelPredictions(
        policy=policy_target,
        value=value_target,
        score_mean=score_target,
        ownership=ownership_target,
    )


def print_comparison(
    example_id: int,
    example: tuple,
    p3achy_preds: ModelPredictions,
    katago_preds: ModelPredictions,
    metrics: DivergenceMetrics,
):
    """Print detailed comparison with visualization."""

    # Extract example data
    board_state = example[0].numpy()
    color = example[2].numpy()
    komi = example[3].numpy()

    # Reconstruct board
    board = board_from_features(board_state, color)

    print(f"\n{'='*80}")
    print(f"EXAMPLE #{example_id}")
    print(f"{'='*80}")
    print(f"Current player: {'Black' if color == constants.BLACK else 'White'}")
    print(f"Komi: {komi:.1f}")
    print()

    # Print board
    print("Board position:")
    print(GoBoard.to_string(board))
    print()

    # Extract ground truth
    ground_truth = extract_ground_truth(example)

    # Print divergence metrics
    print(f"DIVERGENCE METRICS:")
    print(f"  Policy JS Divergence: {metrics.js_divergence_policy:.4f}")
    print(f"  Value Difference: {metrics.value_diff:.4f}")
    print(f"  Score Difference: {metrics.score_diff:.2f} points")
    print(f"  Ownership MSE: {metrics.ownership_mse:.4f}")
    print(f"  Combined Score: {metrics.combined_score:.4f}")
    print()

    # Print ground truth targets
    print(f"GROUND TRUTH (Training Target):")
    print(f"  Value (estimated): {ground_truth.value:.4f}")
    print(f"  Score: {ground_truth.score_mean:.2f}")
    print(f"  Top 5 moves:")
    top_5_gt = np.argsort(ground_truth.policy)[-5:][::-1]
    for rank, idx in enumerate(top_5_gt, 1):
        move_str = move_to_string(idx)
        prob = ground_truth.policy[idx]
        print(f"    {rank}. {move_str:>6s}: {prob:.4f}")

    # Ownership summary and grid
    own_gt = ground_truth.ownership
    black_territory_gt = np.sum(np.maximum(own_gt, 0))
    white_territory_gt = np.sum(np.abs(np.minimum(own_gt, 0)))
    print(f"  Ownership: B={black_territory_gt:.1f} W={white_territory_gt:.1f}")
    print(
        f"  Ownership grid (○ = black, ◆ = weak black, ⋅ = neutral, ◯ = weak white, ● = white):"
    )
    print(textwrap.indent(ownership_to_string(own_gt), "    "))
    print()

    # Print p3achygo predictions
    print(f"P3ACHYGO PREDICTIONS:")
    print(f"  Value (win prob): {p3achy_preds.value:.4f}")
    print(f"  Score mean: {p3achy_preds.score_mean:.2f}")
    print(f"  Top 5 moves:")
    top_5_p3 = np.argsort(p3achy_preds.policy)[-5:][::-1]
    for rank, idx in enumerate(top_5_p3, 1):
        move_str = move_to_string(idx)
        prob = p3achy_preds.policy[idx]
        marker = " ✓" if idx in top_5_gt else ""
        print(f"    {rank}. {move_str:>6s}: {prob:.4f}{marker}")

    # Ownership summary and grid
    own_p3 = p3achy_preds.ownership
    black_territory_p3 = np.sum(np.maximum(own_p3, 0))
    white_territory_p3 = np.sum(np.abs(np.minimum(own_p3, 0)))
    print(f"  Ownership: B={black_territory_p3:.1f} W={white_territory_p3:.1f}")
    print(
        f"  Ownership grid (○ = black, ◆ = weak black, ⋅ = neutral, ◯ = weak white, ● = white):"
    )
    print(textwrap.indent(ownership_to_string(own_p3), "    "))
    print()

    # Print KataGo predictions
    print(f"KATAGO PREDICTIONS:")
    print(f"  Value (win prob): {katago_preds.value:.4f}")
    print(f"  Score mean: {katago_preds.score_mean:.2f}")
    print(f"  Top 5 moves:")
    top_5_kg = np.argsort(katago_preds.policy)[-5:][::-1]
    for rank, idx in enumerate(top_5_kg, 1):
        move_str = move_to_string(idx)
        prob = katago_preds.policy[idx]
        # Mark if also in p3achy top 5 or ground truth
        marker_p3 = " *" if idx in top_5_p3 else ""
        marker_gt = " ✓" if idx in top_5_gt else ""
        print(f"    {rank}. {move_str:>6s}: {prob:.4f}{marker_p3}{marker_gt}")

    # Ownership summary and grid
    own_kg = katago_preds.ownership
    black_territory_kg = np.sum(np.maximum(own_kg, 0))
    white_territory_kg = np.sum(np.abs(np.minimum(own_kg, 0)))
    print(f"  Ownership: B={black_territory_kg:.1f} W={white_territory_kg:.1f}")
    print(
        f"  Ownership grid (○ = black, ◆ = weak black, ⋅ = neutral, ◯ = weak white, ● = white):"
    )
    print(textwrap.indent(ownership_to_string(own_kg), "    "))
    print()

    # Policy agreement
    overlap_p3_kg = len(set(top_5_p3.tolist()) & set(top_5_kg.tolist()))
    overlap_p3_gt = len(set(top_5_p3.tolist()) & set(top_5_gt.tolist()))
    overlap_kg_gt = len(set(top_5_kg.tolist()) & set(top_5_gt.tolist()))
    print(f"Policy agreement:")
    print(f"  p3achy ↔ KataGo: {overlap_p3_kg}/5 moves (* = in both)")
    print(f"  p3achy ↔ Ground Truth: {overlap_p3_gt}/5 moves (✓ = in ground truth)")
    print(f"  KataGo ↔ Ground Truth: {overlap_kg_gt}/5 moves")
    print()


def create_sgf_from_example(example: tuple) -> str:
    """Create SGF representation of position for KataGo analysis."""
    board_state = example[0].numpy()
    color = example[2].numpy()
    komi = example[3].numpy()

    our_stones = board_state[:, :, 0]
    opp_stones = board_state[:, :, 1]

    sgf_lines = [
        "(;FF[4]GM[1]SZ[19]",
        f"KM[{komi:.1f}]",
        "PL[{}]".format("B" if color == constants.BLACK else "W"),
    ]

    black_stones = []
    white_stones = []

    if color == constants.BLACK:
        black_locs = our_stones
        white_locs = opp_stones
    else:
        black_locs = opp_stones
        white_locs = our_stones

    for i in range(19):
        for j in range(19):
            # Our board array: i=0 is top row (row 19), i=18 is bottom row (row 1)
            # SGF coordinates: first letter is column (a-s), second letter is row where a=row 1 (bottom)
            # So we need to flip the row: i=0 → 's' (row 19), i=18 → 'a' (row 1)
            pos = chr(ord("a") + j) + chr(ord("a") + (18 - i))
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


def query_katago(
    katago_binary: str, model_path: str, config_path: str, sgf_content: str
) -> ModelPredictions:
    """Query KataGo via analysis engine with visits=1 (no MCTS)."""
    # Parse SGF to extract board position
    # SGF format: (;FF[4]GM[1]SZ[19]KM[7.5]PL[B]AB[aa][bb]AW[cc][dd])
    black_stones = []
    white_stones = []
    komi = 7.5
    next_player = "B"

    # Extract komi
    komi_match = re.search(r"KM\[([0-9.]+)\]", sgf_content)
    if komi_match:
        komi = float(komi_match.group(1))

    # Extract player to move
    pl_match = re.search(r"PL\[([BW])\]", sgf_content)
    if pl_match:
        next_player = pl_match.group(1)

    # Extract black stones
    ab_match = re.search(r"AB\[([^\]]+(?:\]\[[^\]]+)*)\]", sgf_content)
    if ab_match:
        stones_str = ab_match.group(1)
        black_stones = re.findall(r"([a-s][a-s])", stones_str)

    # Extract white stones
    aw_match = re.search(r"AW\[([^\]]+(?:\]\[[^\]]+)*)\]", sgf_content)
    if aw_match:
        stones_str = aw_match.group(1)
        white_stones = re.findall(r"([a-s][a-s])", stones_str)

    # Convert SGF coordinates (e.g., "aa") to GTP format (e.g., "A1")
    # SGF: lowercase a-s where a=leftmost column, a=bottom row
    # GTP: A-T (skip I) where A=leftmost, 1=bottom row
    def sgf_to_gtp(sgf_pos: str) -> str:
        # SGF column 'a'-'s' (0-18) -> GTP 'A'-'T' (skip 'I')
        col_idx = ord(sgf_pos[0]) - ord("a")
        col_letter = chr(ord("A") + col_idx + (1 if col_idx >= 8 else 0))

        # SGF row 'a'-'s' (0-18) where 'a'=row 1 (bottom)
        row = ord(sgf_pos[1]) - ord("a") + 1

        return f"{col_letter}{row}"

    initial_stones = []
    for pos in black_stones:
        initial_stones.append(["B", sgf_to_gtp(pos)])
    for pos in white_stones:
        initial_stones.append(["W", sgf_to_gtp(pos)])

    # Create analysis query with maxVisits=1 for single neural net eval
    query = {
        "id": "query",
        "initialStones": initial_stones,
        "initialPlayer": next_player,
        "moves": [],
        "rules": "chinese",
        "komi": komi,
        "boardXSize": 19,
        "boardYSize": 19,
        "includeOwnership": True,
        "includePolicy": True,
        "maxVisits": 1,  # No MCTS, just raw neural net output
    }

    query_json = json.dumps(query)

    try:
        # Get or create persistent KataGo process
        if (
            not hasattr(query_katago, "_katago_process")
            or query_katago._katago_process.poll() is not None
        ):
            # Start new KataGo process
            print(f"Starting KataGo process...")
            query_katago._katago_process = subprocess.Popen(
                [
                    katago_binary,
                    "analysis",
                    "-model",
                    model_path,
                    "-config",
                    config_path,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
            query_katago._katago_model_path = model_path
            query_katago._katago_config_path = config_path

            # Wait for KataGo to be ready (it takes ~5-7 seconds to load the model)
            import time

            print(f"Waiting for KataGo to initialize (this takes ~5-7 seconds)...")
            start_time = time.time()
            ready = False
            while not ready and (time.time() - start_time) < 30:
                # Non-blocking check for process death
                poll_result = query_katago._katago_process.poll()
                if poll_result is not None:
                    # Process died - get stderr
                    stderr_output = query_katago._katago_process.stderr.read()
                    raise Exception(
                        f"KataGo process died during initialization with exit code {poll_result}. Stderr: {stderr_output[:500]}"
                    )

                # Give it a moment to initialize
                time.sleep(0.5)

                # After 8 seconds, assume it's ready (increased from 5 to 8 for safety)
                if time.time() - start_time > 8:
                    ready = True
                    print(f"KataGo initialized successfully.")

        process = query_katago._katago_process

        # Send query
        process.stdin.write(query_json + "\n")
        process.stdin.flush()

        # Read response (one line)
        stdout_line = process.stdout.readline()

        if not stdout_line.strip():
            raise Exception(f"KataGo returned empty output")

        # Parse response
        try:
            response = json.loads(stdout_line.strip())
        except json.JSONDecodeError as e:
            raise Exception(
                f"Failed to parse KataGo response. Response: {stdout_line[:500]}"
            ) from e

        # Extract predictions
        # KataGo returns policy array (362 values) when maxVisits=1
        # According to KataGo docs: positive values sum to 1 (already probabilities)
        # and -1 indicates illegal moves
        policy_raw = response.get("policy", [])
        if len(policy_raw) == 362:
            policy_array = np.array(policy_raw, dtype=np.float32)
            # Replace illegal moves (-1) with 0
            policy_array = np.where(policy_array < 0, 0.0, policy_array)
        else:
            # Fallback: try moveInfos (for higher visits)
            move_infos = response.get("moveInfos", [])
            policy_array = np.zeros(362)
            for move_info in move_infos:
                move_str = move_info.get("move", "")
                prob = move_info.get("prior", 0.0)
                if move_str.lower() == "pass":
                    policy_array[361] = prob
                else:
                    try:
                        col_char = move_str[0]
                        row = int(move_str[1:])
                        col = ord(col_char.upper()) - ord("A")
                        if col >= 8:  # Account for no 'I'
                            col -= 1
                        row_idx = 19 - row
                        idx = row_idx * 19 + col
                        if 0 <= idx < 361:
                            policy_array[idx] = prob
                    except (ValueError, IndexError):
                        continue

        # Normalize policy to sum to 1
        policy_array = policy_array / (np.sum(policy_array) + 1e-10)

        value = response.get("rootInfo", {}).get("winrate", 0.5)
        # KataGo uses 'scoreLead' not 'scoreMean'
        score_mean = response.get("rootInfo", {}).get("scoreLead", 0.0)

        ownership_flat = response.get("ownership", [0.0] * 361)
        ownership = np.array(ownership_flat).reshape((19, 19))

        # KataGo reports from current player's perspective
        # If current player is White, we need to flip to Black's perspective
        # (since p3achygo always reports from Black's perspective)
        if next_player == "W":
            value = 1.0 - value  # Flip win rate
            score_mean = -score_mean  # Flip score
            ownership = -ownership  # Flip ownership (positive=black, negative=white)

        return ModelPredictions(
            policy=policy_array, value=value, score_mean=score_mean, ownership=ownership
        )

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"\nError querying KataGo: {e}")
        print(f"Number of initial stones: {len(initial_stones)}")
        if len(initial_stones) > 0:
            print(f"Sample stones: {initial_stones[:5]}")
        # Return neutral predictions on error
        return ModelPredictions(
            policy=np.ones(362) / 362,
            value=0.5,
            score_mean=0.0,
            ownership=np.zeros((19, 19)),
        )


def predict_p3achy(model: tf.keras.Model, example: tuple) -> ModelPredictions:
    """Run p3achygo model inference on an example (GPU)."""
    board_state = example[0]
    game_state = example[1]

    board_state = tf.expand_dims(board_state, 0)
    game_state = tf.expand_dims(game_state, 0)

    # Run inference on GPU
    with tf.device("/GPU:0"):
        outputs = model(board_state, game_state, training=False)

    pi_probs = outputs[1][0].numpy()
    outcome_probs = outputs[3][0].numpy()
    ownership = outputs[4][0].numpy()
    score_probs = outputs[6][0].numpy()

    value = outcome_probs[1]
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


def main():
    parser = argparse.ArgumentParser(
        description="Compare p3achygo and KataGo with visualization"
    )
    parser.add_argument("--p3achy_model", required=True)
    parser.add_argument("--katago_binary", default="katago")
    parser.add_argument("--katago_model", required=True)
    parser.add_argument("--katago_config", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--output", default="comparison_visual.txt")

    args = parser.parse_args()

    # Load model (will use GPU if available)
    print("Loading p3achygo model...")
    model = load_p3achy_model(args.p3achy_model)

    # Load examples
    print(f"Loading {args.num_examples} examples...")
    examples = load_examples(args.examples, args.num_examples)

    # Open output file
    with open(args.output, "w") as f_out:
        # Redirect stdout to both console and file
        import sys

        original_stdout = sys.stdout

        class Tee:
            def __init__(self, *files):
                self.files = files

            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()

            def flush(self):
                for f in self.files:
                    f.flush()

        sys.stdout = Tee(original_stdout, f_out)

        print(f"Comparing {args.num_examples} examples...")
        print(f"Output will be saved to {args.output}")
        print()

        comparisons = []
        for idx, example in enumerate(examples):
            print(f"\rProcessing example {idx + 1}...", end="", file=original_stdout)

            # Get p3achygo predictions
            p3achy_preds = predict_p3achy(model, example)

            # Get KataGo predictions with visits=1 (no MCTS)
            sgf_content = create_sgf_from_example(example)
            katago_preds = query_katago(
                args.katago_binary, args.katago_model, args.katago_config, sgf_content
            )

            # Compute divergence
            metrics = compute_divergence(p3achy_preds, katago_preds)

            # Store comparison
            comparisons.append(
                {
                    "example_id": idx,
                    "metrics": metrics,
                    "p3achy": p3achy_preds,
                    "katago": katago_preds,
                    "example": example,
                }
            )

            # Print detailed comparison
            print_comparison(idx, example, p3achy_preds, katago_preds, metrics)

        print(f"\n\nComparison saved to {args.output}", file=original_stdout)

        # Print and save top 10 most divergent examples
        print("\n" + "=" * 80)
        print("TOP 10 MOST DIVERGENT EXAMPLES")
        print("=" * 80)

        # Sort by different metrics
        categories = {
            "combined": ("Combined Score", lambda x: x["metrics"].combined_score),
            "policy": (
                "Policy JS Divergence",
                lambda x: x["metrics"].js_divergence_policy,
            ),
            "value": ("Value Difference", lambda x: x["metrics"].value_diff),
            "score": ("Score Difference", lambda x: x["metrics"].score_diff),
            "ownership": ("Ownership MSE", lambda x: x["metrics"].ownership_mse),
        }

        top_divergent = {}
        for category, (name, key_fn) in categories.items():
            sorted_comps = sorted(comparisons, key=key_fn, reverse=True)[:20]
            top_divergent[category] = [
                {"example_id": comp["example_id"], "metric_value": float(key_fn(comp))}
                for comp in sorted_comps
            ]

            print(f"\nTop 20 by {name}:")
            for rank, comp in enumerate(sorted_comps, 1):
                example_id = comp["example_id"]
                metric_val = key_fn(comp)
                print(
                    f"  {rank:2d}. Example #{example_id:3d} - {name}: {metric_val:.4f}"
                )

        # Print and save top 20 most agreed-on examples (lowest divergence)
        print("\n" + "=" * 80)
        print("TOP 20 MOST AGREED-ON EXAMPLES (Lowest Divergence)")
        print("=" * 80)

        most_agreed = {}
        for category, (name, key_fn) in categories.items():
            sorted_comps = sorted(comparisons, key=key_fn, reverse=False)[:20]
            most_agreed[category] = [
                {"example_id": comp["example_id"], "metric_value": float(key_fn(comp))}
                for comp in sorted_comps
            ]

            print(f"\nTop 20 lowest {name}:")
            for rank, comp in enumerate(sorted_comps, 1):
                example_id = comp["example_id"]
                metric_val = key_fn(comp)
                print(
                    f"  {rank:2d}. Example #{example_id:3d} - {name}: {metric_val:.4f}"
                )

        # Save to JSON
        json_output_divergent = args.output.replace(".txt", "_top_divergent.json")
        json_output_agreed = args.output.replace(".txt", "_top_agreed.json")

        with open(json_output_divergent, "w") as f:
            json.dump(top_divergent, f, indent=2)

        with open(json_output_agreed, "w") as f:
            json.dump(most_agreed, f, indent=2)

        print(f"\nTop 20 divergent examples saved to {json_output_divergent}")
        print(f"Top 20 agreed-on examples saved to {json_output_agreed}")
        print(f"Full comparison saved to {args.output}", file=original_stdout)
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
