"""
WD ablation: compare policy/value losses and value prediction distributions
across models on the professional game validation set.

Usage (inside container):
  cd /app && PYTHONPATH=python python python/scripts/wd_ablation.py \
    --models b12wd0.02,b12wd0.1 \
    [--val_path /p3achygo-data/pro_dataset/val.tfrecord.zz] \
    [--num_batches 50] \
    [--batch_size 256]
"""

import argparse
import sys
import numpy as np
import keras

sys.path.insert(0, "python")

from dataset import ChunkDataset
import train
from constants import *
from loss_coeffs import LossCoeffs
from backend_tf.model import *

CLEARLY_WON_THRESHOLD = 10.0  # score margin (points) to classify as clearly won/lost


def load_val_ds(val_path: str, batch_size: int) -> ChunkDataset:
    return ChunkDataset(val_path, batch_size)


def make_weights() -> LossWeights:
    coeffs = LossCoeffs.RLCoeffs()
    return LossWeights(
        w_pi=coeffs.w_pi,
        w_pi_aux=coeffs.w_pi_aux,
        w_val=coeffs.w_val,
        w_outcome=coeffs.w_outcome,
        w_score=coeffs.w_score,
        w_own=coeffs.w_own,
        w_q6=coeffs.w_q6,
        w_q16=coeffs.w_q16,
        w_q50=coeffs.w_q50,
        w_gamma=coeffs.w_gamma,
        w_q_err=coeffs.w_q_err,
        w_q_score=coeffs.w_q_score,
        w_q_score_err=coeffs.w_q_score_err,
        w_pi_soft=coeffs.w_pi_soft,
        w_pi_optimistic=coeffs.w_pi_optimistic,
    )


def run_model(model_path: str, val_ds, num_batches: int) -> dict:
    print(f"\nLoading {model_path} ...")
    model = keras.models.load_model(
        model_path,
        custom_objects=P3achyGoModel.custom_objects(),
        compile=False,
    )
    print("Loaded.")

    weights = make_weights()
    policy_losses, outcome_losses = [], []
    move_correct = move_total = outcome_correct = outcome_total = 0
    win_probs_all = []
    win_probs_won = []  # score > +threshold
    win_probs_lost = []  # score < -threshold

    for i, batch in enumerate(val_ds):
        if i >= num_batches:
            break

        (
            inp,
            inp_global,
            color,
            komi,
            score,
            score_one_hot,
            policy,
            policy_aux,
            own,
            q6,
            q16,
            q50,
            q6_score,
            q16_score,
            q50_score,
            game_outcome,
        ) = batch

        targets = GroundTruth(
            policy=policy,
            policy_aux=policy_aux,
            score=score,
            score_one_hot=score_one_hot,
            game_outcome=game_outcome,
            own=own,
            q6=q6,
            q16=q16,
            q50=q50,
            q6_score=q6_score,
            q16_score=q16_score,
            q50_score=q50_score,
        )

        result = train.val_step(inp, inp_global, targets, weights, model)

        policy_losses.append(result.policy_loss.numpy())
        outcome_losses.append(result.outcome_loss.numpy())

        # Move accuracy
        pred_moves = np.argmax(result.predictions.pi_logits.numpy(), axis=1)
        true_moves = np.argmax(policy.numpy(), axis=1)
        move_correct += int(np.sum(pred_moves == true_moves))
        move_total += len(pred_moves)

        # Win probability from softmax: last logit = win
        outcome_probs = tf.nn.softmax(result.predictions.game_outcome, axis=1).numpy()
        win_prob = outcome_probs[:, -1]  # win is last dim
        win_probs_all.extend(win_prob.tolist())

        # Outcome accuracy
        pred_win = np.argmax(outcome_probs, axis=1) == (outcome_probs.shape[1] - 1)
        true_win = score.numpy() >= 0
        outcome_correct += int(np.sum(pred_win == true_win))
        outcome_total += len(true_win)

        # Split by position clarity
        score_np = score.numpy()
        win_probs_won.extend(win_prob[score_np > CLEARLY_WON_THRESHOLD].tolist())
        win_probs_lost.extend(win_prob[score_np < -CLEARLY_WON_THRESHOLD].tolist())

        if (i + 1) % 10 == 0:
            print(
                f"  batch {i+1}/{num_batches}  "
                f"policy={np.mean(policy_losses):.4f}  "
                f"outcome={np.mean(outcome_losses):.4f}"
            )

    wp_all = np.array(win_probs_all)
    wp_won = np.array(win_probs_won)
    wp_lost = np.array(win_probs_lost)

    return {
        "policy_loss": np.mean(policy_losses),
        "policy_loss_std": np.std(policy_losses),
        "outcome_loss": np.mean(outcome_losses),
        "outcome_loss_std": np.std(outcome_losses),
        "move_accuracy": move_correct / move_total if move_total else 0.0,
        "outcome_accuracy": outcome_correct / outcome_total if outcome_total else 0.0,
        "wp_mean": float(np.mean(wp_all)),
        "wp_std": float(np.std(wp_all)),
        "wp_won_mean": float(np.mean(wp_won)) if len(wp_won) else float("nan"),
        "wp_won_std": float(np.std(wp_won)) if len(wp_won) else float("nan"),
        "wp_lost_mean": float(np.mean(wp_lost)) if len(wp_lost) else float("nan"),
        "wp_lost_std": float(np.std(wp_lost)) if len(wp_lost) else float("nan"),
        "n_won": len(wp_won),
        "n_lost": len(wp_lost),
        "_wp_won": wp_won,
        "_wp_lost": wp_lost,
    }


def ascii_histogram(values: np.ndarray, bins: int = 10, bar_width: int = 40) -> str:
    if len(values) == 0:
        return "  (no data)"
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    max_count = max(counts) if max(counts) > 0 else 1
    lines = []
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(count / max_count * bar_width)
        lines.append(f"  [{lo:.1f}-{hi:.1f}]  {bar:<{bar_width}}  {count}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated run names under /p3achygo-data/v4-models/",
    )
    parser.add_argument("--val_path", default="/p3achygo-data/val.tfrecord.zz")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--gen",
        type=int,
        default=200,
        help="Checkpoint generation to load (default: 200)",
    )
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    print(f"Val path:  {args.val_path}")
    print(
        f"Batches:   {args.num_batches} × {args.batch_size} = "
        f"~{args.num_batches * args.batch_size} positions"
    )
    print(f"Models:    {model_names}")

    results = {}
    for name in model_names:
        path = f"/p3achygo-data/v4-models/{name}/model_{args.gen:04d}.keras"
        val_ds = load_val_ds(args.val_path, args.batch_size)
        results[name] = run_model(path, val_ds, args.num_batches)

    # ── Summary table ──────────────────────────────────────────────────────────
    col = 20
    print("\n" + "=" * (30 + col * len(model_names)))
    print("SUMMARY")
    print("=" * (30 + col * len(model_names)))
    print(f"{'Metric':<30}" + "".join(f"{n:>{col}}" for n in model_names))
    print("-" * (30 + col * len(model_names)))

    rows = [
        ("Policy loss (mean)", "policy_loss", ".4f"),
        ("Policy loss (std/batch)", "policy_loss_std", ".4f"),
        ("Outcome loss (mean)", "outcome_loss", ".4f"),
        ("Outcome loss (std/batch)", "outcome_loss_std", ".4f"),
        ("Move accuracy", "move_accuracy", ".3f"),
        ("Outcome accuracy", "outcome_accuracy", ".3f"),
        ("--- Value collapse check ---", None, None),
        ("Win prob mean (all)", "wp_mean", ".4f"),
        ("Win prob std  (all)", "wp_std", ".4f"),
        ("Win prob mean (won,>10pt)", "wp_won_mean", ".4f"),
        ("Win prob std  (won,>10pt)", "wp_won_std", ".4f"),
        ("Win prob mean (lost,<-10pt)", "wp_lost_mean", ".4f"),
        ("Win prob std  (lost,<-10pt)", "wp_lost_std", ".4f"),
        ("N clearly won positions", "n_won", "d"),
        ("N clearly lost positions", "n_lost", "d"),
    ]

    for label, key, fmt in rows:
        if key is None:
            print(f"\n{label}")
            continue
        row = f"{label:<30}"
        for name in model_names:
            v = results[name][key]
            row += f"{v:>{col}{fmt}}"
        print(row)

    # ── Win-prob histograms ────────────────────────────────────────────────────
    for name in model_names:
        r = results[name]
        print(f"\n{'─'*60}")
        print(f"{name}: win-probability distribution")
        print(f"{'─'*60}")
        print(
            f"  Clearly WON (score > +{CLEARLY_WON_THRESHOLD:.0f}pt)  "
            f"n={r['n_won']}  "
            f"mean={r['wp_won_mean']:.3f}  std={r['wp_won_std']:.3f}"
        )
        print(ascii_histogram(r["_wp_won"]))
        print(
            f"  Clearly LOST (score < -{CLEARLY_WON_THRESHOLD:.0f}pt)  "
            f"n={r['n_lost']}  "
            f"mean={r['wp_lost_mean']:.3f}  std={r['wp_lost_std']:.3f}"
        )
        print(ascii_histogram(r["_wp_lost"]))


if __name__ == "__main__":
    main()
