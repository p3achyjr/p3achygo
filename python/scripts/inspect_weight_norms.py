"""
Print per-layer weight norms for a saved p3achygo Keras model.
Usage: PYTHONPATH=python python python/scripts/inspect_weight_norms.py <model_path>
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, "python")
from model import P3achyGoModel


def l2_norm(weights):
    return np.sqrt(sum(np.sum(w**2) for w in weights))


def rms(weights):
    total_params = sum(w.size for w in weights)
    sum_sq = sum(np.sum(w**2) for w in weights)
    return np.sqrt(sum_sq / total_params) if total_params > 0 else 0.0


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.keras>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"Loading model from {model_path} ...")
    model = keras.models.load_model(
        model_path, custom_objects=P3achyGoModel.custom_objects(), compile=False
    )
    print("Model loaded.\n")

    # Collect per-layer stats
    rows = []
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        arrays = [np.array(w) for w in weights]
        num_params = sum(w.size for w in arrays)
        norm = l2_norm(arrays)
        rms_val = rms(arrays)
        rows.append((layer.name, num_params, norm, rms_val))

    # Group into body vs heads
    body_rows = [
        (n, p, no, r)
        for n, p, no, r in rows
        if "policy_head" not in n and "value_head" not in n
    ]
    policy_rows = [(n, p, no, r) for n, p, no, r in rows if "policy_head" in n]
    value_rows = [(n, p, no, r) for n, p, no, r in rows if "value_head" in n]

    col_w = max(len(n) for n, *_ in rows) + 2

    header = f"{'Layer':<{col_w}}  {'Params':>10}  {'L2 norm':>12}  {'RMS':>12}"
    sep = "-" * len(header)

    def print_group(title, group_rows):
        print(title)
        print(sep)
        print(header)
        print(sep)
        for name, params, norm, rms_val in group_rows:
            print(f"{name:<{col_w}}  {params:>10,}  {norm:>12.4f}  {rms_val:>12.6f}")
        total_params = sum(p for _, p, _, _ in group_rows)
        total_norm = (
            l2_norm([np.zeros(1)] * 0)
            if not group_rows
            else np.sqrt(sum(no**2 for _, _, no, _ in group_rows))
        )
        print(sep)
        print(f"{'TOTAL':<{col_w}}  {total_params:>10,}  {total_norm:>12.4f}")
        print()

    print_group("=== BODY ===", body_rows)
    print_group("=== POLICY HEAD ===", policy_rows)
    print_group("=== VALUE HEAD ===", value_rows)

    # Summary
    print("=== SUMMARY ===")
    for title, group_rows in [
        ("Body", body_rows),
        ("Policy head", policy_rows),
        ("Value head", value_rows),
    ]:
        if not group_rows:
            continue
        total_norm = np.sqrt(sum(no**2 for _, _, no, _ in group_rows))
        all_arrays = []
        for layer in model.layers:
            if (
                (
                    title == "Body"
                    and "policy_head" not in layer.name
                    and "value_head" not in layer.name
                )
                or (title == "Policy head" and "policy_head" in layer.name)
                or (title == "Value head" and "value_head" in layer.name)
            ):
                all_arrays.extend([np.array(w) for w in layer.get_weights()])
        rms_val = rms(all_arrays)
        total_params = sum(p for _, p, _, _ in group_rows)
        print(
            f"  {title:<14} params={total_params:>8,}  L2={total_norm:.4f}  RMS={rms_val:.6f}"
        )


if __name__ == "__main__":
    main()
