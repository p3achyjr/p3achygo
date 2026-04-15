"""
Compute sel_mult calibration from .stats files and write a key=value file
that the self-play binary reads via --sel_mult_calibration_file.

Stats files contain a percentile table (one row per field, columns p01, p05,
p10, ..., p95, p99) followed by scalar metadata lines of the form key=value.

Output format (calibration file):
  field.percentile=value   (for all fields and percentiles in the table)
  sel_mult_mean=value      (mean sel_mult from the previous generation)

The C++ side reads field.percentile entries into flat_hash_maps keyed by
percentile string. sel_mult_mean is used by the Python RL loop to compute
sel_mult_base = 1 / sel_mult_mean for the next generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from absl import logging


def _parse_stats_file(
    path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Parse a single .stats file.

    Returns:
      percentiles: field_name -> {percentile_label -> value}
      metadata:    key -> value  (from trailing key=value lines, e.g. sel_mult_mean)
    """
    percentiles: dict[str, dict[str, float]] = {}
    metadata: dict[str, float] = {}
    percentile_labels: list[str] = []

    with open(path, errors="replace") as f:
        lines = f.readlines()

    header_found = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Scalar metadata: lines of the form "key=value" with no spaces in key
        if "=" in line and " " not in line:
            key, _, val_str = line.partition("=")
            try:
                metadata[key] = float(val_str)
            except ValueError:
                pass
            continue
        parts = line.split()
        if not header_found:
            if parts[0] == "field":
                percentile_labels = parts[1:]
                header_found = True
            continue
        if len(parts) < 2 or len(parts) - 1 != len(percentile_labels):
            continue
        field = parts[0]
        percentiles[field] = {
            label: float(val) for label, val in zip(percentile_labels, parts[1:])
        }

    return percentiles, metadata


def compute_calibration(stats_dir: Path, gen: int) -> Optional[dict[str, float]]:
    """Compute calibration values from all .stats files for the given gen.

    Averages each percentile value and sel_mult_mean across batch files.
    Returns a flat dict with entries:
      'field.percentile' -> averaged value   (for all percentile table entries)
      'sel_mult_mean'    -> averaged mean     (for computing sel_mult_base)

    Returns None if no stats files are found.
    """
    stats_files = sorted(stats_dir.glob(f"gen{gen:03d}_*.stats"))
    if not stats_files:
        logging.info(f"No .stats files found for gen {gen} in {stats_dir}. ")
        return None

    # Accumulate percentile table: field -> label -> [values across files]
    pctl_accum: dict[str, dict[str, list[float]]] = {}
    # Accumulate scalar metadata: key -> [values across files]
    meta_accum: dict[str, list[float]] = {}

    for path in stats_files:
        pctls, meta = _parse_stats_file(path)
        for field, labels in pctls.items():
            if field not in pctl_accum:
                pctl_accum[field] = {}
            for label, val in labels.items():
                pctl_accum[field].setdefault(label, []).append(val)
        for key, val in meta.items():
            meta_accum.setdefault(key, []).append(val)

    calib: dict[str, float] = {}
    for field, labels in pctl_accum.items():
        for label, vals in labels.items():
            calib[f"{field}.{label}"] = sum(vals) / len(vals)
    for key, vals in meta_accum.items():
        calib[key] = sum(vals) / len(vals)

    logging.info(
        f"sel_mult calibration from gen {gen} ({len(stats_files)} files): "
        f"{len(calib)} entries"
        + (
            f"  sel_mult_mean={calib['sel_mult_mean']:.4f}"
            if "sel_mult_mean" in calib
            else ""
        )
    )
    return calib or None


def compute_sel_mult_base(calib: dict[str, float]) -> Optional[float]:
    """Return 1 / sel_mult_mean, or None if sel_mult_mean is not in calib."""
    mean = calib.get("sel_mult_mean")
    if mean is None or mean <= 0.0:
        return None
    return 1.0 / mean


def write_calibration_file(calib: dict[str, float], path: Path) -> None:
    """Write calibration as key=value text file.

    Percentile entries are written as 'field.percentile=value'.
    Scalar metadata (e.g. sel_mult_mean) are written as 'key=value'.
    """
    with open(path, "w") as f:
        for key, val in sorted(calib.items()):
            f.write(f"{key}={val:.6f}\n")
