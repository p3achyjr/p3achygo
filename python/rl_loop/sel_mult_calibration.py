"""
Compute sel_mult calibration from .stats files and write a key=value file
that the self-play binary reads via --sel_mult_calibration_file.

Stats files contain a percentile table (one row per field, columns p0..p100
at 2.5% increments). We average the relevant percentile values across all
.stats files for a given generation to produce stable threshold estimates.

Calibration keys and their meaning:
  g1_p50   / g1_p72_5 / g1_p95   : G1 breakpoints on nn_mcts_diff
  g2b_p2_5 / g2b_p25             : G2 bonus breakpoints on top12_q_gap_nz
  g2p_p80  / g2p_p92_5 / g2p_p97_5 : G2 penalty breakpoints on top12_q_gap_nz
  std_p70  / std_p95             : stddev bonus breakpoints on v_outcome_stddev
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from absl import logging


# Maps calibration key -> (stats_field, percentile_label)
_CALIB_SPEC: dict[str, tuple[str, str]] = {
    "g1_p50": ("nn_mcts_diff", "p50"),
    "g1_p72_5": ("nn_mcts_diff", "p72.5"),
    "g1_p95": ("nn_mcts_diff", "p95"),
    "g2b_p2_5": ("top12_q_gap_nz", "p2.5"),
    "g2b_p25": ("top12_q_gap_nz", "p25"),
    "g2p_p70": ("top12_q_gap_nz", "p70"),
    "g2p_p80": ("top12_q_gap_nz", "p80"),
    "g2p_p92_5": ("top12_q_gap_nz", "p92.5"),
    "g2p_p95": ("top12_q_gap_nz", "p95"),
    "g2p_p97_5": ("top12_q_gap_nz", "p97.5"),
    "std_p70": ("v_outcome_stddev", "p70"),
    "std_p95": ("v_outcome_stddev", "p95"),
}


def _parse_stats_file(path: Path) -> dict[str, dict[str, float]]:
    """Parse a single .stats file.

    Returns a dict mapping field_name -> {percentile_label -> value}, e.g.
      {'nn_mcts_diff': {'p0': 0.0, 'p2.5': 0.002, ...}, ...}
    """
    result: dict[str, dict[str, float]] = {}
    percentile_labels: list[str] = []

    with open(path, errors="replace") as f:
        lines = f.readlines()

    header_found = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not header_found:
            # Header row starts with "field"
            if parts[0] == "field":
                percentile_labels = parts[1:]
                header_found = True
            continue
        if len(parts) < 2 or len(parts) - 1 != len(percentile_labels):
            continue
        field = parts[0]
        result[field] = {
            label: float(val) for label, val in zip(percentile_labels, parts[1:])
        }

    return result


def compute_calibration(stats_dir: Path, gen: int) -> Optional[dict[str, float]]:
    """Compute calibration values from all .stats files for the given gen.

    Averages each percentile value across batch files. Returns None if no
    stats files are found (caller should omit the flag and use C++ defaults).
    """
    stats_files = sorted(stats_dir.glob(f"gen{gen:03d}_*.stats"))
    if not stats_files:
        logging.info(
            f"No .stats files found for gen {gen} in {stats_dir}. "
            "Using C++ default sel_mult thresholds."
        )
        return None

    # Accumulate: field -> percentile_label -> list of values across files
    accum: dict[str, dict[str, list[float]]] = {}
    for path in stats_files:
        parsed = _parse_stats_file(path)
        for field, pctls in parsed.items():
            if field not in accum:
                accum[field] = {}
            for label, val in pctls.items():
                accum[field].setdefault(label, []).append(val)

    # Average and extract the values we need
    calib: dict[str, float] = {}
    missing = []
    for key, (field, label) in _CALIB_SPEC.items():
        vals = accum.get(field, {}).get(label)
        if not vals:
            missing.append(f"{field}/{label}")
            continue
        calib[key] = sum(vals) / len(vals)

    if missing:
        logging.warning(
            f"sel_mult calibration: missing fields {missing} in stats files. "
            "Affected thresholds will use C++ defaults."
        )

    logging.info(
        f"sel_mult calibration from gen {gen} ({len(stats_files)} files): "
        + "  ".join(f"{k}={v:.4f}" for k, v in calib.items())
    )
    return calib or None


def write_calibration_file(calib: dict[str, float], path: Path) -> None:
    """Write calibration as key=value text file."""
    with open(path, "w") as f:
        for key, val in calib.items():
            f.write(f"{key}={val:.6f}\n")
