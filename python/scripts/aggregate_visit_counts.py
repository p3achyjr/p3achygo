#!/usr/bin/env python3
"""
Aggregate .visit_count files across a gen range.

Usage:
  python aggregate_visit_counts.py <dir> [--gen-start N] [--gen-end N]

Filename format: gen{NNN}_b{NNN}_g{NNN}_n{NNNNN}_t{NNNNNN}_{HASH}.visit_count
"""

import argparse
import os
import re
import sys
from pathlib import Path


FILENAME_RE = re.compile(r"^gen(\d+)_")
FIELD_RE = re.compile(r"^(.+?):\s+(\d+)")


def parse_gen(filename: str) -> int | None:
    m = FILENAME_RE.match(filename)
    return int(m.group(1)) if m else None


def parse_visit_count_file(path: Path) -> dict[str, int]:
    fields = {}
    with open(path) as f:
        for line in f:
            m = FIELD_RE.match(line.strip())
            if m:
                fields[m.group(1)] = int(m.group(2))
    return fields


def main():
    parser = argparse.ArgumentParser(description="Aggregate .visit_count files")
    parser.add_argument("dir", help="Directory containing .visit_count files")
    parser.add_argument("--gen-start", type=int, default=None, help="Minimum gen (inclusive)")
    parser.add_argument("--gen-end", type=int, default=None, help="Maximum gen (inclusive)")
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(directory.glob("*.visit_count"))
    if not files:
        print("No .visit_count files found.", file=sys.stderr)
        sys.exit(1)

    totals: dict[str, int] = {}
    file_count = 0
    gen_min = gen_max = None

    for path in files:
        gen = parse_gen(path.name)
        if gen is None:
            continue
        if args.gen_start is not None and gen < args.gen_start:
            continue
        if args.gen_end is not None and gen > args.gen_end:
            continue

        fields = parse_visit_count_file(path)
        for key, val in fields.items():
            totals[key] = totals.get(key, 0) + val
        file_count += 1
        if gen_min is None or gen < gen_min:
            gen_min = gen
        if gen_max is None or gen > gen_max:
            gen_max = gen

    if file_count == 0:
        print("No files matched the gen range.")
        sys.exit(0)

    gen_range_str = f"gen {gen_min}" if gen_min == gen_max else f"gen {gen_min}–{gen_max}"
    print(f"Files: {file_count}  ({gen_range_str})")
    print()

    # Print raw totals
    for key, val in totals.items():
        print(f"{key}: {val:,}")

    # Recompute derived per-move ratios if the base fields are present
    trainable_visits = totals.get("Trainable Visits")
    trainable_moves = totals.get("Trainable Moves")
    fast_visits = totals.get("Fast Visits")
    fast_moves = totals.get("Fast Moves")

    if trainable_visits and trainable_moves:
        print(f"\nVisits Per Trainable Move (aggregate): {trainable_visits / trainable_moves:,.1f}")
    if fast_visits and fast_moves:
        print(f"Visits Per Fast Move (aggregate):      {fast_visits / fast_moves:,.1f}")


if __name__ == "__main__":
    main()
