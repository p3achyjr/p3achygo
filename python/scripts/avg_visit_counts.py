#!/usr/bin/env python3
"""Compute average visits-per-move across all .visit_count files in a folder."""

import argparse
import re
import sys
from pathlib import Path


def parse_visit_count_file(path: Path) -> dict[str, int] | None:
    trainable = None
    fast = None
    for line in path.read_text().splitlines():
        m = re.match(r"Visits Per Trainable Move:\s+(\d+)", line)
        if m:
            trainable = int(m.group(1))
        m = re.match(r"Visits Per Fast Move:\s+(\d+)", line)
        if m:
            fast = int(m.group(1))
    if trainable is None or fast is None:
        return None
    return {"trainable": trainable, "fast": fast}


def main():
    parser = argparse.ArgumentParser(
        description="Report average visits-per-move from .visit_count files."
    )
    parser.add_argument("folder", help="Folder containing .visit_count files")
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Search recursively"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    pattern = "**/*.visit_count" if args.recursive else "*.visit_count"
    files = sorted(folder.glob(pattern))

    if not files:
        print(f"No .visit_count files found in {folder}", file=sys.stderr)
        sys.exit(1)

    trainable_vals = []
    fast_vals = []
    skipped = 0

    for f in files:
        result = parse_visit_count_file(f)
        if result is None:
            skipped += 1
            continue
        trainable_vals.append(result["trainable"])
        fast_vals.append(result["fast"])

    n = len(trainable_vals)
    print(f"Files parsed: {n}" + (f"  (skipped: {skipped})" if skipped else ""))
    print(f"Avg visits per trainable move: {sum(trainable_vals) / n:.1f}")
    print(f"Avg visits per fast move:      {sum(fast_vals) / n:.1f}")


if __name__ == "__main__":
    main()
