#!/bin/bash
set -euo pipefail

REMOTE="s3://p3achygo/v4/"
DEFAULT_LOCAL="/p3achygo-data/v4/"

usage() {
  echo "Usage: $0 [push|pull|pull-goldens] [local_path]"
  echo "  push          - sync local -> s3 (default)"
  echo "  pull          - sync s3 -> local"
  echo "  pull-goldens  - sync only s3 goldens/ -> local"
  echo "  local_path defaults to $DEFAULT_LOCAL"
  exit 1
}

DIRECTION="${1:-push}"
LOCAL="${2:-$DEFAULT_LOCAL}"

case "$DIRECTION" in
  push)
    SRC="$LOCAL"
    DST="$REMOTE"
    echo "[$(date)] Creating bucket if not exists..."
    s5cmd mb "$REMOTE" 2>/dev/null || true
    ;;
  pull)
    SRC="$REMOTE*"
    DST="$LOCAL"
    mkdir -p "$LOCAL"
    ;;
  pull-goldens)
    SRC="${REMOTE}goldens/*"
    DST="${LOCAL}goldens/"
    mkdir -p "$DST"
    ;;
  *) usage ;;
esac

echo "[$(date)] Syncing $SRC -> $DST ..."
s5cmd --log debug sync "$SRC" "$DST"
echo "[$(date)] Sync complete."
