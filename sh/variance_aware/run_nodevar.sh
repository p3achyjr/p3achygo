#!/usr/bin/env bash
# Exp 3 step 1: dump per-node (N, v_var, depth) under the live PUCT config and
# fit s_ref(N) (mean sqrt(v_var) per log2(N) bin) for model_0631.
# Usage: ./sh/variance_aware/run_nodevar.sh [gpu=3] [num_examples=200] [visits=1600] [tag=nodevar_n1600]
set -euo pipefail
GPU="${1:-3}"; NUM="${2:-200}"; VISITS="${3:-1600}"
TAG="${4:-nodevar_n$VISITS}"
WT=/root/p3achygo-exp
DATA=/root/p3achygo-data
OUT="$DATA/exp/variance-aware"
mkdir -p "$OUT"
[ -e "$OUT/$TAG.log" ] && { echo "refusing to overwrite $OUT/$TAG.log" >&2; exit 1; }
cd "$WT"
CUDA_VISIBLE_DEVICES="$GPU" setsid nohup ./build/node_var_dump \
  --model_path="$DATA/v4-models/main/_onnx/model_0631_b1.trt" \
  --chunk_path="$DATA/v4/val.tfrecord.zz" \
  --num_examples="$NUM" --visits="$VISITS" \
  --csv_path="$OUT/$TAG.csv" --s_ref_path="$OUT/$TAG.s_ref.txt" \
  > "$OUT/$TAG.log" 2>&1 < /dev/null &
echo "launched pid $! -> $OUT/$TAG.log"
