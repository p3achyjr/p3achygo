#!/usr/bin/env bash
# Exp 1: sigma regret study (prod sigma vs SE-scaled sigma) on model_0631.
# Detached; log + results under $DATA/exp/variance-aware/.
# Usage: ./sh/variance_aware/run_exp1_sigma.sh [gpu=1] [num_examples=300] [skip_first=0] [tag=exp1_sigma]
set -euo pipefail
GPU="${1:-1}"; NUM="${2:-300}"; SKIP="${3:-0}"; TAG="${4:-exp1_sigma}"
WT=/root/p3achygo-exp
DATA=/root/p3achygo-data
OUT="$DATA/exp/variance-aware"
mkdir -p "$OUT"
[ -e "$OUT/$TAG.log" ] && { echo "refusing to overwrite $OUT/$TAG.log" >&2; exit 1; }
cd "$WT"
CUDA_VISIBLE_DEVICES="$GPU" setsid nohup ./build/sigma_regret_study \
  --model_path="$DATA/v4-models/main/_onnx/model_0631_b1.trt" \
  --chunk_path="$DATA/v4/val.tfrecord.zz" \
  --num_examples="$NUM" --skip_first="$SKIP" --verbose=true \
  > "$OUT/$TAG.log" 2>&1 < /dev/null &
echo "launched pid $! -> $OUT/$TAG.log"
