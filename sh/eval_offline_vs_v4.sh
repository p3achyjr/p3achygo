#!/usr/bin/env bash
# v4 main run (cur) vs an offline run (cand), same gen.
# v4 side: builds a fresh TRT engine from the existing v4 ONNX export
# (model_0GEN_b96.trt, leaving any older model_0GEN.trt untouched).
# cand side: exports ONNX from the .pt if missing, then builds the engine.
# Usage: ./sh/eval_offline_vs_v4.sh <run_id> [gen=200] [num_games=100] [visits=400] [tag]
set -euo pipefail
cd "$HOME/p3achygo"

RUN_ID="${1:?run_id}"
GEN="${2:-200}"
NUM_GAMES="${3:-100}"
VISITS="${4:-400}"
TAG="${5:-v4-0${GEN}-vs-${RUN_ID}-0${GEN}-n$VISITS}"

DATA="$HOME/p3achygo-data"
CUR_ONNX="$DATA/v4-models/main/_onnx/model_0$GEN.onnx"
CUR_TRT="$DATA/v4-models/main/_onnx/model_0${GEN}_b96.trt"
CAND_DIR="$DATA/$RUN_ID/models"
CAND_PT="$CAND_DIR/model_0$GEN.pt"
CAND_ONNX="$CAND_DIR/_onnx/model_0$GEN.onnx"
CAND_TRT="$CAND_DIR/_onnx/model_0$GEN.trt"
OUT="$DATA/eval/$TAG"

if [ -e "$OUT.log" ]; then
  echo "refusing to overwrite existing $OUT.log; pass a different tag" >&2
  exit 1
fi
[ -e "$CUR_ONNX" ] || { echo "missing v4 onnx $CUR_ONNX" >&2; exit 1; }
[ -e "$CAND_PT" ] || { echo "missing cand checkpoint $CAND_PT" >&2; exit 1; }

if [ ! -e "$CUR_TRT" ]; then
  ./build/build_and_run_trt_engine --onnx_path="$CUR_ONNX" --batch_size=96 \
    --engine_name="$(basename "$CUR_TRT")" --ds_path="$DATA/v4/val.tfrecord.zz"
fi
if [ ! -e "$CAND_ONNX" ]; then
  PYTHONPATH=python python -m python.scripts.torch_to_onnx --model_path="$CAND_PT" --fp16
fi
if [ ! -e "$CAND_TRT" ]; then
  ./build/build_and_run_trt_engine --onnx_path="$CAND_ONNX" --batch_size=96 \
    --ds_path="$DATA/v4/val.tfrecord.zz"
fi
mkdir -p "$OUT"

./build/eval \
  --cur_model_path="$CUR_TRT" \
  --cand_model_path="$CAND_TRT" \
  --cache_size=32768 \
  --num_games="$NUM_GAMES" \
  --cur_n="$VISITS"  --cur_use_puct=1  --cur_use_lcb=1  --cur_use_bias_cache=true  --cur_p_opt_weight=1.0 \
  --cand_n="$VISITS" --cand_use_puct=1 --cand_use_lcb=1 --cand_use_bias_cache=true --cand_p_opt_weight=1.0 \
  --recorder_path="$OUT" \
  --res_write_path="$OUT.txt" \
  --id=0 2>&1 | tee "$OUT.log"
