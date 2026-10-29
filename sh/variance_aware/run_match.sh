#!/usr/bin/env bash
# Exp 2/3: eval match, cur = config/eval/variance-aware/base.cfg (live PUCT,
# fresh tree every move) vs cand = given cfg, same model_0631 both sides.
# Runs the listed cand cfgs sequentially on one GPU, detached.
# Usage: ./sh/variance_aware/run_match.sh <gpu> <num_games> <cand_cfg_name>...
#   e.g. ./sh/variance_aware/run_match.sh 2 200 se_e1_x0.5 se_e1_x1 se_e1_x2
set -euo pipefail
GPU="$1"; NUM_GAMES="$2"; shift 2
WT=/root/p3achygo-exp
DATA=/root/p3achygo-data
OUT="$DATA/exp/variance-aware"
CFG="$WT/config/eval/variance-aware"
TRT="$DATA/v4-models/main/_onnx/model_0631_b${NUM_GAMES}.trt"
[ -e "$TRT" ] || { echo "missing engine $TRT (batch must equal num_games)" >&2; exit 1; }
mkdir -p "$OUT"
for c in "$@"; do
  [ -e "$CFG/$c.cfg" ] || { echo "missing $CFG/$c.cfg" >&2; exit 1; }
  [ -e "$OUT/match_$c.log" ] && { echo "refusing to overwrite $OUT/match_$c.log" >&2; exit 1; }
done
cd "$WT"
if [ "${VA_INNER:-}" != "1" ]; then
  VA_INNER=1 setsid nohup "$0" "$GPU" "$NUM_GAMES" "$@" > "$OUT/match_launcher_gpu$GPU.log" 2>&1 < /dev/null &
  echo "launched matches [$*] on GPU $GPU -> $OUT/match_<cfg>.log"
  exit 0
fi
for c in "$@"; do
  mkdir -p "$OUT/match_$c"
  CUDA_VISIBLE_DEVICES="$GPU" ./build/eval \
    --cur_model_path="$TRT" --cand_model_path="$TRT" \
    --cur_config="$CFG/base.cfg" --cand_config="$CFG/$c.cfg" \
    --cache_size=32768 --num_games="$NUM_GAMES" \
    --recorder_path="$OUT/match_$c" --res_write_path="$OUT/match_$c.txt" \
    --id=0 > "$OUT/match_$c.log" 2>&1 < /dev/null || echo "match $c failed" >> "$OUT/match_failures.log"
done
