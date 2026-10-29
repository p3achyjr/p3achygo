#!/usr/bin/env bash
# One-time setup on the remote 4x5090 box for the variance-aware experiments.
# Creates a separate git worktree + build dir so /root/p3achygo (another
# agent's checkout) is never touched, builds the binaries, and builds TRT
# engines for model_0631 at batch 1 (studies) and batch 200 (eval matches).
# Usage: ./sh/variance_aware/remote_setup.sh [branch=p3achyjr/variance-aware-exp]
set -euo pipefail

BRANCH="${1:-p3achyjr/variance-aware-exp}"
MAIN=/root/p3achygo
WT=/root/p3achygo-exp
DATA=/root/p3achygo-data
ONNX="$DATA/v4-models/main/_onnx/model_0631.onnx"
VAL="$DATA/v4/val.tfrecord.zz"

[ -e "$ONNX" ] || { echo "missing $ONNX (scp it from local model_cands/_onnx)" >&2; exit 1; }

git -C "$MAIN" fetch origin "$BRANCH"
if [ ! -d "$WT" ]; then
  git -C "$MAIN" worktree add "$WT" "origin/$BRANCH" -b "$BRANCH" 2>/dev/null \
    || git -C "$MAIN" worktree add "$WT" "$BRANCH"
else
  git -C "$WT" pull --ff-only origin "$BRANCH"
fi
cd "$WT"
git log --oneline -1

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_TENSORRT=ON \
  -DBUILD_TESTS=OFF > build/cmake_configure.log 2>&1 || { tail -30 build/cmake_configure.log; exit 1; }
cmake --build build -j64 --target eval sigma_regret_study node_var_dump \
  build_and_run_trt_engine 2>&1 | grep -E "error|Error|Built target (eval|sigma|node|build_and)" || true
ls -la build/eval build/sigma_regret_study build/node_var_dump build/build_and_run_trt_engine

for bs in 1 200; do
  trt="$DATA/v4-models/main/_onnx/model_0631_b${bs}.trt"
  if [ ! -e "$trt" ]; then
    echo "building $trt"
    CUDA_VISIBLE_DEVICES=1 ./build/build_and_run_trt_engine --onnx_path="$ONNX" \
      --batch_size="$bs" --engine_name="$(basename "$trt")" --ds_path="$VAL" \
      > "$DATA/v4-models/main/_onnx/model_0631_b${bs}.trt.log" 2>&1 \
      || { tail -20 "$DATA/v4-models/main/_onnx/model_0631_b${bs}.trt.log"; exit 1; }
  fi
done
ls -la "$DATA"/v4-models/main/_onnx/model_0631_b*.trt
echo "setup done"
