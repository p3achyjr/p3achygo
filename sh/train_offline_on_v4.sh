#!/usr/bin/env bash
# Offline trainer: fresh model from config/<RUN_ID>.json, trained on the existing
# v4 golden chunks from R2. No self-play. Resumable: re-running picks up from the
# latest checkpoint in $RUN/models.
# Usage: ./sh/train_offline_on_v4.sh <run_id> [max_gens=280]
#   e.g. ./sh/train_offline_on_v4.sh v4-b10c384tfmr-1x1
set -euo pipefail
cd "$HOME/p3achygo"
RUN_ID="${1:?run_id (config/<run_id>.json)}"
RUN="$HOME/p3achygo-data/$RUN_ID"
mkdir -p "$RUN/models" "$RUN/chunks"

python -m python.rl_loop.train_loop \
  --run_id="$RUN_ID" \
  --source_run_id=v4 \
  --models_dir="$RUN/models" \
  --chunk_dir="$RUN/chunks" \
  --val_ds_path="$HOME/p3achygo-data/v4/val.tfrecord.zz" \
  --batch_num_path="$RUN/batch_num.txt" \
  --max_gens="${2:-280}" 2>&1 | tee -a "$RUN/train.log"
