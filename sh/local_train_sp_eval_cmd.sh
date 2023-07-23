#!/bin/bash

python -m python.rl_loop.train_sp_eval --sp_bin_path=/app/bazel-bin/cc/selfplay/main \
  --eval_bin_path=/app/bazel-bin/cc/eval/main \
  --run_id=$1 2>&1 --local_run_dir=$2 | tee /tmp/sp_log.txt
