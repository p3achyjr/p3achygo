#!/bin/bash

python -m python.rl_loop.shuffle \
  --bin_path=/app/bazel-bin/cc/shuffler/main \
  --run_id=$1 \
  --local_run_dir=$2
