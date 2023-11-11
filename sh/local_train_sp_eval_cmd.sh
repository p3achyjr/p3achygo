#!/bin/bash

python -m python.rl_loop.train_sp_eval --bin_dir=/app/bazel-bin \
  --run_id=$1 2>&1 --local_run_dir=$2 | tee /tmp/sp.log
