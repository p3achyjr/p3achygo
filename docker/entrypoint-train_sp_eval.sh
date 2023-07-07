#!/bin/bash

# GOOGLE_APPLICATION_CREDENTIALS must be mounted and set.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../.."
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
python -m python.rl_loop.train_sp_eval \
  --sp_bin_path=/app/bazel-bin/cc/selfplay/main \
  --eval_bin_path=/app/bazel-bin/cc/eval/main \
  --run_id=$RUN_ID
