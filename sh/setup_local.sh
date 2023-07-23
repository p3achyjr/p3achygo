#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../.."

docker build -f docker/Dockerfile-base_local -t p3achygo_base-local:latest .
docker build \
  --build-arg p3achygo_base-local:latest \
  -f docker/Dockerfile-shuffler \
  -t p3achygo_shuffler:latest .
docker build \
  --build-arg p3achygo_base-local:latest \
  -f docker/Dockerfile-train_sp_eval \
  -t p3achygo_train-sp-eval:latest
