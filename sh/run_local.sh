#!/bin/bash

# Run //sh/setup_local.sh first.
# Assumes Credentials are already set.
CREDENTIALS_PATH = $1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../.."

docker run -v ${CREDENTIALS_PATH}:/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json p3achygo_shuffler:latest
docker run -v ${CREDENTIALS_PATH}:/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json p3achygo_train-sp-eval:latest
