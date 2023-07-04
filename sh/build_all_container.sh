#!/bin/bash

PLUGIN_PATH="/usr/lib/gcc/x86_64-linux-gnu/9/liblto_plugin.so"
BUILD_ARGS="--config=opt --config=trt --linkopt=-lto-O2 --linkopt=-plugin-opt=$PLUGIN_PATH"

bazel build $BUILD_ARGS //cc/selfplay:main
bazel build $BUILD_ARGS //cc/eval:main
bazel build $BUILD_ARGS //cc/shuffler:main
