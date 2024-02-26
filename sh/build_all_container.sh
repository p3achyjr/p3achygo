#!/bin/bash

BUILD_ARGS=(
    "--config=opt"
    "--config=mimalloc"
    "--copt=-flto"
    "--linkopt=-flto=O2"
)


bazel build "${BUILD_ARGS[@]}" //cc/selfplay:main
bazel build "${BUILD_ARGS[@]}" //cc/eval:main
bazel build "${BUILD_ARGS[@]}" //cc/shuffler:main
bazel build "${BUILD_ARGS[@]}" //cc/nn/engine/scripts:build_and_run_trt_engine
bazel build "${BUILD_ARGS[@]}" //cc/gtp:main
bazel build "${BUILD_ARGS[@]}" //cc/data:main
