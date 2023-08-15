#!/bin/bash

BUILD_ARGS=(
    "--config=opt"
    "--config=trt"
    "--config=mimalloc"
    "--copt=-flto"
    "--linkopt=-flto=O2"
)


bazel build "${BUILD_ARGS[@]}" //cc/gtp:main