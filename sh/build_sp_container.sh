#!/bin/bash

BUILD_ARGS=(
    "--config=dbg"
    "--config=mimalloc"
    "--copt=-flto"
    "--linkopt=-flto=O2"
)

bazel build "${BUILD_ARGS[@]}" //cc/selfplay:main
