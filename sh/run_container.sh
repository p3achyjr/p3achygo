#!/bin/bash

docker run -it --name p3achygo --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --privileged --security-opt seccomp=unconfined \
    -v /home/p3achyjr/p3achygo-data:/p3achygo-data \
    -v ${PWD}:/app \
    --cap-add=CAP_SYS_ADMIN p3achygo-dev:latest 
