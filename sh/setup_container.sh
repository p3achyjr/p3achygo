#!/bin/bash

pip install --upgrade pip && pip install -r /app/requirements.txt

cp /usr/local/lib/python3.8/dist-packages/tensorflow/libtensorflow_cc.so.2 /app/cc/tensorflow/ && \
  cp /usr/local/lib/python3.8/dist-packages/tensorflow/libtensorflow_framework.so.2 /app/cc/tensorflow/ && \
  cp -r /usr/local/lib/python3.8/dist-packages/tensorflow/include /app/cc/tensorflow/
