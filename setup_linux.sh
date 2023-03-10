#!/bin/bash

sudo apt-get update
sudo apt install wget

# Install LLVM
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 12

# Install Python
sudo apt install python-pip3

# Install Bazel
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/bin/bazel

# Add stuff to ~/.bashrc
echo 'export PATH="/usr/lib/llvm-12/bin:/home/axlui/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PYTHONPATH="/home/axlui/p3achyGo/python:$PATH"' >> ~/.bashrc
echo 'export ASAN_SYMBOLIZER_PATH="/usr/local/opt/llvm/bin/llvm-symbolizer"' >> ~/.bashrc

echo 'alias python="python3"' >> ~/.bashrc
echo 'alias pip="pip3"' >> ~/.bashrc
