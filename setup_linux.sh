#!/usr/bin/env bash

sudo apt-get update
sudo apt install wget

# Install LLVM
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 12

# Install Python
sudo apt install python3-pip

# Install TF
sudo pip install tensorflow==2.11.*

# Install Bazel
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/bin/bazel

# Add stuff to ~/.bashrc
grep -qxF 'export PATH="/usr/lib/llvm-12/bin:/home/axlui/.local/bin:$PATH"' ~/bashrc ||
	echo 'export PATH="/usr/lib/llvm-12/bin:/home/axlui/.local/bin:$PATH"' >>~/.bashrc
grep -qxF 'export PYTHONPATH="/home/axlui/p3achygo/python:$PATH"' ~/.bashrc ||
	echo 'export PYTHONPATH="/home/axlui/p3achygo/python:$PATH"' >>~/.bashrc
grep -qxF 'export ASAN_SYMBOLIZER_PATH="/usr/lib/llvm-12/bin/llvm-symbolizer"' ~/.bashrc ||
	echo 'export ASAN_SYMBOLIZER_PATH="/usr/lib/llvm-12/bin/llvm-symbolizer"' >>~/.bashrc
grep -qxF 'export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"' ~/.bashrc ||
	echo 'export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc

# Install mimalloc
RUN git clone https://github.com/microsoft/mimalloc.git && \
  cd mimalloc && \
  mkdir -p out/release && \
  cd out/release && \
  cmake ../.. && \
  make && \
  make install && \
  cd ../../.. && \
  rm -rf mimalloc

source ~/.bashrc
