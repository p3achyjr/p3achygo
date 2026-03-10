#!/usr/bin/env bash
# setup_local.sh
# Mirrors docker/Dockerfile-base_local_trt for a native (non-Docker) environment.
# Run as root or with sudo.
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. APT packages
# ---------------------------------------------------------------------------
apt-get update
apt-get install -y \
  apt-transport-https \
  ca-certificates \
  git \
  gnupg \
  lsb-release \
  software-properties-common \
  wget \
  htop \
  curl \
  python3 \
  python3-pip \
  python3-dev \
  libboost-all-dev

# ---------------------------------------------------------------------------
# 2. Google Cloud SDK
# ---------------------------------------------------------------------------
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
  | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update -y
apt-get install -y google-cloud-cli

# ---------------------------------------------------------------------------
# 4. mimalloc
# ---------------------------------------------------------------------------
git clone https://github.com/microsoft/mimalloc.git
cd mimalloc
mkdir -p out/release
cd out/release
cmake ../..
make
make install
cd ../../..
rm -rf mimalloc

# ---------------------------------------------------------------------------
# 6. Environment / PATH updates  ->  /etc/profile.d/local_env.sh
# ---------------------------------------------------------------------------
cat > /etc/profile.d/local_env.sh << 'EOF'
# Added by setup_local.sh

export PYTHONPATH=/app/python${PYTHONPATH:+:$PYTHONPATH}

# mimalloc + local libs
export LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# TensorRT / TensorRT-RTX  (installed under /opt)
export LD_LIBRARY_PATH=/opt/tensorrt-rtx/lib:/opt/tensorrt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/opt/tensorrt-rtx/bin:/opt/tensorrt/bin${PATH:+:$PATH}
export TENSORRT_ROOT=/opt/tensorrt
export TENSORRT_RTX_ROOT=/opt/tensorrt-rtx
EOF

chmod 644 /etc/profile.d/local_env.sh
echo "Environment written to /etc/profile.d/local_env.sh"
echo "Run 'source /etc/profile.d/local_env.sh' or open a new shell to apply."
