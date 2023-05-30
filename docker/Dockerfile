FROM nvcr.io/nvidia/tensorflow:23.01-tf2-py3

RUN apt update && apt-get install -y \
  apt-transport-https \
  ca-certificates \
  git \
  gnupg \
  lsb-release \
  software-properties-common \
  wget

# Install google-cloud-sdk.
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
  | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y


# Install LLVM.
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 12
RUN rm llvm.sh

COPY requirements.txt /app

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Install Bazel.
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64
RUN chmod +x bazelisk-linux-amd64
RUN cp bazelisk-linux-amd64 /usr/local/lib/bazel/bin/bazel
RUN cp bazelisk-linux-amd64 /usr/bin/bazel
RUN cp bazelisk-linux-amd64 /usr/local/bin/bazel
RUN rm bazelisk-linux-amd64

COPY . /app

WORKDIR /app
