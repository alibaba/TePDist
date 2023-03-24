#!/usr/bin/env bash
# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e
ubuntu_version=$(cat /etc/issue | grep -i ubuntu | awk '{print $2}' | \
  awk -F'.' '{print $1}')

if [[ "$1" != "" ]] && [[ "$1" != "--without_cmake" ]]; then
  echo "Unknown argument '$1'"
  exit 1
fi

# For China mainland users, using Aliyun or Huawei Cloud mirror for Ubuntu.
# sed -i 's/[a-zA-Z]*.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
sed -i 's/[a-zA-Z]*.ubuntu.com/repo.huaweicloud.com/g' /etc/apt/sources.list

# Install dependencies from ubuntu deb repository.
apt-key adv --keyserver keyserver.ubuntu.com --recv 084ECFC5828AB726
apt-get update

apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    coinor-libcbc-dev \
    curl \
    git \
    jq \
    libcurl4-openssl-dev \
    libtool \
    libssl-dev \
    lsof \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-setuptools \
    rsync \
    sudo \
    swig \
    tzdata \
    unzip \
    vim \
    wget \
    zip \
    zlib1g-dev

apt-get clean
rm -rf /var/lib/apt/lists/*
ln -sf python3 /usr/bin/python

# Set up time zone.
export TIME_ZONE=Asia/Shanghai
ln -sf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime

# For China mainland users, change to Aliyun or Huawei Clould pip source.
# pip3 install --upgrade -i http://mirrors.aliyun.com/pypi/simple/ pip
# pip3 config set global.index-url http://mirrors.aliyun.com/pypi/simple/
pip3 install --upgrade -i https://repo.huaweicloud.com/repository/pypi/simple/ pip
pip3 config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple/

pip3 install wheel
pip3 install --upgrade setuptools==39.1.0
pip3 install --upgrade six==1.12.0
pip3 install --upgrade absl-py
pip3 install --upgrade protobuf==3.19.6
pip3 install --upgrade numpy==1.19.5
pip3 install --upgrade scipy==1.4.1
pip3 install --upgrade h5py
pip3 install scikit-learn==0.24.1
pip3 install keras-preprocessing
pip3 install pandas==1.1.5
pip3 install packaging
pip3 install psutil
pip3 install py-cpuinfo
pip3 install requests
pip3 install regex
pip3 install sentencepiece

# Select bazel version.
BAZEL_VERSION="2.0.0"

set +e
local_bazel_ver=$(bazel version 2>&1 | grep -i label | awk '{print $3}')

if [[ "$local_bazel_ver" == "$BAZEL_VERSION" ]]; then
  exit 0
fi

set -e

# Install bazel.
mkdir -p /bazel
cd /bazel
if [[ ! -f "bazel-$BAZEL_VERSION-installer-linux-x86_64.sh" ]]; then
  # Origin URL:
  # curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

  # For China mainland users, please use the URL below:
  curl -fSsL -O https://mirrors.huaweicloud.com/bazel/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
fi
chmod +x /bazel/bazel-*.sh
/bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Enable bazel auto completion.
echo "source /usr/local/lib/bazel/bin/bazel-complete.bash" >> ~/.bashrc
