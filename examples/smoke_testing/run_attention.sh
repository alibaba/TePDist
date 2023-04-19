#!/usr/bin/env bash
# coding=utf-8
# Copyright (c) 2022 Alibaba PAI team.
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

cur_path=`pwd`
echo "cur_path:", $cur_path
export PYTHONPATH=$cur_path:$PYTHONPATH

# Change SERVER_PORT setting same to the server. 
SERVER_PORT=${SERVER_PORT:-2222} \
TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_lazy_compilation=false --tf_xla_cpu_global_jit=true --tf_xla_clustering_debug" \
CUDA_VISIBLE_DEVICES= \
python3 attention.py
