#!/usr/bin/env bash
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

train_batch_size=4
optimizer=adam

communication_fp16=False
dist_hlo_baseline=1

time_str=`date +%m%d_%H%M%S`
LOGNAME=run${time_str}_gpt_BS${train_batch_size}.log

# Change SERVER_PORT setting same to the server. 
SERVER_PORT=${SERVER_PORT:-2222} \
TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_lazy_compilation=false --tf_xla_cpu_global_jit=true --tf_xla_clustering_debug" \
CUDA_VISIBLE_DEVICES= \
python3 pretrain_gpt_moe.py \
    --train_batch_size=${train_batch_size} \
    --config=./pretrain_gpt.json \
    --mode=train \
    --optimizer=${optimizer} \
    --stop_at_step=10 \
    --dist_hlo_baseline=${dist_hlo_baseline} \
    --log_every_step=1 \
    --communication_fp16=${communication_fp16} \
    --fake_input=True \
    2>&1 | tee ${LOGNAME}
