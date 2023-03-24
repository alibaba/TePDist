#!/bin/bash
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
# =============================================================================

JQ_EXEC=$(which jq)

extract_worker_info() {
  if [ $2 -eq 0 ]
  then
    WORKER_JSON=$(${JQ_EXEC} .master $1)

    if [ "${WORKER_JSON}" = "null" ]
    then
      echo "miss master info in cluster spec file!"
      exit
    fi
  elif [ $2 -gt 0 ]
  then
    SLAVE_WORKER_ID=$(($2-1))
    WORKER_JSON=$(${JQ_EXEC} .workers[${SLAVE_WORKER_ID}] $1)

    if [ "${WORKER_JSON}" = "null" ]
    then
      echo "miss workers info in cluster spec file!"
      exit
    fi
  else
    echo "invalid task index!"
    exit
  fi
}

extract_ip_port_gpu() {
  IP=$(echo $1 | ${JQ_EXEC} .ip | sed 's/\"//g')
  PORT=$(echo $1 | ${JQ_EXEC} .port | sed 's/\"//g')
  GPU_IDS=$(echo $1 | ${JQ_EXEC} .gpu_ids | sed 's/\"//g')
}

if [ -z "$1" ]
then
  echo "cluster spec file is not supplied"
  exit
fi

if [ -z "$2" ]
then
  echo "task index is not supplied"
  exit
fi

# cluster spec file name:
CLUSTER_SPEC_NAME=$1
echo $CLUSTER_SPEC_NAME

# worker id:
TASK_INDEX=$2

extract_worker_info $CLUSTER_SPEC_NAME $TASK_INDEX
echo $WORKER_JSON

extract_ip_port_gpu "$WORKER_JSON"
echo $IP, $PORT, $GPU_IDS

if [ "${IP}" = "null" ]
then
  echo "miss ip for worker ${TASK_INDEX} in cluster spec file!"
  exit
fi

if [ "${PORT}" = "null" ]
then
  echo "miss port for worker ${TASK_INDEX} in cluster spec file!"
  exit
fi

if [ "${GPU_IDS}" = "null" ]
then
  echo "miss gpu_ids for worker ${TASK_INDEX} in cluster spec file!"
  exit
fi

NCCL_DEBUG=INFO FRONTEND="TF-1.14" \
CLUSTER_SPEC="$CLUSTER_SPEC_NAME" CUDA_VISIBLE_DEVICES=$GPU_IDS \
nohup bazel-bin/tensorflow/compiler/xla/rpc/grpc_service_gpu --port=$PORT --platform=CPU --ip=$IP --task_index=$TASK_INDEX


