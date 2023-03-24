# Welcome to TePDist

## Introduction

### Overview

TePDist (TEnsor Program DISTributed) is an automatic distributed training system infrastucture for DL models, not just an algorithm.

TePDist system operates in a client/server mode. The client is supposed to be any front-end that can generate XLA HLO. The server is responsible for distributed strategy planning and automatic distributed task launching. The motivation behind decoupling the client and server is to facilitate future integration with different front-end frameworks. TePDist has its own runtime graph and task scheduler for distributed running.

TePDist system is now developing upon an previous version of community TensorFlow. A sub-module linked to the origin code version is set in this repository for reference. We plan to migrate the codes to much newer community version soon.


### Features

**TePDist has chosen HLO as the input IR for distributed strategy planning.** The largest models we have seen contain tens of thousands of HLO instructions. Our system can easily handle this scale. At the HLO level, the connections between instructions are sparse. Most instructions read only one or two other instructions although the instructions could be more than then thousands. For SPMD strategy exploration, the cost of distributed communication comes from the connections between these instructions. The sparsity of the connections give TePDist the opportunity in exploring strategies on HLO.

**TePDist's distributed strategy exploration is fully automated.** TePDist's automatic planned strategies can cover all kinds of current knowning parallel schems, such as Data parallel (including token parallel), Model parallel (e.g, sharding or Zero) and Pipeline parallel. Of course, TePDist also allows users to intervene in strategy exploration through annotation.

**TePDist has reasonably decomposed the strategy exploration problem.** TePDist uses various methods to decompose the strategy exploration problem into optimization sub-problems and use various algorithms to solve them separately, effectively managing the complexity of the problem. In summary, TePDist partitions the entire graph into subgraphs based on critical nodes (see the paper for more details). Within the subgraph, cones are further partitioned. Between subgraphs, dynamic programming algorithms are used, while ILP algorithms are used between cones within the subgraph.

**The pipeline stage automatic partitioning is quite distinctive.** Before partitioning the stage, there is no need to arrange the DAG into a topological linear sequence. TePDist model the stage partitioning as an ILP problem and use an ILP Solver to automatically find the cutting scheme with the smallest amount of communication.


### Tutorials
\<To put the video link here\>


## Environment Preparing

Before using TePDist, we need to prepare its running environment, which is composed by one or more RPC server(s) and a model building and running frondend client. The RPC server is developed based on the XLA infrastructure. The client we currently choose is a modified TensorFlow version with XLA support enabled. Pytorch support is also under developing.

To prepare such environment, we could follow two ways introduced below. The 'docker' tool should be installed onto the physical machine OS. Since we are supposed to use GPU as training resources, the CUDA driver and nvidia-docker tool should also be installed for convienence.

### A. Download the prebuild docker (will update later).

The prebuild docker installed with TePDist server and client could be download from the Docker Hub:

> \<To put the Docker pull address later\>

### B. Build the developing docker and construct TePDist runtime from source code.

At this time, building TePDist depends on the basic building framework driven by 'Bazel', it would be installed automatically by the docker build scripts.

- 1. Checkout the codes.
```shell
$ git clone https://github.com/alibaba/TePDist.git
```
> If we are building the package on an A100 server or under CUDA 11+ environment, we need to use the 'Dockerfile.cuda11.4.ubuntu18.04' docker file and the build scripts with '\_A100' postfix during next steps.

- 2. Start a devoloping docker and attach it like the commands below.
```shell
[host]$ cd TePDist/docker
[host]$ sudo docker build -t tepdist_image -f Dockerfile.cuda10.1.ubuntu18.04 .
[host]$ sudo nvidia-docker run --net=host --ipc=host -it --name tepdist_dev \
            -v <path_with_tepdist_src_codes>:/root tepdist_image:latest /bin/bash
```

- 3. Compile and install the TF wheel package as the front-end client for TePDist.
```shell
[docker]$ cd <tepdist_path_in_docker>/tf_tepdist
[docker]$ ./build_tensorflow_wheel
    ... After long time building ...
[docker]$ pip3 install tensorflow_whl/<wheel_package_name>
```

- 4. Compile the server binary.
```shell
[docker]$ cd <tepdist_path_in_docker>/tf_tepdist
[docker]$ ./build_xla_service
```

If no errors were found during above steps, we could try to run the basic examples.


## Running TePDist

### Server Starting

We need to build a json file to describe the running resouces, both under single machine case or multi-node cluster case. The templates are put under 'tf\_tepdist' directory. Just chose one and modify it like below:
> Suppose the config name is 'one\_node.json'.
```json
{
  "master" : {
    "ip" : "localhost",
    "port" : "2222",
    "gpu_ids" : "1,2"
  }
}
```

Then we could use the 'launch\_worker.sh' script to start the TePDist server(s), by calling it with correct argument on (inside) each machine (docker instance), like this:

```shell
[docker]$ cd <tepdist_path_in_docker>
[docker]$ ./launch_worker.sh one_node.json 0 >& server_0.log
```
> The first argument is the cluster specification json file. The second is the task index. The master server is 0 and other slave servers should be 1 to N.

If we see some log information (from 'server\_0.log' due to the command above) like below, it means the server was successfully started.
```shell
YYYY-MM-DD HH:mm:ss.uuuuuu: I tensorflow/core/platform/profile_utils/cpu_utils.cc:XXX] CPU Frequency: 2499450000 Hz
YYYY-MM-DD HH:mm:ss.uuuuuu: I tensorflow/compiler/xla/service/service.cc:XXX] XLA service 0x7f5c14b88640 initialized for platform Host (this does not gu     arantee that XLA will be used). Devices:
YYYY-MM-DD HH:mm:ss.uuuuuu: I tensorflow/compiler/xla/service/service.cc:XXX]   StreamExecutor device (0): Host, Default Version
YYYY-MM-DD HH:mm:ss.uuuuuu: I tensorflow/compiler/xla/rpc/grpc_service_gpu.cc:XXX] Server listening on localhost:2222
```

### Studying Examples

To run the basic cases of TePDist, please check the 'examples' directory. We provide **Wide-Resnet, GPT2 and MoE** model examples. Running the corresponding 'run.sh' script in the developing docker would start the client.

Before doing this, we need to make sure the TePDist server is already running and the correct master IP and port is passed by the 'SERVER\_IP' and 'SERVER\_PORT' environ var by client starting scripts.

Besides these, three minor cases are also put into 'smoke\_testing' directory, for helping us faster understand the TePDist system running process.


### Trying Self-Build Models

Now we support using TensorFlow as the front-end to build our customized models. The 'smoke\_testing' examples are also showing how to build and run customized models and utilize the TePDist automatic distributed training facility. To be noted that, at this time, we need to set some options while using TePDist.
```python
  # We create and pass a ConfigProto object to the Session or Estimator which
  # runs the model, from the client side.
  session_config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(allow_growth=True,
                              force_gpu_compatible=True,
                              per_process_gpu_memory_fraction=1.0))

  # Set off remapping and memory_optimization to avoid bad cases.
  from tensorflow.core.protobuf import rewriter_config_pb2
  off = rewriter_config_pb2.RewriterConfig.OFF
  session_config.graph_options.rewrite_options.remapping = off
  session_config.graph_options.rewrite_options.memory_optimization = off

  # Disable of enable (= on) distributed variable initialization.
  session_config.graph_options.rewrite_options.init_from_remote = off

  # For large models, disable grappler optimizers timeout check
  session_config.graph_options.rewrite_options.meta_optimizer_timeout_ms = -1
```


## Papers and Reports

Please cite us by using the following BibTeX entry.
```latex
@ARTICLE{2023arXiv230208141Z,
       author = {{Zhang}, Shiwei and {Diao}, Lansong and {Wang}, Siyu and {Cao}, Zongyan and {Gu}, Yiliang and {Si}, Chang and {Shi}, Ziji and {Zheng}, Zhen and {Wu}, Chuan and {Lin}, Wei},
        title = "{Auto-Parallelizing Large Models with Rhino: A Systematic Approach on Production AI Platform}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Distributed, Parallel, and Cluster Computing, Computer Science - Machine Learning, Computer Science - Programming Languages},
         year = 2023,
        month = feb,
          eid = {arXiv:2302.08141},
          doi = {10.48550/arXiv.2302.08141},
archivePrefix = {arXiv},
       eprint = {2302.08141},
 primaryClass = {cs.DC},
}
```
