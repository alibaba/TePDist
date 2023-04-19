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

import tensorflow.compat.v1 as tf
# For annotated rule based strategy only.
# from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding

tf.compat.v1.enable_resource_variables()

a = tf.Variable([[0.2] * 16, [0.4] * 16, [0.6] * 16, [0.8] * 16], name="a")
# For annotated rule based strategy, make tensor parallel annotation.
# a = xla_sharding.split(a, split_dimension=1, num_devices=2)
b = tf.Variable([[0, 1, 2, 3], [3, 2, 1, 0]] * 8, dtype=tf.float32, name="b")
c = tf.matmul(a, b)
d = tf.reduce_sum(tf.nn.softmax(c))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = opt.minimize(d)

server = tf.train.Server.create_local_server()

session_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    gpu_options=tf.GPUOptions(allow_growth=True,
                              force_gpu_compatible=True,
                              per_process_gpu_memory_fraction=1.0))

from tensorflow.core.protobuf import rewriter_config_pb2
off = rewriter_config_pb2.RewriterConfig.OFF
session_config.graph_options.rewrite_options.remapping = off
session_config.graph_options.rewrite_options.memory_optimization = off
session_config.graph_options.rewrite_options.init_from_remote = off
session_config.graph_options.rewrite_options.meta_optimizer_timeout_ms = -1

with tf.Session(server.target, config=session_config) as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(5):
        res = sess.run([d, train_op])
        print(res)
