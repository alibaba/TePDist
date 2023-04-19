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

import numpy as np
import tensorflow as tf

tf.compat.v1.enable_resource_variables()

# batch size
batch_size = 6400

# output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input_data = tf.placeholder(tf.float32, [batch_size, image_height, image_width, channels])
true_out = tf.placeholder(tf.float32, [batch_size, 10])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, channels, k_output], seed=21123))

# Apply Convolution
conv_layer = tf.nn.conv2d(input_data, weight, strides=[1, 2, 2, 1], padding='SAME')
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
flatten = tf.reshape(conv_layer, [-1, 1600])

fc_var = tf.Variable(tf.truncated_normal([1600, 10], seed=123123))
logits = tf.matmul(flatten, fc_var)
cost = tf.losses.softmax_cross_entropy(true_out, logits)

opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = opt.minimize(cost)

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

input_x=np.random.randn(batch_size, image_height, image_width, channels)
input_y=np.eye(batch_size, 10)

with tf.Session(server.target, config=session_config) as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(5):
        ret_d, _ = sess.run([cost, train_op], feed_dict={input_data : input_x, true_out : input_y})
        print(ret_d)
