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
from tensorflow.python.layers.base import Layer
# For annotated rule based strategy only.
# from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding

tf.compat.v1.enable_resource_variables()

class SelfAttention(Layer):
    def __init__(self, config, **kwargs):
        self.use_bias = kwargs.pop("use_bias", True)
        super(SelfAttention, self).__init__(**kwargs)
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = config["attention_head_size"]
        self.initializer = tf.ones_initializer

    def build(self, input_shape):
        self.q_head_weight = self.add_weight(
            shape=(self.hidden_size, self.attention_head_size * self.num_attention_heads),
            initializer=self.initializer,
            dtype=tf.float32,
            name='query/kernel',
        )
        self.k_head_weight = self.add_weight(
            shape=(self.hidden_size, self.attention_head_size * self.num_attention_heads),
            initializer=self.initializer,
            dtype=tf.float32,
            name='key/kernel',
        )
        self.v_head_weight = self.add_weight(
            shape=(self.hidden_size, self.attention_head_size * self.num_attention_heads),
            initializer=self.initializer,
            dtype=tf.float32,
            name='value/kernel',
        )
        if self.use_bias:
            self.q_head_bias = self.add_weight(
                shape=(self.attention_head_size * self.num_attention_heads,),
                initializer=self.initializer,
                dtype=tf.float32,
                name='query/bias',
            )
            self.k_head_bias = self.add_weight(
                shape=(self.attention_head_size * self.num_attention_heads,),
                initializer=self.initializer,
                dtype=tf.float32,
                name='key/bias',
            )
            self.v_head_bias = self.add_weight(
                shape=(self.attention_head_size * self.num_attention_heads,),
                initializer=self.initializer,
                dtype=tf.float32,
                name='value/bias',
            )
        super(SelfAttention, self).build(input_shape)

    def _abs_attn_core(self, q_head, k_head, v_head, attn_mask, training,
                       scale):
        attn_score = tf.einsum('bind,bjnd->bnij', q_head, k_head)
        attn_score = tf.multiply(attn_score, scale)

        attn_mask = tf.expand_dims(attn_mask, axis=[1])
        adder = (1.0 - tf.cast(attn_mask, tf.float32)) * -10000.0
        attn_score += adder

        attn_prob = tf.nn.softmax(attn_score)
        attn_prob = tf.nn.dropout(attn_prob, 0.01)

        attn_vec = tf.einsum('bnij,bjnd->bind', attn_prob, v_head)
        return attn_vec

    def call(self, attention_input, attention_mask, kv=None, training=False):

        q_input = attention_input
        if kv is None:
            k_input = attention_input
            v_input = attention_input
        else:
            k_input = v_input = kv

        batch_size = tf.shape(attention_mask)[0]
        q_seq_length = tf.shape(attention_mask)[1]
        kv_seq_length = tf.shape(attention_mask)[2]

        q_head_h = tf.einsum('bih,hx->bix', q_input, self.q_head_weight)
        if self.use_bias:
            q_head_h = tf.nn.bias_add(q_head_h, self.q_head_bias)

        k_head_h = tf.einsum('bih,hx->bix', k_input, self.k_head_weight)
        if self.use_bias:
            k_head_h = tf.nn.bias_add(k_head_h, self.k_head_bias)

        v_head_h = tf.einsum('bih,hx->bix', v_input, self.v_head_weight)
        if self.use_bias:
            v_head_h = tf.nn.bias_add(v_head_h, self.v_head_bias)

        q_head_h = tf.reshape(q_head_h, [batch_size, q_seq_length, self.num_attention_heads, self.attention_head_size])
        k_head_h = tf.reshape(k_head_h, [batch_size, kv_seq_length, self.num_attention_heads, self.attention_head_size])
        v_head_h = tf.reshape(v_head_h, [batch_size, kv_seq_length, self.num_attention_heads, self.attention_head_size])

        scale = 1 / (self.attention_head_size ** 0.5)
        attn_vec = self._abs_attn_core(q_head_h, k_head_h, v_head_h, attention_mask, training, scale)
        attn_vec = tf.reshape(attn_vec, [batch_size, q_seq_length, self.attention_head_size * self.num_attention_heads])
        # For annotated rule based strategy, make tensor parallel annotation.
        # attn_vec = xla_sharding.split(attn_vec, split_dimension=2, num_devices=2)
        return attn_vec

config = {
    "vocab_size": 21128,
    "hidden_size": 64,
    "train_batch_size": 16,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "max_sequential": 32,
    "num_attention_heads": 8,
    "attention_head_size": 64,
    "dropout_prob": 0.0,
    "is_training": True
}


attention = SelfAttention(config)
batch_size = config["train_batch_size"]
seq_length = config["max_sequential"]
hidden_size = config["hidden_size"]
input_mask = tf.ones([batch_size, seq_length, seq_length])
input_data = tf.reshape(tf.range(0, batch_size, dtype=tf.float32), [batch_size, 1, 1]) * \
             tf.random.uniform([], 0.1, 0.11, dtype=tf.float32, seed=123123)
input_data = tf.broadcast_to(input_data, [batch_size, seq_length, hidden_size])

# For annotated rule based strategy, make data parallel on input.
# input_data = xla_sharding.split(input_data, split_dimension=0, num_devices=2)

loss = tf.reduce_mean(attention(input_data, input_mask))
opt = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = opt.minimize(loss)

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
    for _ in range(10):
        loss_ret, _ = sess.run([loss, train_op])
        print ("loss: ", loss_ret)

