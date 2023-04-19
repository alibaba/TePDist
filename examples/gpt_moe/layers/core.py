# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import layers as keras_layers
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
from .activations import gelu, relu
from utils import get_initializer
from model import tffloat


def LayerNormalization(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.keras.layers.LayerNormalization(axis = -1)(input_tensor)


class LayerNorm(base.Layer):
    """
    a class of layer normalization
    """

    def __init__(self, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.layer_norm = LayerNormalization

    def call(self, input_tensor):
        output = self.layer_norm(input_tensor)
        return output


class M4LayerNorm(base.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        """
        Construct a layernorm module in the M4 style No bias and no subtraction of mean.
        """
        super(M4LayerNorm, self).__init__(**kwargs)
        self.variance_epsilon = epsilon

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.weight = self.add_weight("weight", shape=(input_shape[-1],), initializer="ones")
        super(M4LayerNorm, self).build(input_shape)

    def call(self, hidden_states):
        variance = tf.math.reduce_mean(tf.math.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * tf.math.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class Embedding(keras_layers.Embedding, base.Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        super(Embedding, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            dtype=tffloat,
            **kwargs)


class Dense(keras_layers.Dense, base.Layer):

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Dense, self).__init__(units=units,
                                    activation=activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint,
                                    trainable=trainable,
                                    name=name,
                                    dtype=tffloat,
                                    **kwargs)


class Dropout(keras_layers.Dropout, base.Layer):
    def __init__(self, rate=0.5,
                 noise_shape=None,
                 seed=123123,
                 name=None,
                 **kwargs):
        super(Dropout, self).__init__(rate=rate,
                                      noise_shape=noise_shape,
                                      seed=seed,
                                      name=name,
                                      **kwargs)

    def call(self, inputs, training=False):
        return super(Dropout, self).call(inputs, training=training)


class dense_dropoutput_layernorm(base.Layer):
    def __init__(self, config, **kwargs):
        super(dense_dropoutput_layernorm, self).__init__(**kwargs)
        self.dense = Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(
                config.initializer_range),
            name="dense")
        self.LayerNorm = LayerNormalization
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(
            hidden_states + input_tensor, name="LayerNorm")
        return hidden_states


class M4DenseReluDense(base.Layer):
    def __init__(self, config, activation='relu', **kwargs):
        super(M4DenseReluDense, self).__init__(**kwargs)
        self.wi = Dense(config.intermediate_size, use_bias=False, kernel_initializer=get_initializer(
            config.initializer_range), name="wi")
        self.wo = Dense(config.hidden_size, use_bias=False, kernel_initializer=get_initializer(
            config.initializer_range), name="wo")
        self.dropout = Dropout(config.hidden_dropout_prob)
        if activation == 'relu':
            self.act = relu
        elif activation == 'gelu':
            self.act = gelu
        else:
            raise ValueError("Unknown activation!")

    def call(self, hidden_states, training=False):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class M4LayerFF(base.Layer):
    def __init__(self, config, **kwargs):
        super(M4LayerFF, self).__init__(**kwargs)
        self.DenseReluDense = M4DenseReluDense(config, name="DenseReluDense")
        self.layer_norm = M4LayerNorm(name="layer_norm")
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, training=False, prenorm=True):
        if prenorm:
            normed_hidden_states = self.layer_norm(hidden_states)
        else:
            normed_hidden_states = hidden_states
        dense_output = self.DenseReluDense(normed_hidden_states, training=training)
        hidden_states = hidden_states + self.dropout(dense_output, training=training)
        if not prenorm:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class MoePointWiseFFN(base.Layer):
    """
    create multiple dense layers as experts, and select one for forward pass.
    args:
    config: configuration
    kwargs: name, etc.

    input:
    inputs: input tensor, bz * len * hidden
    training: whether it is training

    outputs:
    layer_output: output of moe, bz * len * hidden
    """

    def __init__(self, config, **kwargs):
        super(MoePointWiseFFN, self).__init__(**kwargs)
        self.initializer = get_initializer(config.initializer_range)
        # self.inter_experts = [Dense(config.intermediate_size, activation=gelu, kernel_initializer=get_initializer(config.initializer_range), name='inter_expert_{}'.format(i)) for i in range(config.num_experts)]
        # self.out_experts = [Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='out_expert_{}'.format(i)) for i in range(config.num_experts)]
        self.layer_norm = LayerNormalization
        self.num_experts = config.num_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.gate = Dense(
            config.num_experts,
            kernel_initializer=get_initializer(
                config.initializer_range),
            name='gate')

    def build(self, input_shape):
        self.inter_experts = self.add_weight(
            shape=(self.num_experts, self.hidden_size, self.intermediate_size),
            initializer=self.initializer,
            dtype=tffloat,
            name='inter_experts',
        )
        self.out_experts = self.add_weight(
            shape=(self.num_experts, self.intermediate_size, self.hidden_size),
            initializer=self.initializer,
            dtype=tffloat,
            name='out_experts',
        )
        super(MoePointWiseFFN, self).build(input_shape)

    def call(self, inputs, training=False):
        gate = self.gate(inputs)  # bz * len * hidden
        # TODO
        # add softmax for gate, and element-wisely multiply the expert outputs
        val, idx = tf.math.top_k(gate, k=2)
        final_gate = tf.nn.softmax(val)  # bz * len * topk
        inter_selected_expert = tf.gather(
            self.inter_experts, idx, axis=0)
        intermediate = tf.einsum(
            'blh,blnhx->blnx',
            inputs,
            inter_selected_expert)
        gated_intermediate = tf.einsum('blnx,bln->blnx', intermediate, final_gate)
        out_selected_expert = tf.gather(self.out_experts, idx, axis=0)
        outputs = tf.einsum(
            'blnx,blnxh->blh',
            gated_intermediate,
            out_selected_expert)
        layer_output = self.layer_norm(outputs + inputs, name='layer_norm')
        return layer_output
