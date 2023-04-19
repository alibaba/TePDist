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


import tensorflow.compat.v1 as tf
from tensorflow.python.layers.base import Layer
from .core import LayerNormalization, Dropout
from utils import get_initializer, get_shape_list
from model import tffloat


class GptEmbeddings(Layer):
    """Construct the embeddings from word, position embeddings.
    """

    def __init__(self, config, **kwargs):
        super(GptEmbeddings, self).__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.max_position_embeddings = config.max_position_embeddings

        self.initializer = get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.word_embeddings = self.add_weight(
            "word_embeddings",
            dtype=tffloat,
            shape=[self.vocab_size, self.hidden_size],
            initializer=self.initializer,
        )

        self.position_embeddings = self.add_weight(
            "position_embeddings",
            dtype=tffloat,
            shape=[self.max_position_embeddings, self.hidden_size],
            initializer=self.initializer,
        )

        super(GptEmbeddings, self).build(input_shape)

    def call(self, inputs, training=False):
        input_embeddings = tf.gather(self.word_embeddings, inputs)

        input_shape = get_shape_list(input_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        position_embeddings = tf.gather(self.position_embeddings, tf.range(0, seq_length))
        position_embeddings = tf.expand_dims(position_embeddings, 0)

        input_embeddings += position_embeddings

        output = input_embeddings
        return output
