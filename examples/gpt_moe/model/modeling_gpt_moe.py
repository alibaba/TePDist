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

import os
import json
import tensorflow.compat.v1 as tf
from model import FLAGS, tffloat
from utils import get_initializer, get_shape_list
import layers

class GptMoeConfig(object):
    """Configuration for `Transformer`.

    Args:

      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.

    """

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 intermediate_size,
                 num_hidden_layers,
                 num_attention_heads,
                 max_position_embeddings,
                 type_vocab_size,
                 attention_head_size,
                 num_experts=0,
                 expert_capacity_dim=0,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.num_experts = num_experts
        self.expert_capacity_dim = expert_capacity_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.attention_head_size = attention_head_size
        
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                tf.logging.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err


def create_look_ahead_mask(from_tensor):
    from_shape = get_shape_list(from_tensor)
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    mask = tf.linalg.band_part(tf.ones((from_seq_length, from_seq_length)), -1, 0)

    mask = tf.cast(
        tf.reshape(mask, [1, from_seq_length, from_seq_length]), tffloat)

    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tffloat)
    mask = broadcast_ones * mask
    return mask


class SelfAttention(layers.Layer):
    def __init__(self, config, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = config.attention_head_size

        self.initializer = get_initializer(config.initializer_range)
        self.dropout = layers.Dropout(config.attention_probs_dropout_prob)

    def build(self, input_shape):
        self.q_head_weight = self.add_weight(
            shape=(self.hidden_size, self.attention_head_size * self.num_attention_heads),
            initializer=self.initializer,
            dtype=tffloat,
            name='query/kernel',
        )
        self.q_head_bias = self.add_weight(
            shape=(self.attention_head_size * self.num_attention_heads,),
            initializer=self.initializer,
            dtype=tffloat,
            name='query/bias',
        )
        self.k_head_weight = self.add_weight(
            shape=(self.hidden_size, self.attention_head_size * self.num_attention_heads),
            initializer=self.initializer,
            dtype=tffloat,
            name='key/kernel',
        )
        self.k_head_bias = self.add_weight(
            shape=(self.attention_head_size * self.num_attention_heads,),
            initializer=self.initializer,
            dtype=tffloat,
            name='key/bias',
        )
        self.v_head_weight = self.add_weight(
            shape=(self.hidden_size, self.attention_head_size * self.num_attention_heads),
            initializer=self.initializer,
            dtype=tffloat,
            name='value/kernel',
        )
        self.v_head_bias = self.add_weight(
            shape=(self.attention_head_size * self.num_attention_heads,),
            initializer=self.initializer,
            dtype=tffloat,
            name='value/bias',
        )

        super(SelfAttention, self).build(input_shape)

    def _abs_attn_core(self, q_head, k_head, v_head, attn_mask, training,
                       scale):
        attn_score = tf.einsum('bind,bjnd->bnij', q_head, k_head)
        attn_score = tf.multiply(attn_score, scale)

        attn_mask = tf.expand_dims(attn_mask, axis=[1])
        adder = (1.0 - tf.cast(attn_mask, tffloat)) * -10000.0
        attn_score += adder

        attn_prob = tf.nn.softmax(attn_score)
        attn_prob = self.dropout(attn_prob, training=training)

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
        seq_length = tf.shape(attention_mask)[1]

        q_head_h = tf.einsum('bih,hx->bix', q_input, self.q_head_weight)
        q_head_h = tf.nn.bias_add(q_head_h, self.q_head_bias)

        k_head_h = tf.einsum('bih,hx->bix', k_input, self.k_head_weight)
        k_head_h = tf.nn.bias_add(k_head_h, self.k_head_bias)

        v_head_h = tf.einsum('bih,hx->bix', v_input, self.v_head_weight)
        v_head_h = tf.nn.bias_add(v_head_h, self.v_head_bias)

        q_head_h = tf.reshape(q_head_h, [batch_size, seq_length, self.num_attention_heads, self.attention_head_size])
        k_head_h = tf.reshape(k_head_h, [batch_size, seq_length, self.num_attention_heads, self.attention_head_size])
        v_head_h = tf.reshape(v_head_h, [batch_size, seq_length, self.num_attention_heads, self.attention_head_size])

        scale = 1 / (self.attention_head_size ** 0.5)
        attn_vec = self._abs_attn_core(q_head_h, k_head_h, v_head_h, attention_mask, training, scale)
        attn_vec = tf.reshape(attn_vec, [batch_size, seq_length, self.attention_head_size * self.num_attention_heads])
        return attn_vec

class Attention(layers.Layer):
    def __init__(self, config, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.self_attention = SelfAttention(config, name="self")
        self.dense_output = dense_dropoutput(config, name="output")

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs
        self_outputs = self.self_attention(input_tensor, attention_mask, training=training)
        attention_output = self.dense_output(self_outputs, training=training)
        return attention_output

class dense_dropoutput(layers.Layer):
    def __init__(self, config, **kwargs):
        super(dense_dropoutput, self).__init__(**kwargs)
        self.dense = layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = layers.Dropout(config.hidden_dropout_prob)

    def call(self, input, training=False):
        hidden_states = self.dense(input)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states


class EncoderBlock(layers.Layer):
    def __init__(self, config, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = Attention(config, name="attention")
        # Use gelu_new, then match results

        self.LayerNorm = layers.LayerNormalization
        self.intermediate = layers.Dense(
            units=config.intermediate_size,
            activation=layers.gelu_new,
            kernel_initializer=get_initializer(config.initializer_range),
            name="intermediate/dense")

        self.bert_output = dense_dropoutput(config, name="output")

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        hidden_states = self.LayerNorm(hidden_states, name="layernorm1")
        attention_output = self.attention([hidden_states, attention_mask], training=training)
        layernorm_output = self.LayerNorm(attention_output + hidden_states, name="layernorm2")
        intermediate_output = self.intermediate(layernorm_output)
        layer_output = self.bert_output(intermediate_output, training=training)
        return layer_output, attention_output


class MoeEncoderBlock(layers.Layer):
    def __init__(self, config, **kwargs):
        super(MoeEncoderBlock, self).__init__(**kwargs)
        self.attention = Attention(config, name="attention")
        # Use gelu_new, then match results

        self.LayerNorm = layers.LayerNormalization
        self.moe_ffn = layers.MoeLayer(config, layer_norm=False, name="moe_ffn")

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        hidden_states = self.LayerNorm(hidden_states, name="layernorm1")
        attention_output = self.attention([hidden_states, attention_mask], training=training)
        layernorm_output = self.LayerNorm(attention_output + hidden_states, name="layernorm2")
        layer_output, _ = self.moe_ffn(layernorm_output, training=training)
        return layer_output, attention_output


class GptEncoder(layers.Layer):
    def __init__(self, config, **kwargs):
        super(GptEncoder, self).__init__(**kwargs)
        self.layer = []
        if config.num_experts > 0:
            for i in range(config.num_hidden_layers):
                if i % 2:
                    self.layer.append(EncoderBlock(config, name="layer_{}".format(i)))
                else:
                    self.layer.append(MoeEncoderBlock(config, name="layer_{}".format(i)))
        else:
            for i in range(config.num_hidden_layers):
                self.layer.append(EncoderBlock(config, name="layer_{}".format(i)))

        self.num_layers = config.num_hidden_layers

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs

        all_hidden_states = ()
        all_att_outputs = ()
        for i, layer_module in enumerate(self.layer):
            layer_output, att_output = layer_module([hidden_states, attention_mask], training=training)
            hidden_states = layer_output
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_att_outputs = all_att_outputs + (att_output,)

        final_outputs = []
        for hidden_states in all_hidden_states:
            final_outputs.append(hidden_states)

        return final_outputs, all_att_outputs


class GptMoeBackbone(layers.Layer):
    def __init__(self, config, **kwargs):

        self.embeddings = layers.GptEmbeddings(config, name="embeddings")

        self.encoder = GptEncoder(config, name="gpt-decoder")

        super(GptMoeBackbone, self).__init__(config, **kwargs)

    def call(self, inputs,
             input_mask=None,
             segment_ids=None,
             training=False):

        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            input_mask = inputs[1] if len(inputs) > 1 else input_mask
            segment_ids = inputs[2] if len(inputs) > 2 else segment_ids
        else:
            input_ids = inputs

        input_shape = get_shape_list(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if segment_ids is None:
            segment_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        tf.logging.info("********** Encoding batch size: {} {}*********** ".format(batch_size, seq_length))
        embedding_output = self.embeddings(input_ids, training=training)
        #attention_mask = layers.get_attn_mask_gpt(input_ids, input_mask)
        attention_mask = create_look_ahead_mask(input_ids) * \
                         tf.cast(tf.reshape(input_mask, [batch_size, 1, seq_length]), tffloat)

        outputs = self.encoder([embedding_output,
                                attention_mask
                                ], training=training)

        return outputs


class LMHead(layers.Layer):
    def __init__(self, config, embeddings, **kwargs):
        super(LMHead, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.dense = layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation=layers.gelu_new,
            name="transform/dense",
        )

        self.LayerNorm = layers.LayerNormalization
        self.embeddings = embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(
                self.vocab_size,
            ),
            initializer="zeros",
            trainable=True,
            name="output_bias")
        super(LMHead, self).build(input_shape)

    def call(self, hidden_states):
        shape = get_shape_list(hidden_states)
        hidden_states = tf.reshape(hidden_states, [-1, shape[-1]])
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(
            hidden_states, name="transform/LayerNorm")
        logits = layers.Dense(self.vocab_size, use_bias=False)(hidden_states)
        logits = tf.nn.bias_add(logits, self.bias)
        return logits


class GptMoePreTrainedModel(layers.Layer):
    def __init__(self, **kwargs):
        super(GptMoePreTrainedModel, self).__init__()

        self.config = GptMoeConfig(**kwargs)
        self.model_name = "GPT" if self.config.num_experts == 0 else "GPT_MoE"

        self.transformer = GptMoeBackbone(self.config, name=self.model_name.lower())
        self.lm = LMHead(self.config, self.transformer.embeddings, name="cls/predictions")
        self.model_dim = self.config.hidden_size

    def call(self, inputs, **kwargs):
        training = kwargs['mode'] == tf.estimator.ModeKeys.TRAIN
        tf.logging.info("********** {} PreTrainedModel ***********".format(self.model_name))
        if kwargs.get("output_features", True) == True:
            outputs = self.transformer(inputs, training=training)
            return outputs
        else:
            outputs = self.transformer(inputs, training=training)
            sequence_output = outputs[0][-1] * (self.model_dim ** -0.5)
            input_shape = get_shape_list(sequence_output)
            batch_size = input_shape[0]
            seq_length = input_shape[1]

            lm_logits = self.lm(sequence_output)
            return lm_logits

