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


import tensorflow.compat.v1 as tf
from tensorflow.python.layers.base import Layer
from model import FLAGS, tffloat
from utils import get_initializer
from .core import LayerNormalization
from .activations import get_activation_fn


class MoeGating(Layer):
    """
    Computes moe gating for Mixture-of-Experts.

    Dimensions cheat sheet:
    D: device num
    G: group_dim
    S: group_size_dim
    E: number of experts
    C: capacity per expert
    M: model_dim (same as input_dim, same as output_dim)
    B: original batch_dim
    L: original sequence_length_dim

    Args:
        num_experts: number of experts.
        expert_capacity_dim: number of examples per minibatch(group) per expert.
          Each example is typically a vector of size input_dim, representing
          embedded token or an element of Transformer layer output.
        local_dispatch: whether dispatch is local to the group (G dim)
        hidden_size: hidden dim of gate inputs.
        second_expert_policy: 'all' or 'random', we optionally 'random'-ize dispatch
          to second-best expert proportional to (weight / second_expert_threshold).
        second_expert_threshold: threshold for probability normalization for
          second_expert_policy == 'random'.

    Returns:
        A tuple (dispatch_tensor, combine_tensor, aux_loss).
        - dispatch_tensor: G'SEC Tensor, scattering/dispatching inputs to
          experts.
        - combine_tensor: G'SEC Tensor.
          combining expert outputs.
        - aux_loss: auxiliary loss, equalizing the expert assignment ratios.

    """

    def __init__(self, config, **kwargs):
        super(MoeGating, self).__init__(config, **kwargs)
        self.num_experts = config.num_experts
        self.local_dispatch = config.local_dispatch
        self.expert_capacity_dim = config.expert_capacity_dim
        self.hidden_size = config.hidden_size
        self.initializer = get_initializer(config.initializer_range)
        self.loss_coef = config.loss_coef
        self.min_expert_capacity = config.min_expert_capacity
        self.is_training = config.is_training

    def build(self, input_shape):
        # gating weights for each expert.
        self.gating_weight = self.add_weight(
            shape=(self.hidden_size, self.num_experts),
            initializer=self.initializer,
            dtype=tffloat,
            name='gating_weight',
        )
        super(MoeGating, self).build(input_shape)

    def call(self, inputs, total_token_num):
        """Computes gating outputs for Mixture-of-Experts.

        Args:
            inputs: G'SM Tensor.

        Returns:
            - dispatch_tensor: G`SEC Tensor, scattering/dispatching inputs to experts.
            - combine_tensor: G`SEC Tensor. combining expert outputs.
            - aux_loss: auxiliary loss, equalizing the expert assignment ratios.
        """
        orig_inputs = inputs
        if not self.local_dispatch:
            inputs = tf.reshape(inputs, [1, -1, inputs.shape[2]])

        aux_loss, combine_tensor, dispatch_mask = self.gating_internel(inputs, total_token_num)

        if not self.local_dispatch:
            dispatch_mask = tf.reshape(
                dispatch_mask, [orig_inputs.shape[0], -1, dispatch_mask.shape[2], dispatch_mask.shape[3]])
            combine_tensor = tf.reshape(
                combine_tensor, [orig_inputs.shape[0], -1, combine_tensor.shape[2], combine_tensor.shape[3]])

        return combine_tensor, dispatch_mask, aux_loss

    def gating_internel(self, logits):
        raise NotImplementedError("gating_internel not implemented in subclass!")


class Top2Gating(MoeGating):
    def __init__(self, config, **kwargs):
        self.second_expert_policy = config.second_expert_policy
        self.second_expert_threshold = config.second_expert_threshold

        super(Top2Gating, self).__init__(config, **kwargs)

    def gating_internel(self, inputs, total_token_num):
        logits = tf.einsum('GSM,ME->GSE', inputs, self.gating_weight)  # G'SE
        raw_gates = tf.nn.softmax(logits)  # along E dim, G'SE
        tf.logging.info("raw_gates:{}".format(raw_gates))

        while self.expert_capacity_dim % 4:
          self.expert_capacity_dim += 1
          tf.logging.info(
              'Setting expert_capacity_dim=%r ('
              'num_experts=%r name_scope=%r)',
              self.expert_capacity_dim, self.num_experts,
              tf.get_default_graph().get_name_scope())

        # First top gate idx and gate val
        top_gate_index_1 = tf.math.argmax(raw_gates, axis=-1, output_type=tf.int32)  # G'S
        #tf.summary.tensor_summary('top_gate_index_1', top_gate_index_1)
        mask_1 = tf.one_hot(top_gate_index_1, self.num_experts, dtype=tffloat)  # G'SE
        density_1_proxy = raw_gates
        importance = tf.ones_like(mask_1[:, :, 0])
        gate_1 = tf.einsum('GSE,GSE->GS', raw_gates, mask_1)  # G'S

        # Second top gate idx and gate val
        gates_without_top_1 = raw_gates * (1.0 - mask_1)
        top_gate_index_2 = tf.math.argmax(gates_without_top_1, axis=-1, output_type=tf.int32)  # G'S
        #tf.summary.tensor_summary('top_gate_index_2', top_gate_index_2)
        mask_2 = tf.one_hot(top_gate_index_2, self.num_experts, dtype=tffloat)  # G'SE
        gate_2 = tf.einsum('GSE,GSE->GS', gates_without_top_1, mask_2)  # G'S

        # We reshape the mask as [X*S, E], and compute cumulative sums of
        # assignment indicators for each expert index e \in 0..E-1 independently.
        # First occurrence of assignment indicator is excluded, see exclusive=True
        # flag below.
        position_in_expert_1 = tf.cumsum(mask_1, exclusive=True, axis=1)

        # GS Tensor
        capacity = tf.cast(self.expert_capacity_dim, dtype=position_in_expert_1.dtype)

        # GE Tensor (reducing S out of GSE tensor mask_1)
        # density_1[:, e] represents assignment ratio (num assigned / total) to
        # expert e as top_1 expert without taking capacity into account.
        density_denom = tf.reduce_mean(
            importance, axis=(1))[:, tf.newaxis] + 1e-6
        density_1 = tf.reduce_mean(mask_1, axis=(1)) / density_denom
        # density_1_proxy[:, e] represents mean of raw_gates for expert e, including
        # those of examples not assigned to e with top_k.
        density_1_proxy = tf.reduce_mean(density_1_proxy, axis=1) / density_denom

        with tf.name_scope('aux_loss'):
            # The MoE paper (https://arxiv.org/pdf/1701.06538.pdf) uses an aux loss of
            # reduce_mean(density_1_proxy * density_1_proxy). Here we replace one of
            # the density_1_proxy with the discrete density_1 following mesh_tensorflow.
            aux_loss = tf.reduce_mean(density_1_proxy * density_1)  # element-wise
            aux_loss *= self.num_experts * self.num_experts  # const coefficient

        mask_1 *= tf.cast(tf.less(position_in_expert_1, capacity), dtype=mask_1.dtype)
        position_in_expert_1 = tf.einsum('GSE,GSE->GS', position_in_expert_1, mask_1)

        # How many examples in this sequence go to this expert
        mask_1_count = tf.einsum('GSE->GE', mask_1)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = tf.einsum('GSE->GS', mask_1)

        if self.second_expert_policy == 'all':
            pass
        elif self.second_expert_policy == 'random':
            # gate_2 is between 0 and 1, reminder:
            #
            #   raw_gates = tf.nn.softmax(logits)
            #   index_1 = tf.math.argmax(raw_gates, axis=-1, output_type=tf.int32)
            #   mask_1 = tf.one_hot(index_1, num_experts, dtype=tffloat)
            #   gate_1 = tf.einsum('GSE,GSE->GS', raw_gates, mask_1)
            #
            # E.g. if gate_2 exceeds second_expert_threshold, then we definitely
            # dispatch to second-best expert. Otherwise we dispatch with probability
            # proportional to (gate_2 / threshold).
            #
            sampled_2 = tf.less(
                tf.random.uniform(gate_2.shape, dtype=gate_2.dtype),
                (gate_2 / max(self.second_expert_threshold, 1e-9)))
            gate_2 *= tf.cast(sampled_2, gate_2.dtype)
            mask_2 *= tf.cast(tf.expand_dims(sampled_2, -1), mask_2.dtype)
        else:
            raise ValueError(self.second_expert_policy)

        # Sum token count of first and second top gate.
        position_in_expert_2 = tf.cumsum(
            mask_2, exclusive=True, axis=1) + tf.expand_dims(mask_1_count, 1)

        mask_2 *= tf.cast(tf.less(position_in_expert_2, capacity), mask_2.dtype)
        position_in_expert_2 = tf.einsum('GSE,GSE->GS', position_in_expert_2, mask_2)
        mask_2_flat = tf.reduce_sum(mask_2, axis=-1)

        gate_1 *= mask_1_flat
        gate_2 *= mask_2_flat

        # Normalize top-k gates.
        denom = gate_1 + gate_2
        # To avoid divide by 0.
        denom = tf.where(denom > 0, denom, tf.ones_like(denom))
        gate_1 /= denom
        gate_2 /= denom

        # First top gate as first part of combine tensor
        b = tf.one_hot(
            tf.cast(position_in_expert_1, dtype=tf.int32),
            self.expert_capacity_dim,
            dtype=tffloat,
            name='one_hot_b_0')  # G'SE
        a = tf.expand_dims(gate_1 * mask_1_flat, -1) * tf.one_hot(
            top_gate_index_1, self.num_experts, dtype=tffloat)  # G'SE
        first_part_of_combine_tensor = tf.einsum(
            'GSE,GSC->GSEC', a, b, name='first_part_of_combine_tensor')  # G'SEC

        # Second top gate as first part of combine tensor
        b = tf.one_hot(
            tf.cast(position_in_expert_2, dtype=tf.int32),
            self.expert_capacity_dim,
            dtype=tffloat,
            name='one_hot_b_1')  # G'SE
        a = tf.expand_dims(gate_2 * mask_2_flat, -1) * tf.one_hot(
            top_gate_index_2, self.num_experts, dtype=tffloat)  # G'SE
        second_part_of_combine_tensor = tf.einsum(
            'GSE,GSC->GSEC', a, b, name='second_part_of_combine_tensor')  # G'SEC

        # Combine tensors of two parts.
        combine_tensor = tf.math.add(
            first_part_of_combine_tensor,
            second_part_of_combine_tensor,
            name='combine_tensor')  # G'SEC
        dispatch_mask = tf.cast(
            tf.cast(combine_tensor, tf.bool), tffloat, name='dispatch_mask')  # G'SEC

        return aux_loss, combine_tensor, dispatch_mask


class SwitchGating(MoeGating):
    def __init__(self, config, **kwargs):
        super(SwitchGating, self).__init__(config, **kwargs)
        self.min_expert_capacity = config.min_expert_capacity
        self.switch_policy_train = config.switch_policy_train
        self.switch_policy_eval = config.switch_policy_eval
        self.switch_dropout = config.switch_dropout
        self.capacity_factor_train = config.capacity_factor_train
        self.capacity_factor_eval = config.capacity_factor_eval

    def gating_internel(self, inputs, total_token_num):
        if self.is_training:
          policy = self.switch_policy_train
          capacity_factor = self.capacity_factor_train
        else:
          policy = self.switch_policy_eval
          capacity_factor = self.capacity_factor_eval

        # if not self.expert_capacity_dim:
        # Always calculate expert_capacity_dim from current related hyper-params
        # instead of reading this value from config.json.
        num_experts = self.num_experts
        # How to calculate capacity, please check the easytransfer docs. 
        capacity = float(int(total_token_num) / int(num_experts)) * float(capacity_factor)
        tf.logging.info('capacity=%r', capacity)
        int_capacity = int(capacity)
        offset = 1 if capacity > float(int_capacity) else 0
        self.expert_capacity_dim = max(int(offset) + int_capacity, self.min_expert_capacity)
        tf.logging.info(
            'Setting expert_capacity_dim=%r ('
            'num_experts=%r name_scope=%r)',
            self.expert_capacity_dim, self.num_experts,
            tf.get_default_graph().get_name_scope())

        if self.is_training and policy == "input_dropout":
          inputs = tf.nn.dropout(inputs, 1.0 - self.switch_dropout)

        # Replicate gating weight
        self.gating_weight = xla_sharding.replicate(self.gating_weight)

        logits = tf.einsum('GSM,ME->GSE', inputs, self.gating_weight)  # G'SE
        raw_gates = tf.nn.softmax(logits)  # along E dim, G'SE

        if policy == "argmax" or policy == "input_dropout":
            # expert_gate, expert_index = tf.math.top_k(raw_gates, k=1)
            # expert_index = tf.squeeze(expert_index, [2])
            expert_index = tf.math.argmax(raw_gates, axis=2)
            #if self.is_training:
            #  tf.summary.scalar('expert_gate', expert_gate)
        else:
            raise ValueError("Unknown Switch gating policy %s" % policy)

        expert_mask = tf.one_hot(expert_index, self.num_experts, dtype=tffloat)  # G'SE
        density_1_proxy = raw_gates  # G'SE
        importance = tf.ones_like(expert_mask[:, :, 0])  # G'SE
        gate_1 = tf.einsum('GSE,GSE->GS', raw_gates, expert_mask)  # G'S

        # We reshape the mask as [X*S, E], and compute cumulative sums of
        # assignment indicators for each expert index e \in 0..E-1 independently.
        # First occurrence of assignment indicator is excluded, see exclusive=True
        # flag below.
        position_in_expert_1 = tf.cumsum(expert_mask, exclusive=True, axis=1)

        # GS Tensor
        capacity = tf.cast(self.expert_capacity_dim, dtype=position_in_expert_1.dtype)

        # GE Tensor (reducing S out of GSE tensor mask_1)
        # density_1[:, e] represents assignment ratio (num assigned / total) to
        # expert e as top_1 expert without taking capacity into account.
        density_denom = tf.reduce_mean(
            importance, axis=(1))[:, tf.newaxis] + 1e-6
        density_1 = tf.reduce_mean(expert_mask, axis=(1)) / density_denom
        # density_1_proxy[:, e] represents mean of raw_gates for expert e, including
        # those of examples not assigned to e with top_k.
        density_1_proxy = tf.reduce_mean(density_1_proxy, axis=1) / density_denom

        with tf.name_scope('aux_loss'):
            # The MoE paper (https://arxiv.org/pdf/1701.06538.pdf) uses an aux loss of
            # reduce_mean(density_1_proxy * density_1_proxy). Here we replace one of
            # the density_1_proxy with the discrete density_1 following mesh_tensorflow.
            aux_loss = tf.reduce_mean(density_1_proxy * density_1)  # element-wise
            aux_loss *= self.num_experts * self.num_experts * self.loss_coef  # const coefficient
            # Logging for aux_loss and expert load balance
            #if self.is_training:
            #  tf.summary.scalar('aux_loss', aux_loss)
            #  mask_count_experts = tf.reduce_sum(expert_mask, axis=1)
            # total_routed = tf.reduce_sum(mask_count_experts)
            #  expert_fraction = tf.to_float(mask_count_experts / total_routed)
            #  #tf.logging.info("expert_fraction:{}".format(expert_fraction))
            #  split_fractions = tf.split(
            #      expert_fraction,
            #      num_or_size_splits=self.num_experts,
            #      axis=-1)
            #  for fraction in split_fractions:
            #    tf.summary.scalar("experts/" + fraction.name.replace(":", "/"),
            #                      tf.reduce_mean(fraction))

        expert_mask *= tf.cast(tf.less(position_in_expert_1, capacity), dtype=expert_mask.dtype)
        position_in_expert_1 = tf.einsum('GSE,GSE->GS', position_in_expert_1, expert_mask)

        # [batch, group] - mostly ones, but zeros where something didn't fit
        mask_1_flat = tf.einsum('GSE->GS', expert_mask)

        gate_1 *= mask_1_flat

        # First top gate as first part of combine tensor
        b = tf.one_hot(
            tf.cast(position_in_expert_1, dtype=tf.int32),
            self.expert_capacity_dim,
            dtype=tffloat,
            name='one_hot_b_0')  # G'SE
        a = tf.expand_dims(gate_1 * mask_1_flat, -1) * tf.one_hot(
            expert_index, self.num_experts, dtype=tffloat)  # G'SE
        combine_tensor = tf.einsum(
            'GSE,GSC->GSEC', a, b, name='first_part_of_combine_tensor')  # G'SEC

        dispatch_mask = tf.cast(
            tf.cast(combine_tensor, tf.bool), tffloat, name='dispatch_mask')  # G'SEC

        return aux_loss, combine_tensor, dispatch_mask


class FeedForwardNetwork(Layer):
    """Construct MOE FeedForward Networks.

      Args:
          reshaped_inputs: G`SM Tensor.

      Returns:
          outputs: G`SM Tensor.
          aux_loss: scalar auxiliary loss.
      """

    def __init__(self, config, **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.initializer = get_initializer(config.initializer_range)
        self.num_experts = config.num_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.moe_gating_policy = config.moe_gating
        self.activation_fn = get_activation_fn(config.activation_fn)
        if self.moe_gating_policy == "top_2":
          self.gate = Top2Gating(config, name='top_2_gating')
        elif self.moe_gating_policy == "switch":
          self.gate = SwitchGating(config, name='switch_gating')
        else:
          raise RuntimeError(
              "Not supported gating policy {}, expect 'top_2' or 'switch'."
                  .format(self.moe_gating_policy))

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

        super(FeedForwardNetwork, self).build(input_shape)

    def call(self, reshaped_inputs, total_token_num, training=True, name="FFN"):
        combine_tensor, dispatch_mask, aux_loss = \
            self.gate(reshaped_inputs, total_token_num)  # G'SEC

        # dispatched inputs
        dispatch_inputs = tf.einsum(
            "GSEC,GSM->EGCM", dispatch_mask, reshaped_inputs, name="dispatch_inputs")  # EG'CM

        # inter_experts forward
        intermediate = tf.einsum(
          'EGCM,EMH->EGCH', dispatch_inputs, self.inter_experts, name="dispatched_inter_outputs")  # EG'CH
        # activation function
        activated_inters = self.activation_fn(intermediate)  # EG'CH

        # output_experts forward
        output_experts = tf.einsum(
            'EGCH,EHM->EGCM', activated_inters, self.out_experts, name="dispatched_outputs")

        combined_outputs = tf.einsum(
            'GSEC,EGCM->GSM', combine_tensor, output_experts, name="combined_outputs")

        return combined_outputs, aux_loss


class MoeLayer(Layer):
    """Construct MOE Networks.

    Dimensions cheat sheet:
    D: device num
    G: global group_dim
    S: group_size_dim
    E: number of experts
    C: capacity per expert
    M: model_dim (same as input_dim, same as output_dim)
    B: original batch_dim
    L: original sequence_length_dim

    Args:
        inputs: input tensor. B * L * M.

    Returns:
        outputs: output tensor. B * L * M.
        aux_loss: scalar auxiliary loss.

    Note:
        G * S = D * B * L
        G' * D = G
    """

    def __init__(self, config, layer_norm=True, **kwargs):
        super(MoeLayer, self).__init__(**kwargs)
        self.local_dispatch = config.local_dispatch
        self.num_local_groups = config.num_local_groups  # G'
        self.ffn = FeedForwardNetwork(config, **kwargs)
        self.layer_norm = LayerNormalization if layer_norm else None

    def build(self, input_shape):
        super(MoeLayer, self).build(input_shape)

    def call(self, inputs, training=True, name="MoeLayer"):
        # ffn forward
        # Dispatch inputs to each slice devices.
        orig_inputs = inputs  # BLM
        batch_size = FLAGS.train_batch_size
        token_len = orig_inputs.shape[1]
        total_token_num = batch_size * token_len

        # local dispatch inputs
        if self.num_local_groups:
            inputs = tf.reshape(
                orig_inputs, [
                    self.num_local_groups,
                    -1,
                    orig_inputs.shape[-1]
                ],
                name='grouped_inputs')  # G'SM, BL=G'S
        outputs, aux_loss = self.ffn(
            inputs, total_token_num, training, name="dispatched_expert_outputs")

        # restore dispatched outputs
        if self.num_local_groups:
            outputs = tf.reshape(
                outputs, [
                    -1,
                    orig_inputs.shape[1],
                    orig_inputs.shape[-1]
                ],
                name='outputs')  # BLM

        # add & layer-norm
        if self.layer_norm:
            outputs = self.layer_norm(outputs + orig_inputs)

        return outputs, aux_loss
