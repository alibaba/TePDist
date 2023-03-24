# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""AdamWeightDecay for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export

import re

@tf_export(v1=["train.AdamWeightDecayOptimizer"])
class AdamWeightDecayOptimizer(optimizer.Optimizer):
  """Optimizer that implements the AdamWeightDecay algorithm.

  See [IIya 2017](http://arxiv.org/pdf/1711.05101)
  ([pdf](https://arxiv.org/pdf/1711.05101.pdf)).
  """

  def __init__(self,
               learning_rate=0.001,
               weight_decay_rate=0.0,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               exclude_from_weight_decay=None,
               use_locking=False,
               name="AdamWeightDecayOptimizer"):
    r"""A basic Adam optimizer that includes "correct" L2 weight decay.
    """
    super(AdamWeightDecayOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._weight_decay_rate = weight_decay_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._exclude_from_weight_decay = exclude_from_weight_decay

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(
        initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
    self._create_non_slot_variable(
        initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    weight_decay_rate = self._call_if_callable(self._weight_decay_rate)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)

    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._weight_decay_rate_t = ops.convert_to_tensor(
        weight_decay_rate, name='weight_decay_rate')
    self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    use_weight_decay = True if self._do_use_weight_decay(var.name) else False
    return training_ops.apply_adam_weight_decay(
        var,
        m,
        v,
        math_ops.cast(beta1_power, var.dtype.base_dtype),
        math_ops.cast(beta2_power, var.dtype.base_dtype),
        math_ops.cast(self._lr_t, var.dtype.base_dtype),
        math_ops.cast(self._beta1_t, var.dtype.base_dtype),
        math_ops.cast(self._beta2_t, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
        grad,
        math_ops.cast(self._weight_decay_rate_t, var.dtype.base_dtype),
        use_weight_decay=use_weight_decay,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators()
    use_weight_decay = True if self._do_use_weight_decay(var.name) else False
    return training_ops.resource_apply_adam_weight_decay(
        var.handle,
        m.handle,
        v.handle,
        math_ops.cast(beta1_power, grad.dtype.base_dtype),
        math_ops.cast(beta2_power, grad.dtype.base_dtype),
        math_ops.cast(self._lr_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
        math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
        math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
        grad,
        math_ops.cast(self._weight_decay_rate_t, var.dtype.base_dtype),
        use_weight_decay=use_weight_decay,
        use_locking=self._use_locking)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    raise RuntimeError("Unimplemented _apply_sparse_shared method")

  def _apply_sparse(self, grad, var):
    raise RuntimeError("Unimplemented _apply_sparse method")

  def _resource_scatter_add(self, x, i, v):
    raise RuntimeError("Unimplemented _resource_scatter_add method")

  def _resource_apply_sparse(self, grad, var, indices):
    raise RuntimeError("Unimplemented _resource_apply_sparse method")

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      beta1_power, beta2_power = self._get_beta_accumulators()
      with ops.colocate_with(beta1_power):
        update_beta1 = beta1_power.assign(
            beta1_power * self._beta1_t, use_locking=self._use_locking)
        update_beta2 = beta2_power.assign(
            beta2_power * self._beta2_t, use_locking=self._use_locking)
    return control_flow_ops.group(
        *update_ops + [update_beta1, update_beta2], name=name_scope)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self._weight_decay_rate:
        return False
    if self._exclude_from_weight_decay:
        for r in self._exclude_from_weight_decay:
            if re.search(r, param_name) is not None:
                return False
    return True
