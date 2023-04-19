# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for categorical preprocessing layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.preprocessing import categorical_crossing
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class CategoryCrossingTest(keras_parameterized.TestCase):

  def test_crossing_basic(self):
    layer = categorical_crossing.CategoryCrossing()
    inputs_0 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [1, 1]],
        values=['a', 'b', 'c'],
        dense_shape=[2, 2])
    inputs_1 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [1, 2]], values=['d', 'e'], dense_shape=[2, 3])
    output = layer([inputs_0, inputs_1])
    self.assertAllClose(np.asarray([[0, 0], [1, 0], [1, 1]]), output.indices)
    self.assertAllEqual([b'a_X_d', b'b_X_e', b'c_X_e'], output.values)

  def test_crossing_sparse_inputs(self):
    layer = categorical_crossing.CategoryCrossing(num_bins=1)
    inputs_0 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [1, 1]],
        values=['a', 'b', 'c'],
        dense_shape=[2, 2])
    inputs_1 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [1, 2]], values=['d', 'e'], dense_shape=[2, 3])
    output = layer([inputs_0, inputs_1])
    self.assertAllClose(np.asarray([[0, 0], [1, 0], [1, 1]]), output.indices)
    self.assertAllClose([0, 0, 0], output.values)

  def test_crossing_sparse_inputs_with_hash_key(self):
    layer = categorical_crossing.CategoryCrossing(num_bins=2, hash_key=133)
    inputs_0 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [1, 1]],
        values=['a', 'b', 'c'],
        dense_shape=[2, 2])
    inputs_1 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [1, 2]], values=['d', 'e'], dense_shape=[2, 3])
    output = layer([inputs_0, inputs_1])
    self.assertAllClose(np.asarray([[0, 0], [1, 0], [1, 1]]), output.indices)
    self.assertAllClose([1, 0, 1], output.values)

    layer_2 = categorical_crossing.CategoryCrossing(num_bins=2, hash_key=137)
    output = layer_2([inputs_0, inputs_1])
    self.assertAllClose(np.asarray([[0, 0], [1, 0], [1, 1]]), output.indices)
    # Note the output is different with above.
    self.assertAllClose([0, 1, 0], output.values)

  def test_crossing_sparse_inputs_depth_int(self):
    layer = categorical_crossing.CategoryCrossing(depth=1)
    inputs_0 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [2, 0]],
        values=['a', 'b', 'c'],
        dense_shape=[3, 1])
    inputs_1 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [2, 0]],
        values=['d', 'e', 'f'],
        dense_shape=[3, 1])
    output = layer([inputs_0, inputs_1])
    self.assertIsInstance(output, sparse_tensor.SparseTensor)
    output = sparse_ops.sparse_tensor_to_dense(output)
    expected_out = [[b'a', b'd'], [b'b', b'e'], [b'c', b'f']]
    self.assertAllEqual(expected_out, output)

  def test_crossing_sparse_inputs_depth_tuple(self):
    layer = categorical_crossing.CategoryCrossing(depth=(2, 3))
    inputs_0 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [2, 0]],
        values=['a', 'b', 'c'],
        dense_shape=[3, 1])
    inputs_1 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [2, 0]],
        values=['d', 'e', 'f'],
        dense_shape=[3, 1])
    inputs_2 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [2, 0]],
        values=['g', 'h', 'i'],
        dense_shape=[3, 1])
    inp_0_t = input_layer.Input(shape=(1,), sparse=True, dtype=dtypes.string)
    inp_1_t = input_layer.Input(shape=(1,), sparse=True, dtype=dtypes.string)
    inp_2_t = input_layer.Input(shape=(1,), sparse=True, dtype=dtypes.string)
    out_t = layer([inp_0_t, inp_1_t, inp_2_t])
    model = training.Model([inp_0_t, inp_1_t, inp_2_t], out_t)
    output = model.predict([inputs_0, inputs_1, inputs_2])
    self.assertIsInstance(output, sparse_tensor.SparseTensor)
    output = sparse_ops.sparse_tensor_to_dense(output)
    expected_outputs_0 = [[b'a_X_d', b'a_X_g', b'd_X_g', b'a_X_d_X_g']]
    expected_outputs_1 = [[b'b_X_e', b'b_X_h', b'e_X_h', b'b_X_e_X_h']]
    expected_outputs_2 = [[b'c_X_f', b'c_X_i', b'f_X_i', b'c_X_f_X_i']]
    expected_out = array_ops.concat(
        [expected_outputs_0, expected_outputs_1, expected_outputs_2], axis=0)
    self.assertAllEqual(expected_out, output)

  def test_crossing_hashed_two_bins(self):
    layer = categorical_crossing.CategoryCrossing(num_bins=2)
    inputs_0 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [1, 1]],
        values=['a', 'b', 'c'],
        dense_shape=[2, 2])
    inputs_1 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [1, 2]], values=['d', 'e'], dense_shape=[2, 3])
    output = layer([inputs_0, inputs_1])
    self.assertAllClose(np.asarray([[0, 0], [1, 0], [1, 1]]), output.indices)
    self.assertEqual(output.values.numpy().max(), 1)
    self.assertEqual(output.values.numpy().min(), 0)

  def test_crossing_hashed_ragged_inputs(self):
    layer = categorical_crossing.CategoryCrossing(num_bins=2)
    inputs_0 = ragged_factory_ops.constant(
        [['omar', 'skywalker'], ['marlo']],
        dtype=dtypes.string)
    inputs_1 = ragged_factory_ops.constant(
        [['a'], ['b']],
        dtype=dtypes.string)
    out_data = layer([inputs_0, inputs_1])
    expected_output = [[0, 0], [0]]
    self.assertAllClose(expected_output, out_data)
    inp_0_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    inp_1_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    out_t = layer([inp_0_t, inp_1_t])
    model = training.Model(inputs=[inp_0_t, inp_1_t], outputs=out_t)
    self.assertAllClose(expected_output, model.predict([inputs_0, inputs_1]))

    non_hashed_layer = categorical_crossing.CategoryCrossing()
    out_t = non_hashed_layer([inp_0_t, inp_1_t])
    model = training.Model(inputs=[inp_0_t, inp_1_t], outputs=out_t)
    expected_output = [[b'omar_X_a', b'skywalker_X_a'], [b'marlo_X_b']]
    self.assertAllEqual(expected_output, model.predict([inputs_0, inputs_1]))

  def test_crossing_ragged_inputs_depth_int(self):
    layer = categorical_crossing.CategoryCrossing(depth=1)
    inputs_0 = ragged_factory_ops.constant([['a'], ['b'], ['c']])
    inputs_1 = ragged_factory_ops.constant([['d'], ['e'], ['f']])
    output = layer([inputs_0, inputs_1])
    expected_output = [[b'a', b'd'], [b'b', b'e'], [b'c', b'f']]
    self.assertIsInstance(output, ragged_tensor.RaggedTensor)
    self.assertAllEqual(expected_output, output)

    layer = categorical_crossing.CategoryCrossing(depth=2)
    inp_0_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    inp_1_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    out_t = layer([inp_0_t, inp_1_t])
    model = training.Model([inp_0_t, inp_1_t], out_t)
    expected_output = [[b'a', b'd', b'a_X_d'], [b'b', b'e', b'b_X_e'],
                       [b'c', b'f', b'c_X_f']]
    self.assertAllEqual(expected_output, model.predict([inputs_0, inputs_1]))

  def test_crossing_ragged_inputs_depth_tuple(self):
    layer = categorical_crossing.CategoryCrossing(depth=[2, 3])
    inputs_0 = ragged_factory_ops.constant([['a'], ['b'], ['c']])
    inputs_1 = ragged_factory_ops.constant([['d'], ['e'], ['f']])
    inputs_2 = ragged_factory_ops.constant([['g'], ['h'], ['i']])
    inp_0_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    inp_1_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    inp_2_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    out_t = layer([inp_0_t, inp_1_t, inp_2_t])
    model = training.Model([inp_0_t, inp_1_t, inp_2_t], out_t)
    expected_output = [[b'a_X_d', b'a_X_g', b'd_X_g', b'a_X_d_X_g'],
                       [b'b_X_e', b'b_X_h', b'e_X_h', b'b_X_e_X_h'],
                       [b'c_X_f', b'c_X_i', b'f_X_i', b'c_X_f_X_i']]
    output = model.predict([inputs_0, inputs_1, inputs_2])
    self.assertIsInstance(output, ragged_tensor.RaggedTensor)
    self.assertAllEqual(expected_output, output)

  def test_invalid_mixed_sparse_and_ragged_input(self):
    with self.assertRaises(ValueError):
      layer = categorical_crossing.CategoryCrossing(num_bins=2)
      inputs_0 = ragged_factory_ops.constant(
          [['omar'], ['marlo']],
          dtype=dtypes.string)
      inputs_1 = sparse_tensor.SparseTensor(
          indices=[[0, 1], [1, 2]], values=['d', 'e'], dense_shape=[2, 3])
      layer([inputs_0, inputs_1])

  def test_crossing_with_dense_inputs(self):
    layer = categorical_crossing.CategoryCrossing()
    inputs_0 = np.asarray([[1, 2]])
    inputs_1 = np.asarray([[1, 3]])
    output = layer([inputs_0, inputs_1])
    self.assertAllEqual([[b'1_X_1', b'1_X_3', b'2_X_1', b'2_X_3']], output)

  def test_crossing_dense_inputs_depth_int(self):
    layer = categorical_crossing.CategoryCrossing(depth=1)
    inputs_0 = constant_op.constant([['a'], ['b'], ['c']])
    inputs_1 = constant_op.constant([['d'], ['e'], ['f']])
    output = layer([inputs_0, inputs_1])
    expected_output = [[b'a', b'd'], [b'b', b'e'], [b'c', b'f']]
    self.assertAllEqual(expected_output, output)

    layer = categorical_crossing.CategoryCrossing(depth=2)
    inp_0_t = input_layer.Input(shape=(1,), dtype=dtypes.string)
    inp_1_t = input_layer.Input(shape=(1,), dtype=dtypes.string)
    out_t = layer([inp_0_t, inp_1_t])
    model = training.Model([inp_0_t, inp_1_t], out_t)
    crossed_output = [[b'a_X_d'], [b'b_X_e'], [b'c_X_f']]
    expected_output = array_ops.concat([expected_output, crossed_output],
                                       axis=1)
    self.assertAllEqual(expected_output, model.predict([inputs_0, inputs_1]))

  def test_crossing_dense_inputs_depth_tuple(self):
    layer = categorical_crossing.CategoryCrossing(depth=[2, 3])
    inputs_0 = constant_op.constant([['a'], ['b'], ['c']])
    inputs_1 = constant_op.constant([['d'], ['e'], ['f']])
    inputs_2 = constant_op.constant([['g'], ['h'], ['i']])
    inp_0_t = input_layer.Input(shape=(1,), dtype=dtypes.string)
    inp_1_t = input_layer.Input(shape=(1,), dtype=dtypes.string)
    inp_2_t = input_layer.Input(shape=(1,), dtype=dtypes.string)
    out_t = layer([inp_0_t, inp_1_t, inp_2_t])
    model = training.Model([inp_0_t, inp_1_t, inp_2_t], out_t)
    expected_outputs_0 = [[b'a_X_d', b'a_X_g', b'd_X_g', b'a_X_d_X_g']]
    expected_outputs_1 = [[b'b_X_e', b'b_X_h', b'e_X_h', b'b_X_e_X_h']]
    expected_outputs_2 = [[b'c_X_f', b'c_X_i', b'f_X_i', b'c_X_f_X_i']]
    expected_output = array_ops.concat(
        [expected_outputs_0, expected_outputs_1, expected_outputs_2], axis=0)
    self.assertAllEqual(expected_output,
                        model.predict([inputs_0, inputs_1, inputs_2]))

  def test_crossing_hashed_with_dense_inputs(self):
    layer = categorical_crossing.CategoryCrossing(num_bins=2)
    inputs_0 = np.asarray([[1, 2]])
    inputs_1 = np.asarray([[1, 3]])
    output = layer([inputs_0, inputs_1])
    self.assertAllClose([[1, 1, 0, 0]], output)

  def test_crossing_compute_output_signature(self):
    input_shapes = [
        tensor_shape.TensorShape([2, 2]),
        tensor_shape.TensorShape([2, 3])
    ]
    input_specs = [
        tensor_spec.TensorSpec(input_shape, dtypes.string)
        for input_shape in input_shapes
    ]
    layer = categorical_crossing.CategoryCrossing()
    output_spec = layer.compute_output_signature(input_specs)
    self.assertEqual(output_spec.shape.dims[0], input_shapes[0].dims[0])
    self.assertEqual(output_spec.dtype, dtypes.string)

    layer = categorical_crossing.CategoryCrossing(num_bins=2)
    output_spec = layer.compute_output_signature(input_specs)
    self.assertEqual(output_spec.shape.dims[0], input_shapes[0].dims[0])
    self.assertEqual(output_spec.dtype, dtypes.int64)

  def test_crossing_with_invalid_hash_key(self):
    with self.assertRaises(ValueError):
      _ = categorical_crossing.CategoryCrossing(hash_key=133)

  @tf_test_util.run_v2_only
  def test_config_with_custom_name(self):
    layer = categorical_crossing.CategoryCrossing(num_bins=2, name='hashing')
    config = layer.get_config()
    layer_1 = categorical_crossing.CategoryCrossing.from_config(config)
    self.assertEqual(layer_1.name, layer.name)

    layer = categorical_crossing.CategoryCrossing(name='hashing')
    config = layer.get_config()
    layer_1 = categorical_crossing.CategoryCrossing.from_config(config)
    self.assertEqual(layer_1.name, layer.name)


if __name__ == '__main__':
  test.main()
