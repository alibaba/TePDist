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
# pylint: disable=protected-access
"""A `Network` is way to compose layers: the topological form of a `Model`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools
import json
import os

import six
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import layer_utils as trackable_layer_utils
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util as trackable_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import serialization
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls


# pylint: disable=g-import-not-at-top
try:
  import h5py
except ImportError:
  h5py = None

try:
  import yaml
except ImportError:
  yaml = None
# pylint: enable=g-import-not-at-top


class Network(base_layer.Layer):
  """A `Network` is a composition of layers.

  `Network` is the topological form of a "model". A `Model`
  is simply a `Network` with added training routines.

  Two types of `Networks` exist: Graph Networks and Subclass Networks. Graph
  networks are used in the Keras Functional and Sequential APIs. Subclassed
  networks are used when a user subclasses the `Model` class. In general,
  more Keras features are supported with Graph Networks than with Subclassed
  Networks, specifically:

  - Model cloning (`keras.models.clone`)
  - Serialization (`model.get_config()/from_config`, `model.to_json()/to_yaml()`
  - Whole-model saving (`model.save()`)

  A Graph Network can be instantiated by passing two arguments to `__init__`.
  The first argument is the `keras.Input` Tensors that represent the inputs
  to the Network. The second argument specifies the output Tensors that
  represent the outputs of this Network. Both arguments can be a nested
  structure of Tensors.

  Example:

  ```
  inputs = {'x1': keras.Input(shape=(10,)), 'x2': keras.Input(shape=(1,))}
  t = keras.layers.Dense(1, activation='relu')(inputs['x1'])
  outputs = keras.layers.Add()([t, inputs['x2'])
  network = Network(inputs, outputs)
  ```

  A Graph Network constructed using the Functional API can also include raw
  TensorFlow functions, with the exception of functions that create Variables
  or assign ops.

  Example:

  ```
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(1)(inputs)
  outputs = tf.nn.relu(x)
  network = Network(inputs, outputs)
  ```

  Subclassed Networks can be instantiated via `name` and (optional) `dynamic`
  keyword arguments. Subclassed Networks keep track of their Layers, and their
  `call` method can be overridden. Subclassed Networks are typically created
  indirectly, by subclassing the `Model` class.

  Example:

  ```
  class MyModel(keras.Model):
    def __init__(self):
      super(MyModel, self).__init__(name='my_model', dynamic=False)

      self.layer1 = keras.layers.Dense(10, activation='relu')

    def call(self, inputs):
      return self.layer1(inputs)
  ```

  Allowed args in `super().__init__`:
    name: String name of the model.
    dynamic: (Subclassed models only) Set this to `True` if your model should
      only be run eagerly, and should not be used to generate a static
      computation graph. This attribute is automatically set for Functional API
      models.
    trainable: Boolean, whether the model's variables should be trainable.
    dtype: (Subclassed models only) Default dtype of the model's weights (
      default of `None` means use the type of the first input). This attribute
      has no effect on Functional API models, which do not have weights of their
      own.
  """

  # See tf.Module for the usage of this property.
  # The key of _layer_call_argspecs is a layer. tf.Module._flatten will fail to
  # flatten the key since it is trying to convert Trackable/Layer to a string.
  _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
      ('_layer_call_argspecs', '_compiled_trainable_state',
       '_output_mask_cache', '_output_tensor_cache', '_output_shape_cache'),
      base_layer.Layer._TF_MODULE_IGNORED_PROPERTIES
  ))

  def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
    # Signature detection
    if (len(args) == 2 or
        len(args) == 1 and 'outputs' in kwargs or
        'inputs' in kwargs and 'outputs' in kwargs):
      # Graph network
      self._init_graph_network(*args, **kwargs)
    else:
      # Subclassed network
      self._init_subclassed_network(**kwargs)

    tf_utils.assert_no_legacy_layers(self.layers)

  # Several Network methods have "no_automatic_dependency_tracking"
  # annotations. Since Network does automatic dependency tracking on attribute
  # assignment, including for common data structures such as lists, by default
  # we'd have quite a few empty dependencies which users don't care about (or
  # would need some way to ignore dependencies automatically, which is confusing
  # when applied to user code). Some attributes, such as _layers, would cause
  # structural issues (_layers being the place where Layers assigned to tracked
  # attributes are stored).
  #
  # Aside from these aesthetic and structural issues, useless dependencies on
  # empty lists shouldn't cause issues; adding or removing them will not break
  # checkpoints, but may cause "all Python objects matched" assertions to fail
  # (in which case less strict assertions may be substituted if necessary).
  @trackable.no_automatic_dependency_tracking
  def _base_init(self, **kwargs):
    # The following are implemented as property functions:
    # self.trainable_weights
    # self.non_trainable_weights
    # self.input_spec
    # self.losses
    # self.updates

    generic_utils.validate_kwargs(kwargs, {'trainable', 'dtype', 'dynamic',
                                           'name', 'autocast'})

    super(Network, self).__init__(**kwargs)

    self.input_names = None
    self.output_names = None
    self._saved_model_inputs_spec = None

    # This is True for Sequential networks and Functional networks.
    self._compute_output_and_mask_jointly = False

    # Don't reset compilation if already done. This may occur if calling
    # `__init__` (or `_init_graph_network`) on an already-compiled model
    # such as a Sequential model. Sequential models may need to rebuild
    # themselves after compilation.
    self._maybe_create_attribute('_is_compiled', False)
    self._maybe_create_attribute('optimizer', None)

    self._trackable_saver = (
        trackable_utils.saver_with_op_caching(self))

  @trackable.no_automatic_dependency_tracking
  def _init_graph_network(self, inputs, outputs, **kwargs):
    generic_utils.validate_kwargs(
        kwargs, {'name', 'trainable'},
        'Functional models may only specify `name` and `trainable` keyword '
        'arguments during initialization. Got an unexpected argument:')
    # Normalize and set self.inputs, self.outputs.
    if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
      inputs = inputs[0]
    if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
      outputs = outputs[0]
    self._nested_outputs = outputs
    self._nested_inputs = inputs
    self.inputs = nest.flatten(inputs)
    self.outputs = nest.flatten(outputs)

    # Models constructed with a single Tensor or list of Tensors can
    # be called with a dict, where the keys of the dict are the names
    # of the `Input` objects. Extra keys are ignored.
    self._enable_dict_to_input_mapping = (
        not nest.is_sequence(self._nested_inputs) or
        (isinstance(self._nested_inputs, (list, tuple)) and
         not any(nest.is_sequence(t) for t in self._nested_inputs)))

    if any(not hasattr(tensor, '_keras_history') for tensor in self.outputs):
      base_layer_utils.create_keras_history(self._nested_outputs)

    self._base_init(**kwargs)
    self._validate_graph_inputs_and_outputs()

    # A Network does not create weights of its own, thus it is already
    # built.
    self.built = True
    self._build_input_shape = nest.map_structure(lambda x: x.shape, inputs)
    self._compute_output_and_mask_jointly = True
    self._is_graph_network = True
    # `_expects_training_arg` is True since the `training` argument is always
    # present in the signature of the `call` method of a graph network.
    self._expects_training_arg = True
    self._expects_mask_arg = True
    # A graph network does not autocast inputs, as its layers will cast them
    # instead.
    self._autocast = False

    self._input_layers = []
    self._output_layers = []
    self._input_coordinates = []
    self._output_coordinates = []

    # This is for performance optimization when calling the Network on new
    # inputs. Every time the Network is called on a set on input tensors,
    # we compute the output tensors, output masks and output shapes in one pass,
    # then cache them here. When any of these outputs is queried later, we
    # retrieve it from there instead of recomputing it.
    self._output_mask_cache = {}
    self._output_tensor_cache = {}
    self._output_shape_cache = {}

    # Build self._output_layers:
    for x in self.outputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      self._output_layers.append(layer)
      self._output_coordinates.append((layer, node_index, tensor_index))

    # Build self._input_layers:
    for x in self.inputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      # It's supposed to be an input layer, so only one node
      # and one tensor output.
      assert node_index == 0
      assert tensor_index == 0
      self._input_layers.append(layer)
      self._input_coordinates.append((layer, node_index, tensor_index))

    # Keep track of the network's nodes and layers.
    nodes, nodes_by_depth, layers, _ = _map_graph_network(
        self.inputs, self.outputs)
    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._layers = layers
    self._layer_call_argspecs = {}
    for layer in self._layers:
      self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
      layer._attribute_sentinel.add_parent(self._attribute_sentinel)

    # Build self.input_names and self.output_names.
    self._set_output_names()
    self.input_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for layer in self._input_layers:
      self.input_names.append(layer.name)
      if layer.is_placeholder:
        self._feed_input_names.append(layer.name)
        # Use batch_input_shape here because non-eager composite tensors may not
        # have a shape attribute that's meaningful (sparse, for instance, has
        # a tensor that's non-constant and needs to be fed). This means that
        # input layers that create placeholders will need to have the
        # batch_input_shape attr to allow for input shape validation.
        self._feed_input_shapes.append(layer._batch_input_shape)
        self._feed_inputs.append(layer.input)

    self._compute_tensor_usage_count()
    self._set_save_spec(self._nested_inputs)

  @property
  def input(self):
    """Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input tensor or list of input tensors.

    Raises:
      RuntimeError: If called in Eager mode.
      AttributeError: If no inbound nodes are found.
    """
    if self._is_graph_network:
      return self._nested_inputs
    return super(Network, self).input

  @property
  def input_shape(self):
    """Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).

    Raises:
        AttributeError: if the layer has no defined input_shape.
        RuntimeError: if called in Eager mode.
    """
    if self._is_graph_network:
      return nest.map_structure(backend.int_shape, self.input)
    return super(Network, self).input_shape

  @property
  def output(self):
    """Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
      Output tensor or list of output tensors.

    Raises:
      AttributeError: if the layer is connected to more than one incoming
        layers.
      RuntimeError: if called in Eager mode.
    """
    if self._is_graph_network:
      return self._nested_outputs
    return super(Network, self).output

  @property
  def output_shape(self):
    """Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
        Output shape, as an integer shape tuple
        (or list of shape tuples, one tuple per output tensor).

    Raises:
        AttributeError: if the layer has no defined output shape.
        RuntimeError: if called in Eager mode.
    """
    if self._is_graph_network:
      return nest.map_structure(backend.int_shape, self.output)
    return super(Network, self).output_shape

  def _set_output_names(self):
    """Assigns unique names to the Network's outputs.

    Output layers with multiple output tensors would otherwise lead to duplicate
    names in self.output_names.
    """
    uniquified = []
    output_names = set()
    prefix_count = {}
    for layer in self._output_layers:
      proposal = layer.name
      while proposal in output_names:
        existing_count = prefix_count.get(layer.name, 1)
        proposal = '{}_{}'.format(layer.name, existing_count)
        prefix_count[layer.name] = existing_count + 1
      output_names.add(proposal)
      uniquified.append(proposal)
    self.output_names = uniquified

  @trackable.no_automatic_dependency_tracking
  def _init_subclassed_network(self, **kwargs):
    self._base_init(**kwargs)
    self._is_graph_network = False
    self.inputs = None
    self.outputs = None

  @property
  @trackable_layer_utils.cache_recursive_attribute('dynamic')
  def dynamic(self):
    if self._is_graph_network:
      return any(layer.dynamic for layer in self.layers)
    return self._dynamic or any(layer.dynamic for layer in self.layers)

  @property
  def _layer_checkpoint_dependencies(self):
    """Dictionary of layer dependencies to be included in the checkpoint."""
    # Use getattr because this function can be called from __setattr__, at which
    # point the _is_graph_network attribute has not been created.
    if (not getattr(self, '_is_graph_network', False) and
        base_layer_utils.is_subclassed(self)):
      return {}  # Only add layer dependencies for graph networks

    weight_layer_index = 0

    dependencies = collections.OrderedDict()
    for layer_index, layer in enumerate(self.layers):
      try:
        if layer.weights:
          # Keep a separate index for layers which have weights. This allows
          # users to insert Layers without weights anywhere in the network
          # without breaking checkpoints.
          dependencies['layer_with_weights-%d' % weight_layer_index] = layer
          weight_layer_index += 1
      except ValueError:
        # The layer might have weights, but may not be built yet. We just treat
        # it as layer without weight.
        pass

      # Even if it doesn't have weights, we should still track everything in
      # case it has/will have Trackable dependencies.
      dependencies['layer-%d' % layer_index] = layer
    return dependencies

  @property
  def _checkpoint_dependencies(self):
    dependencies = [
        trackable.TrackableReference(name=name, ref=layer)
        for name, layer in self._layer_checkpoint_dependencies.items()]
    dependencies.extend(super(Network, self)._checkpoint_dependencies)
    return dependencies

  def _lookup_dependency(self, name):
    layer_dependencies = self._layer_checkpoint_dependencies
    if name in layer_dependencies:
      return layer_dependencies[name]
    return super(Network, self)._lookup_dependency(name)

  def _handle_deferred_layer_dependencies(self, layers):
    """Handles layer checkpoint dependencies that are added after init."""
    layer_checkpoint_dependencies = self._layer_checkpoint_dependencies
    layer_to_name = {v: k for k, v in layer_checkpoint_dependencies.items()}
    for layer in layers:
      if layer in layer_to_name:
        self._handle_deferred_dependencies(name=layer_to_name[layer],
                                           trackable=layer)

  def __setattr__(self, name, value):
    if not getattr(self, '_self_setattr_tracking', True):
      super(Network, self).__setattr__(name, value)
      return

    if all(
        isinstance(v, (base_layer.Layer,
                       data_structures.TrackableDataStructure)) or
        trackable_layer_utils.has_weights(v) for v in nest.flatten(value)):
      try:
        self._is_graph_network
      except AttributeError:
        # six.raise_from supresses the original AttributeError from being raised
        six.raise_from(
            RuntimeError('It looks like you are subclassing `Model` and you '
                         'forgot to call `super(YourClass, self).__init__()`.'
                         ' Always start with this line.'), None)

    super(Network, self).__setattr__(name, value)

    # Keep track of metric instance created in subclassed model/layer.
    # We do this so that we can maintain the correct order of metrics by adding
    # the instance to the `metrics` list as soon as it is created.
    from tensorflow.python.keras import metrics as metrics_module  # pylint: disable=g-import-not-at-top
    if isinstance(value, metrics_module.Metric):
      self._metrics.append(value)

  @property
  @trackable_layer_utils.cache_recursive_attribute('stateful')
  def stateful(self):
    return any(getattr(layer, 'stateful', False) for layer in self.layers)

  def reset_states(self):
    for layer in self.layers:
      if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
        layer.reset_states()

  @property
  @deprecation.deprecated(
      date=None,
      instructions='This property should not be used in TensorFlow 2.0, '
      'as updates are applied automatically.')
  @doc_controls.do_not_generate_docs
  def state_updates(self):
    """Deprecated, do NOT use!

    Returns the `updates` from all layers that are stateful.

    This is useful for separating training updates and
    state updates, e.g. when we need to update a layer's internal state
    during prediction.

    Returns:
        A list of update ops.
    """
    state_updates = []
    for layer in self.layers:
      if getattr(layer, 'stateful', False):
        if hasattr(layer, 'updates'):
          state_updates += layer.updates
    return state_updates

  @property
  def weights(self):
    """Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    """
    return self._dedup_weights(self._undeduplicated_weights)

  @property
  def _undeduplicated_weights(self):
    """Returns the undeduplicated list of all layer variables/weights."""
    self._assert_weights_created()
    weights = []
    for layer in self._layers:
      weights += layer.weights
    weights += (self._trainable_weights + self._non_trainable_weights)
    return weights

  @property
  @tracking.cached_per_instance
  def _should_compute_mask(self):
    return self._is_graph_network and super(Network, self)._should_compute_mask

  def compute_mask(self, inputs, mask):
    if not self._is_graph_network:
      return None

    # TODO(omalleyt): b/123540974 This function is not really safe to call
    # by itself because it will duplicate any updates and losses in graph
    # mode by `call`ing the Layers again.
    output_tensors = self._run_internal_graph(inputs, mask=mask)
    return nest.map_structure(lambda t: t._keras_mask, output_tensors)

  @property
  def layers(self):
    return list(
        trackable_layer_utils.filter_empty_layer_containers(self._layers))

  def get_layer(self, name=None, index=None):
    """Retrieves a layer based on either its name (unique) or index.

    If `name` and `index` are both provided, `index` will take precedence.
    Indices are based on order of horizontal graph traversal (bottom-up).

    Arguments:
        name: String, name of layer.
        index: Integer, index of layer.

    Returns:
        A layer instance.

    Raises:
        ValueError: In case of invalid layer name or index.
    """
    # TODO(fchollet): We could build a dictionary based on layer names
    # since they are constant, but we have not done that yet.
    if index is not None and name is not None:
      raise ValueError('Provide only a layer name or a layer index.')

    if index is not None:
      if len(self.layers) <= index:
        raise ValueError('Was asked to retrieve layer at index ' + str(index) +
                         ' but model only has ' + str(len(self.layers)) +
                         ' layers.')
      else:
        return self.layers[index]

    if name is not None:
      for layer in self.layers:
        if layer.name == name:
          return layer
      raise ValueError('No such layer: ' + name + '.')
    raise ValueError('Provide either a layer name or layer index.')

  @property
  def trainable_weights(self):
    self._assert_weights_created()
    return self._dedup_weights(
        trackable_layer_utils.gather_trainable_weights(
            trainable=self.trainable,
            sub_layers=self._layers,
            extra_variables=self._trainable_weights))

  @property
  def non_trainable_weights(self):
    self._assert_weights_created()
    return self._dedup_weights(
        trackable_layer_utils.gather_non_trainable_weights(
            trainable=self.trainable,
            sub_layers=self._layers,
            extra_variables=self._non_trainable_weights +
            self._trainable_weights))

  @generic_utils.default
  def build(self, input_shape):
    """Builds the model based on input shapes received.

    This is to be used for subclassed models, which do not know at instantiation
    time what their inputs look like.

    This method only exists for users who want to call `model.build()` in a
    standalone way (as a substitute for calling the model on real data to
    build it). It will never be called by the framework (and thus it will
    never throw unexpected errors in an unrelated workflow).

    Args:
     input_shape: Single tuple, TensorShape, or list of shapes, where shapes
         are tuples, integers, or TensorShapes.

    Raises:
      ValueError:
        1. In case of invalid user-provided data (not of type tuple,
           list, or TensorShape).
        2. If the model requires call arguments that are agnostic
           to the input shapes (positional or kwarg in call signature).
        3. If not all layers were properly built.
        4. If float type inputs are not supported within the layers.

      In each of these cases, the user should build their model by calling it
      on real tensor data.
    """
    if self._is_graph_network:
      super(Network, self).build(input_shape)
      return

    # If subclass network
    if input_shape is None:
      raise ValueError('Input shape must be defined when calling build on a '
                       'model subclass network.')
    valid_types = (tuple, list, tensor_shape.TensorShape)
    if not isinstance(input_shape, valid_types):
      raise ValueError('Specified input shape is not one of the valid types. '
                       'Please specify a batch input shape of type tuple or '
                       'list of input shapes. User provided '
                       'input type: {}'.format(type(input_shape)))

    if input_shape and not self.inputs:
      # We create placeholders for the `None`s in the shape and build the model
      # in a Graph. Since tf.Variable is compatible with both eager execution
      # and graph building, the variables created after building the model in
      # a Graph are still valid when executing eagerly.
      if context.executing_eagerly():
        graph = func_graph.FuncGraph('build_graph')
      else:
        graph = backend.get_graph()
      with graph.as_default():
        if isinstance(input_shape, list):
          x = [base_layer_utils.generate_placeholders_from_shape(shape)
               for shape in input_shape]
        elif isinstance(input_shape, dict):
          x = {
              k: base_layer_utils.generate_placeholders_from_shape(shape)
              for k, shape in input_shape.items()
          }
        else:
          x = base_layer_utils.generate_placeholders_from_shape(input_shape)

        kwargs = {}
        call_signature = self._call_full_argspec
        call_args = call_signature.args
        # Exclude `self`, `inputs`, and any argument with a default value.
        if len(call_args) > 2:
          if call_signature.defaults:
            call_args = call_args[2:-len(call_signature.defaults)]
          else:
            call_args = call_args[2:]
          for arg in call_args:
            if arg == 'training':
              # Case where `training` is a positional arg with no default.
              kwargs['training'] = False
            else:
              # Has invalid call signature with unknown positional arguments.
              raise ValueError(
                  'Currently, you cannot build your model if it has '
                  'positional or keyword arguments that are not '
                  'inputs to the model, but are required for its '
                  '`call` method. Instead, in order to instantiate '
                  'and build your model, `call` your model on real '
                  'tensor data with all expected call arguments.')
        elif len(call_args) < 2:
          # Signature without `inputs`.
          raise ValueError('You can only call `build` on a model if its `call` '
                           'method accepts an `inputs` argument.')
        try:
          self.call(x, **kwargs)
        except (errors.InvalidArgumentError, TypeError):
          raise ValueError('You cannot build your model by calling `build` '
                           'if your layers do not support float type inputs. '
                           'Instead, in order to instantiate and build your '
                           'model, `call` your model on real tensor data (of '
                           'the correct dtype).')

    super(Network, self).build(input_shape)

  def call(self, inputs, training=None, mask=None):
    """Calls the model on new inputs.

    In this case `call` just reapplies
    all ops in the graph to the new inputs
    (e.g. build a new computational graph from the provided inputs).

    Arguments:
        inputs: A tensor or list of tensors.
        training: Boolean or boolean scalar tensor, indicating whether to run
          the `Network` in training mode or inference mode.
        mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

    Returns:
        A tensor if there is a single output, or
        a list of tensors if there are more than one outputs.
    """
    if not self._is_graph_network:
      raise NotImplementedError('When subclassing the `Model` class, you should'
                                ' implement a `call` method.')

    return self._run_internal_graph(
        inputs, training=training, mask=mask)

  def compute_output_shape(self, input_shape):
    if not self._is_graph_network:
      return super(Network, self).compute_output_shape(input_shape)

    # Convert any shapes in tuple format to TensorShapes.
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

    if len(nest.flatten(input_shape)) != len(nest.flatten(self._input_layers)):
      raise ValueError('Invalid input_shape argument ' + str(input_shape) +
                       ': model has ' + str(len(self._input_layers)) +
                       ' tensor inputs.')

    # Use the tuple of TensorShape as the cache key, since tuple is hashable
    # and can be used as hash key.
    try:
      cache_key = tuple(tf_utils.convert_shapes(input_shape, to_tuples=True))
      if cache_key in self._output_shape_cache:
        # Cache hit. Return shapes as TensorShapes.
        return self._output_shape_cache[cache_key]
    except ValueError:
      # In case there are unknown TensorShape, eg for sparse tensor input,
      # We skip the caching since the shape is unknown.
      pass

    layers_to_output_shapes = {}
    for layer, shape in zip(self._input_layers, nest.flatten(input_shape)):
      # It's an input layer: then `compute_output_shape` is identity,
      # and there is only one node and one tensor..
      shape_key = layer.name + '_0_0'
      layers_to_output_shapes[shape_key] = shape

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    # Iterate over nodes, by depth level.
    if len(depth_keys) > 1:
      for depth in depth_keys:
        nodes = self._nodes_by_depth[depth]
        for node in nodes:
          layer = node.layer
          if layer in self._input_layers:
            # We've already covered the input layers
            # a few lines above.
            continue
          # Get the input shapes for the first argument of the node
          layer_input_shapes = []
          layer_inputs = node.call_args[0]
          for layer_input in nest.flatten(layer_inputs):
            kh = layer_input._keras_history
            input_layer_key = kh.layer.name + '_%s_%s' % (kh.node_index,
                                                          kh.tensor_index)
            layer_input_shapes.append(layers_to_output_shapes[input_layer_key])
          layer_input_shapes = nest.pack_sequence_as(layer_inputs,
                                                     layer_input_shapes)
          # Layers expect shapes to be tuples for `compute_output_shape`.
          layer_input_shapes = tf_utils.convert_shapes(
              layer_input_shapes, to_tuples=True)
          layer_output_shapes = layer.compute_output_shape(layer_input_shapes)
          # Convert back to TensorShapes.
          layer_output_shapes = tf_utils.convert_shapes(
              layer_output_shapes, to_tuples=False)

          node_index = layer._inbound_nodes.index(node)  # pylint: disable=protected-access
          for j, shape in enumerate(nest.flatten(layer_output_shapes)):
            shape_key = layer.name + '_%s_%s' % (node_index, j)
            layers_to_output_shapes[shape_key] = shape

      # Read final output shapes from layers_to_output_shapes.
      output_shapes = []
      for i in range(len(self._output_layers)):
        layer, node_index, tensor_index = self._output_coordinates[i]
        shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
        output_shapes.append(layers_to_output_shapes[shape_key])
      output_shapes = nest.pack_sequence_as(self._nested_outputs, output_shapes)
      # Store in cache.
      self._output_shape_cache[cache_key] = output_shapes

    # Return shapes as TensorShapes.
    return output_shapes

  def _run_internal_graph(self, inputs, training=None, mask=None):
    """Computes output tensors for new inputs.

    # Note:
        - Can be run on non-Keras tensors.

    Arguments:
        inputs: Tensor or nested structure of Tensors.
        training: Boolean learning phase.
        mask: (Optional) Tensor or nested structure of Tensors.

    Returns:
        Two lists: output_tensors, output_masks
    """
    inputs = self._flatten_to_reference_inputs(inputs)
    if mask is None:
      masks = [None for _ in range(len(inputs))]
    else:
      masks = self._flatten_to_reference_inputs(mask)
    for input_t, mask in zip(inputs, masks):
      input_t._keras_mask = mask

    # Dictionary mapping reference tensors to computed tensors.
    tensor_dict = {}
    for x, y in zip(self.inputs, inputs):
      y = self._conform_to_reference_input(y, ref_input=x)
      x_id = str(id(x))
      tensor_dict[x_id] = [y] * self._tensor_usage_count[x_id]

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    for depth in depth_keys:
      nodes = self._nodes_by_depth[depth]
      for node in nodes:
        if node.is_input:
          continue  # Input tensors already exist.

        if not all(
            str(id(tensor)) in tensor_dict
            for tensor in nest.flatten(node.keras_inputs)):
          continue  # Node is not computable, try skipping.

        layer = node.layer
        args, kwargs = node.map_arguments(tensor_dict)
        outputs = layer(*args, **kwargs)

        # Update tensor_dict.
        for x, y in zip(nest.flatten(node.outputs), nest.flatten(outputs)):
          x_id = str(id(x))
          tensor_dict[x_id] = [y] * self._tensor_usage_count[x_id]

    output_tensors = []
    output_shapes = []
    for x in self.outputs:
      assert str(id(x)) in tensor_dict, 'Could not compute output ' + str(x)
      tensor = tensor_dict[str(id(x))].pop()
      output_shapes.append(x.shape)
      output_tensors.append(tensor)

    if output_shapes is not None:
      input_shapes = [x.shape for x in inputs]
      try:
        cache_key = tuple(tf_utils.convert_shapes(input_shapes, to_tuples=True))
        self._output_shape_cache[cache_key] = nest.pack_sequence_as(
            self._nested_outputs, output_shapes)
      except ValueError:
        # In case there are unknown TensorShape, eg for sparse tensor input,
        # We skip the caching since the shape is unknown.
        pass

    output_tensors = nest.pack_sequence_as(self._nested_outputs, output_tensors)
    return output_tensors

  def _flatten_to_reference_inputs(self, tensors):
    """Maps `tensors` to their respective `keras.Input`."""
    if self._enable_dict_to_input_mapping and isinstance(tensors, dict):
      ref_inputs = self._nested_inputs
      if not nest.is_sequence(ref_inputs):
        ref_inputs = [self._nested_inputs]

      try:
        # Flatten in the order `Input`s were passed during Model construction.
        return [tensors[inp._keras_history.layer.name] for inp in ref_inputs]
      except KeyError:
        # TODO(b/151582614)
        return nest.flatten(tensors)

    # Otherwise both self.inputs and tensors will already be in same order.
    return nest.flatten(tensors)

  def _conform_to_reference_input(self, tensor, ref_input):
    """Set shape and dtype based on `keras.Input`s."""
    # Shape handling (only for non-CompositeTensors).
    if isinstance(tensor, ops.Tensor) and isinstance(ref_input, ops.Tensor):
      # Allow (None,) and (None, 1) Tensors to be passed interchangably. Use the
      # shape specified by the `keras.Input`.
      if tensor.shape.rank is not None and ref_input.shape.rank is not None:
        should_squeeze_last_dim = (
            tensor.shape.rank == ref_input.shape.rank + 1 and
            tensor.shape[-1] == 1)
        should_expand_last_dim = (
            tensor.shape.rank == ref_input.shape.rank - 1 and
            ref_input.shape[-1] == 1)
        if should_squeeze_last_dim:
          tensor = array_ops.squeeze_v2(tensor, axis=-1)
        elif should_expand_last_dim:
          tensor = array_ops.expand_dims_v2(tensor, axis=-1)

      # Add shape hints to Tensors that might have None shape dims but have
      # shapes defined by the `keras.Input`.
      try:
        tensor.set_shape(tensor.shape.merge_with(ref_input.shape))
      except ValueError:
        logging.warning(
            'Model was constructed with shape {} for input {}, but it was '
            'called on an input with incompatible shape {}.'.format(
                ref_input.shape, ref_input, tensor.shape))

    # Dtype handling.
    if isinstance(ref_input, (ops.Tensor, composite_tensor.CompositeTensor)):
      tensor = math_ops.cast(tensor, dtype=ref_input.dtype)

    return tensor

  def get_config(self):
    if not self._is_graph_network:
      raise NotImplementedError
    return copy.deepcopy(get_network_config(self))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Instantiates a Model from its config (output of `get_config()`).

    Arguments:
        config: Model config dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A model instance.

    Raises:
        ValueError: In case of improperly formatted config dict.
    """
    input_tensors, output_tensors, created_layers = reconstruct_from_config(
        config, custom_objects)
    model = cls(inputs=input_tensors, outputs=output_tensors,
                name=config.get('name'))
    connect_ancillary_layers(model, created_layers)
    return model

  def save(self,
           filepath,
           overwrite=True,
           include_optimizer=True,
           save_format=None,
           signatures=None,
           options=None):
    """Saves the model to Tensorflow SavedModel or a single HDF5 file.

    The savefile includes:

    - The model architecture, allowing to re-instantiate the model.
    - The model weights.
    - The state of the optimizer, allowing to resume training
        exactly where you left off.

    This allows you to save the entirety of the state of a model
    in a single file.

    Saved models can be reinstantiated via `keras.models.load_model`.
    The model returned by `load_model` is a compiled model ready to be used
    (unless the saved model was never compiled in the first place).

    Models built with the Sequential and Functional API can be saved to both the
    HDF5 and SavedModel formats. Subclassed models can only be saved with the
    SavedModel format.

    Note that the model weights may have different scoped names after being
    loaded. Scoped names include the model/layer names, such as
    `"dense_1/kernel:0"`. It is recommended that you use the layer properties to
     access specific variables, e.g. `model.get_layer("dense_1").kernel`.

    Arguments:
        filepath: String, PathLike, path to SavedModel or H5 file to save the
            model.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.
        save_format: Either `'tf'` or `'h5'`, indicating whether to save the
            model to Tensorflow SavedModel or HDF5. Defaults to 'tf' in TF 2.X,
            and 'h5' in TF 1.X.
        signatures: Signatures to save with the SavedModel. Applicable to the
            'tf' format only. Please see the `signatures` argument in
            `tf.saved_model.save` for details.
        options: Optional `tf.saved_model.SaveOptions` object that specifies
            options for saving to SavedModel.

    Example:

    ```python
    from keras.models import load_model

    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model

    # returns a compiled model
    # identical to the previous one
    model = load_model('my_model.h5')
    ```
    """
    save.save_model(self, filepath, overwrite, include_optimizer, save_format,
                    signatures, options)

  def save_weights(self, filepath, overwrite=True, save_format=None):
    """Saves all layer weights.

    Either saves in HDF5 or in TensorFlow format based on the `save_format`
    argument.

    When saving in HDF5 format, the weight file has:
      - `layer_names` (attribute), a list of strings
          (ordered names of model layers).
      - For every layer, a `group` named `layer.name`
          - For every such layer group, a group attribute `weight_names`,
              a list of strings
              (ordered names of weights tensor of the layer).
          - For every weight in the layer, a dataset
              storing the weight value, named after the weight tensor.

    When saving in TensorFlow format, all objects referenced by the network are
    saved in the same format as `tf.train.Checkpoint`, including any `Layer`
    instances or `Optimizer` instances assigned to object attributes. For
    networks constructed from inputs and outputs using `tf.keras.Model(inputs,
    outputs)`, `Layer` instances used by the network are tracked/saved
    automatically. For user-defined classes which inherit from `tf.keras.Model`,
    `Layer` instances must be assigned to object attributes, typically in the
    constructor. See the documentation of `tf.train.Checkpoint` and
    `tf.keras.Model` for details.

    While the formats are the same, do not mix `save_weights` and
    `tf.train.Checkpoint`. Checkpoints saved by `Model.save_weights` should be
    loaded using `Model.load_weights`. Checkpoints saved using
    `tf.train.Checkpoint.save` should be restored using the corresponding
    `tf.train.Checkpoint.restore`. Prefer `tf.train.Checkpoint` over
    `save_weights` for training checkpoints.

    The TensorFlow format matches objects and variables by starting at a root
    object, `self` for `save_weights`, and greedily matching attribute
    names. For `Model.save` this is the `Model`, and for `Checkpoint.save` this
    is the `Checkpoint` even if the `Checkpoint` has a model attached. This
    means saving a `tf.keras.Model` using `save_weights` and loading into a
    `tf.train.Checkpoint` with a `Model` attached (or vice versa) will not match
    the `Model`'s variables. See the [guide to training
    checkpoints](https://www.tensorflow.org/guide/checkpoint) for details
    on the TensorFlow format.

    Arguments:
        filepath: String or PathLike, path to the file to save the weights to.
            When saving in TensorFlow format, this is the prefix used for
            checkpoint files (multiple files are generated). Note that the '.h5'
            suffix causes weights to be saved in HDF5 format.
        overwrite: Whether to silently overwrite any existing file at the
            target location, or provide the user with a manual prompt.
        save_format: Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
            '.keras' will default to HDF5 if `save_format` is `None`. Otherwise
            `None` defaults to 'tf'.

    Raises:
        ImportError: If h5py is not available when attempting to save in HDF5
            format.
        ValueError: For invalid/unknown format arguments.
    """
    self._assert_weights_created()
    filepath = path_to_string(filepath)
    filepath_is_h5 = _is_hdf5_filepath(filepath)
    if save_format is None:
      if filepath_is_h5:
        save_format = 'h5'
      else:
        save_format = 'tf'
    else:
      user_format = save_format.lower().strip()
      if user_format in ('tensorflow', 'tf'):
        save_format = 'tf'
      elif user_format in ('hdf5', 'h5', 'keras'):
        save_format = 'h5'
      else:
        raise ValueError(
            'Unknown format "%s". Was expecting one of {"tf", "h5"}.' % (
                save_format,))
    if save_format == 'tf' and filepath_is_h5:
      raise ValueError(
          ('save_weights got save_format="tf"/"tensorflow", but the '
           'filepath ("%s") looks like an HDF5 file. Omit the ".h5"/".keras" '
           'when saving in TensorFlow format.')
          % filepath)

    if save_format == 'h5' and h5py is None:
      raise ImportError(
          '`save_weights` requires h5py when saving in hdf5.')
    if save_format == 'tf':
      check_filepath = filepath + '.index'
    else:
      check_filepath = filepath
    # If file exists and should not be overwritten:
    if not overwrite and os.path.isfile(check_filepath):
      proceed = ask_to_proceed_with_overwrite(check_filepath)
      if not proceed:
        return
    if save_format == 'h5':
      with h5py.File(filepath, 'w') as f:
        hdf5_format.save_weights_to_hdf5_group(f, self.layers)
    else:
      if context.executing_eagerly():
        session = None
      else:
        session = backend.get_session()
      optimizer = getattr(self, 'optimizer', None)
      if (optimizer
          and not isinstance(optimizer, trackable.Trackable)):
        logging.warning(
            ('This model was compiled with a Keras optimizer (%s) but is being '
             'saved in TensorFlow format with `save_weights`. The model\'s '
             'weights will be saved, but unlike with TensorFlow optimizers in '
             'the TensorFlow format the optimizer\'s state will not be '
             'saved.\n\nConsider using a TensorFlow optimizer from `tf.train`.')
            % (optimizer,))
      self._trackable_saver.save(filepath, session=session)
      # Record this checkpoint so it's visible from tf.train.latest_checkpoint.
      checkpoint_management.update_checkpoint_state_internal(
          save_dir=os.path.dirname(filepath),
          model_checkpoint_path=filepath,
          save_relative_paths=True,
          all_model_checkpoint_paths=[filepath])

  def load_weights(self, filepath, by_name=False, skip_mismatch=False):
    """Loads all layer weights, either from a TensorFlow or an HDF5 weight file.

    If `by_name` is False weights are loaded based on the network's
    topology. This means the architecture should be the same as when the weights
    were saved.  Note that layers that don't have weights are not taken into
    account in the topological ordering, so adding or removing layers is fine as
    long as they don't have weights.

    If `by_name` is True, weights are loaded into layers only if they share the
    same name. This is useful for fine-tuning or transfer-learning models where
    some of the layers have changed.

    Only topological loading (`by_name=False`) is supported when loading weights
    from the TensorFlow format. Note that topological loading differs slightly
    between TensorFlow and HDF5 formats for user-defined classes inheriting from
    `tf.keras.Model`: HDF5 loads based on a flattened list of weights, while the
    TensorFlow format loads based on the object-local names of attributes to
    which layers are assigned in the `Model`'s constructor.

    Arguments:
        filepath: String or PathLike, path to the weights file to load. For
            weight files in TensorFlow format, this is the file prefix (the
            same as was passed to `save_weights`).
        by_name: Boolean, whether to load weights by name or by topological
            order. Only topological loading is supported for weight files in
            TensorFlow format.
        skip_mismatch: Boolean, whether to skip loading of layers where there is
            a mismatch in the number of weights, or a mismatch in the shape of
            the weight (only valid when `by_name=True`).

    Returns:
        When loading a weight file in TensorFlow format, returns the same status
        object as `tf.train.Checkpoint.restore`. When graph building, restore
        ops are run automatically as soon as the network is built (on first call
        for user-defined classes inheriting from `Model`, immediately if it is
        already built).

        When loading weights in HDF5 format, returns `None`.

    Raises:
        ImportError: If h5py is not available and the weight file is in HDF5
            format.
        ValueError: If `skip_mismatch` is set to `True` when `by_name` is
          `False`.
    """

    if skip_mismatch and not by_name:
      raise ValueError(
          'When calling model.load_weights, skip_mismatch can only be set to '
          'True when by_name is True.')

    filepath = path_to_string(filepath)
    if _is_hdf5_filepath(filepath):
      save_format = 'h5'
    else:
      try:
        py_checkpoint_reader.NewCheckpointReader(filepath)
        save_format = 'tf'
      except errors_impl.DataLossError:
        # The checkpoint is not readable in TensorFlow format. Try HDF5.
        save_format = 'h5'
    if save_format == 'tf':
      status = self._trackable_saver.restore(filepath)
      if by_name:
        raise NotImplementedError(
            'Weights may only be loaded based on topology into Models when '
            'loading TensorFlow-formatted weights (got by_name=True to '
            'load_weights).')
      if not context.executing_eagerly():
        session = backend.get_session()
        # Restore existing variables (if any) immediately, and set up a
        # streaming restore for any variables created in the future.
        trackable_utils.streaming_restore(status=status, session=session)
      status.assert_nontrivial_match()
      return status
    if h5py is None:
      raise ImportError(
          '`load_weights` requires h5py when loading weights from HDF5.')
    if self._is_graph_network and not self.built:
      raise NotImplementedError(
          'Unable to load weights saved in HDF5 format into a subclassed '
          'Model which has not created its variables yet. Call the Model '
          'first, then load the weights.')
    self._assert_weights_created()
    with h5py.File(filepath, 'r') as f:
      if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']
      if by_name:
        hdf5_format.load_weights_from_hdf5_group_by_name(
            f, self.layers, skip_mismatch=skip_mismatch)
      else:
        hdf5_format.load_weights_from_hdf5_group(f, self.layers)

  def _updated_config(self):
    """Util shared between different serialization methods.

    Returns:
        Model config with Keras version information added.
    """
    from tensorflow.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

    config = self.get_config()
    model_config = {
        'class_name': self.__class__.__name__,
        'config': config,
        'keras_version': keras_version,
        'backend': backend.backend()
    }
    return model_config

  def to_json(self, **kwargs):
    """Returns a JSON string containing the network configuration.

    To load a network from a JSON save file, use
    `keras.models.model_from_json(json_string, custom_objects={})`.

    Arguments:
        **kwargs: Additional keyword arguments
            to be passed to `json.dumps()`.

    Returns:
        A JSON string.
    """
    model_config = self._updated_config()
    return json.dumps(
        model_config, default=serialization.get_json_type, **kwargs)

  def to_yaml(self, **kwargs):
    """Returns a yaml string containing the network configuration.

    To load a network from a yaml save file, use
    `keras.models.model_from_yaml(yaml_string, custom_objects={})`.

    `custom_objects` should be a dictionary mapping
    the names of custom losses / layers / etc to the corresponding
    functions / classes.

    Arguments:
        **kwargs: Additional keyword arguments
            to be passed to `yaml.dump()`.

    Returns:
        A YAML string.

    Raises:
        ImportError: if yaml module is not found.
    """
    if yaml is None:
      raise ImportError(
          'Requires yaml module installed (`pip install pyyaml`).')
    return yaml.dump(self._updated_config(), **kwargs)

  def summary(self, line_length=None, positions=None, print_fn=None):
    """Prints a string summary of the network.

    Arguments:
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements
            in each line. If not provided,
            defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use. Defaults to `print`.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.

    Raises:
        ValueError: if `summary()` is called before the model is built.
    """
    if not self.built:
      raise ValueError('This model has not yet been built. '
                       'Build the model first by calling `build()` or calling '
                       '`fit()` with some data, or specify '
                       'an `input_shape` argument in the first layer(s) for '
                       'automatic build.')
    layer_utils.print_summary(self,
                              line_length=line_length,
                              positions=positions,
                              print_fn=print_fn)

  def _validate_graph_inputs_and_outputs(self):
    """Validates the inputs and outputs of a Graph Network."""
    # Check for redundancy in inputs.
    if len({id(i) for i in self.inputs}) != len(self.inputs):
      raise ValueError('The list of inputs passed to the model '
                       'is redundant. '
                       'All inputs should only appear once.'
                       ' Found: ' + str(self.inputs))

    for x in self.inputs:
      # Check that x has appropriate `_keras_history` metadata.
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Input tensors to a ' + cls_name + ' ' +
                         'must come from `tf.keras.Input`. '
                         'Received: ' + str(x) +
                         ' (missing previous layer metadata).')
      # Check that x is an input tensor.
      # pylint: disable=protected-access
      layer = x._keras_history.layer
      if len(layer._inbound_nodes) > 1 or (
          layer._inbound_nodes and not layer._inbound_nodes[0].is_input):
        cls_name = self.__class__.__name__
        logging.warning(cls_name + ' inputs must come from '
                        '`tf.keras.Input` (thus holding past layer metadata), '
                        'they cannot be the output of '
                        'a previous non-Input layer. '
                        'Here, a tensor specified as '
                        'input to "' + self.name + '" was not an Input tensor, '
                        'it was generated by layer ' + layer.name + '.\n'
                        'Note that input tensors are '
                        'instantiated via `tensor = tf.keras.Input(shape)`.\n'
                        'The tensor that caused the issue was: ' + str(x.name))

    # Check compatibility of batch sizes of Input Layers.
    input_batch_sizes = [
        training_utils.get_static_batch_size(x._keras_history.layer)
        for x in self.inputs
    ]
    consistent_batch_size = None
    for batch_size in input_batch_sizes:
      if batch_size is not None:
        if (consistent_batch_size is not None and
            batch_size != consistent_batch_size):
          raise ValueError('The specified batch sizes of the Input Layers'
                           ' are incompatible. Found batch sizes: {}'.format(
                               input_batch_sizes))
        consistent_batch_size = batch_size

    for x in self.outputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Output tensors to a ' + cls_name + ' must be '
                         'the output of a TensorFlow `Layer` '
                         '(thus holding past layer metadata). Found: ' + str(x))

  def _insert_layers(self, layers, relevant_nodes=None):
    """Inserts Layers into the Network after Network creation.

    This is only valid for Keras Graph Networks.  Layers added via this function
    will be included in the `call` computation and `get_config` of this Network.
    They will not be added to the Network's outputs.


    Arguments:
      layers: Arbitrary nested structure of Layers. Layers must be reachable
        from one or more of the `keras.Input` Tensors that correspond to this
        Network's inputs.
      relevant_nodes: Nodes from the Layers that should be considered part of
        this Network. If `None`, all Nodes will be considered part of this
        Network.

    Raises:
      ValueError: If the layers depend on `Input`s not found in this Model.
    """
    layers = nest.flatten(layers)
    tf_utils.assert_no_legacy_layers(layers)
    node_to_depth = {}
    for depth, nodes in self._nodes_by_depth.items():
      node_to_depth.update({node: depth for node in nodes})
    # The nodes of these Layers that are relevant to this Network. If not
    # provided, assume all Nodes are relevant
    if not relevant_nodes:
      relevant_nodes = nest.flatten([layer._inbound_nodes for layer in layers])
    network_nodes = set(relevant_nodes + list(node_to_depth.keys()))

    def _get_min_depth(node):
      """Gets the minimum depth at which node can be computed."""
      min_depth = 0
      for layer, node_id, _, _ in node.iterate_inbound():
        inbound_node = layer._inbound_nodes[node_id]
        if inbound_node in node_to_depth:
          min_depth = min(min_depth, node_to_depth[inbound_node])
        elif inbound_node not in network_nodes:
          continue
        else:
          # Previous relevant nodes haven't been processed yet.
          return None
      # New node is one shallower than its shallowest input.
      return min_depth - 1

    # Insert nodes into `_nodes_by_depth` and other node attrs.
    unprocessed_nodes = copy.copy(relevant_nodes)
    i = 0
    while unprocessed_nodes:
      i += 1
      # Do a sanity check. This can occur if `Input`s from outside this Model
      # are being relied on.
      if i > 10000:
        raise ValueError('Layers could not be added due to missing '
                         'dependencies.')

      node = unprocessed_nodes.pop(0)
      depth = _get_min_depth(node)
      if depth is None:  # Defer until inbound nodes are processed.
        unprocessed_nodes.append(node)
        continue
      node_key = _make_node_key(node.layer.name,
                                node.layer._inbound_nodes.index(node))
      if node_key not in self._network_nodes:
        node_to_depth[node] = depth
        self._network_nodes.add(node_key)
        self._nodes_by_depth[depth].append(node)

    # Insert layers and update other layer attrs.
    layer_set = set(self._layers)
    deferred_layers = []
    for layer in layers:
      if layer not in layer_set:
        self._layers.append(layer)
        deferred_layers.append(layer)
        self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)

        # This allows the added layer to broadcast mutations to the current
        # layer, which is necessary to ensure cache correctness.
        layer._attribute_sentinel.add_parent(self._attribute_sentinel)
        layer_set.add(layer)
    self._handle_deferred_layer_dependencies(deferred_layers)

    self._compute_tensor_usage_count()

  def _compute_tensor_usage_count(self):
    """Compute the #. of tensor usages for all the output tensors of layers.

    The computed tensor usage count is saved as `self._tensor_usage_count`. This
    is later used for saving memory in eager computation by releasing
    no-longer-needed tensors as early as possible.
    """
    tensor_usage_count = collections.Counter()
    available_tensors = set(str(id(tensor)) for tensor in self.inputs)

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    depth_keys = depth_keys[1:]

    for depth in depth_keys:
      for node in self._nodes_by_depth[depth]:
        input_tensors = {
            str(id(tensor)) for tensor in nest.flatten(node.keras_inputs)
        }
        if input_tensors.issubset(available_tensors):
          for tensor in nest.flatten(node.keras_inputs):
            tensor_usage_count[str(id(tensor))] += 1

          for output_tensor in nest.flatten(node.outputs):
            available_tensors.add(str(id(output_tensor)))

    for tensor in self.outputs:
      tensor_usage_count[str(id(tensor))] += 1

    self._tensor_usage_count = tensor_usage_count

  def _assert_weights_created(self):
    """Asserts that all the weights for the network have been created.

    For a non-dynamic network, the weights must already be created after the
    layer has been called. For a dynamic network, the exact list of weights can
    never be known for certain since it may change at any time during execution.

    We run this check right before accessing weights or getting the Numpy value
    for the current weights. Otherwise, if the layer has never been called,
    the user would just get an empty list, which is misleading.

    Raises:
      ValueError: if the weights of the network has not yet been created.
    """
    if self.dynamic:
      return
    if (not self._is_graph_network and
        'build' in self.__class__.__dict__ and
        not self.built):
      # For any model that has customized build() method but hasn't
      # been invoked yet, this will cover both sequential and subclass model.
      raise ValueError('Weights for model %s have not yet been created. '
                       'Weights are created when the Model is first called on '
                       'inputs or `build()` is called with an `input_shape`.' %
                       self.name)

  def _graph_network_add_loss(self, symbolic_loss):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [symbolic_loss])
    # Losses must be keyed on inputs no matter what in order to be supported in
    # DistributionStrategy.
    add_loss_layer = base_layer.AddLoss(
        unconditional=False, dtype=symbolic_loss.dtype)
    add_loss_layer(symbolic_loss)
    new_nodes.extend(add_loss_layer.inbound_nodes)
    new_layers.append(add_loss_layer)
    self._insert_layers(new_layers, new_nodes)

  def _graph_network_add_metric(self, value, aggregation, name):
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [value])
    add_metric_layer = base_layer.AddMetric(
        aggregation, name, dtype=value.dtype)
    add_metric_layer(value)
    new_nodes.extend(add_metric_layer.inbound_nodes)
    new_layers.append(add_metric_layer)
    self._insert_layers(new_layers, new_nodes)

  @trackable.no_automatic_dependency_tracking
  def _set_save_spec(self, inputs):
    if self._saved_model_inputs_spec is not None:
      return  # Already set.

    input_names = self.input_names
    if not input_names:
      input_names = compile_utils.create_pseudo_input_names(inputs)

    flat_inputs = nest.flatten(inputs)
    specs = []
    for name, tensor in zip(input_names, flat_inputs):
      specs.append(
          tf_utils.get_tensor_spec(tensor, dynamic_batch=False, name=name))
    specs = nest.pack_sequence_as(inputs, specs)

    self._saved_model_inputs_spec = specs

  def _get_save_spec(self, dynamic_batch=True):
    if self._saved_model_inputs_spec is None:
      return None

    return nest.map_structure(
        lambda t: tf_utils.get_tensor_spec(t, dynamic_batch=dynamic_batch),
        self._saved_model_inputs_spec)

  @property
  def _trackable_saved_model_saver(self):
    return network_serialization.NetworkSavedModelSaver(self)


def _is_hdf5_filepath(filepath):
  return (filepath.endswith('.h5') or filepath.endswith('.keras') or
          filepath.endswith('.hdf5'))


def _make_node_key(layer_name, node_index):
  return layer_name + '_ib-' + str(node_index)


def _map_graph_network(inputs, outputs):
  """Validates a network's topology and gather its layers and nodes.

  Arguments:
    inputs: List of input tensors.
    outputs: List of outputs tensors.

  Returns:
    A tuple `(nodes, nodes_by_depth, layers, layers_by_depth)`.
    - nodes: list of Node instances.
    - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
    - layers: list of Layer instances.
    - layers_by_depth: dict mapping ints (depth) to lists of layer instances.

  Raises:
    ValueError: In case the network is not valid (e.g. disconnected graph).
  """
  # "depth" is number of layers between output Node and the Node.
  # Nodes are ordered from inputs -> outputs.
  nodes_in_decreasing_depth, layer_indices = _build_map(outputs)
  network_nodes = {
      _make_node_key(node.layer.name, node.layer._inbound_nodes.index(node))
      for node in nodes_in_decreasing_depth
  }

  nodes_depths = {}  # dict {node: depth value}
  layers_depths = {}  # dict {layer: depth value}

  for node in reversed(nodes_in_decreasing_depth):
    # If the depth is not set, the node has no outbound nodes (depth 0).
    depth = nodes_depths.setdefault(node, 0)

    # Update the depth of the corresponding layer
    previous_depth = layers_depths.get(node.layer, 0)
    # If we've seen this layer before at a higher depth,
    # we should use that depth instead of the node depth.
    # This is necessary for shared layers that have inputs at different
    # depth levels in the graph.
    depth = max(depth, previous_depth)
    layers_depths[node.layer] = depth
    nodes_depths[node] = depth

    # Update the depth of inbound nodes.
    # The "depth" of a node is the max of the depths
    # of all nodes it is connected to + 1.
    for node_dep in node.parent_nodes:
      previous_depth = nodes_depths.get(node_dep, 0)
      nodes_depths[node_dep] = max(depth + 1, previous_depth)

  # Handle inputs that are not connected to outputs.
  # We do not error out here because the inputs may be used to compute losses
  # and metrics.
  for input_t in inputs:
    input_layer = input_t._keras_history[0]
    if input_layer not in layers_depths:
      layers_depths[input_layer] = 0
      layer_indices[input_layer] = -1
      nodes_depths[input_layer._inbound_nodes[0]] = 0
      network_nodes.add(_make_node_key(input_layer.name, 0))

  # Build a dict {depth: list of nodes with this depth}
  nodes_by_depth = collections.defaultdict(list)
  for node, depth in nodes_depths.items():
    nodes_by_depth[depth].append(node)

  # Build a dict {depth: list of layers with this depth}
  layers_by_depth = collections.defaultdict(list)
  for layer, depth in layers_depths.items():
    layers_by_depth[depth].append(layer)

  # Get sorted list of layer depths.
  depth_keys = list(layers_by_depth.keys())
  depth_keys.sort(reverse=True)

  # Set self.layers ordered by depth.
  layers = []
  for depth in depth_keys:
    layers_for_depth = layers_by_depth[depth]
    # Network.layers needs to have a deterministic order:
    # here we order them by traversal order.
    layers_for_depth.sort(key=lambda x: layer_indices[x])
    layers.extend(layers_for_depth)

  # Get sorted list of node depths.
  depth_keys = list(nodes_by_depth.keys())
  depth_keys.sort(reverse=True)

  # Check that all tensors required are computable.
  # computable_tensors: all tensors in the graph
  # that can be computed from the inputs provided.
  computable_tensors = set()
  for x in inputs:
    computable_tensors.add(id(x))

  layers_with_complete_input = []  # To provide a better error msg.
  for depth in depth_keys:
    for node in nodes_by_depth[depth]:
      layer = node.layer
      if layer and not node.is_input:
        for x in nest.flatten(node.keras_inputs):
          if id(x) not in computable_tensors:
            raise ValueError('Graph disconnected: '
                             'cannot obtain value for tensor ' + str(x) +
                             ' at layer "' + layer.name + '". '
                             'The following previous layers '
                             'were accessed without issue: ' +
                             str(layers_with_complete_input))
        for x in nest.flatten(node.outputs):
          computable_tensors.add(id(x))
        layers_with_complete_input.append(layer.name)

  # Ensure name unicity, which will be crucial for serialization
  # (since serialized nodes refer to layers by their name).
  all_names = [layer.name for layer in layers]
  for name in all_names:
    if all_names.count(name) != 1:
      raise ValueError('The name "' + name + '" is used ' +
                       str(all_names.count(name)) + ' times in the model. '
                       'All layer names should be unique.')
  return network_nodes, nodes_by_depth, layers, layers_by_depth


def _build_map(outputs):
  """This method topologically sorts nodes in order from inputs to outputs.

  It uses a depth-first search to topologically sort nodes that appear in the
  _keras_history connectivity metadata of `outputs`.

  Args:
    outputs: the output tensors whose _keras_history metadata should be walked.
    This may be an arbitrary nested structure.

  Returns:
    A tuple like (ordered_nodes, layer_to_first_traversal_index)
    ordered_nodes: list of nodes appearing in the keras history, topologically
      sorted from original inputs to the `outputs`.
      (If outputs have different sets of ancestors, the inputs to one output
      may appear after a different output).
    layer_to_first_traversal_index:
      A dict mapping layer to the traversal index in the DFS where it is
      seen. Note: if a layer is shared by several nodes, the dict will only
      store the index corresponding to the *first* time the layer seen.
  """
  finished_nodes = set()
  nodes_in_progress = set()
  nodes_in_decreasing_depth = []  # nodes from inputs -> outputs.
  layer_indices = {}  # layer -> in traversal order.
  for output in nest.flatten(outputs):
    _build_map_helper(output, finished_nodes, nodes_in_progress,
                      nodes_in_decreasing_depth, layer_indices)
  return nodes_in_decreasing_depth, layer_indices


def _build_map_helper(tensor, finished_nodes, nodes_in_progress,
                      nodes_in_decreasing_depth, layer_indices):
  """Recursive helper for `_build_map`."""
  layer, node_index, _ = tensor._keras_history  # pylint: disable=protected-access
  node = layer._inbound_nodes[node_index]  # pylint: disable=protected-access

  # Don't repeat work for shared subgraphs
  if node in finished_nodes:
    return

  # Prevent cycles.
  if node in nodes_in_progress:
    raise ValueError('The tensor ' + str(tensor) + ' at layer "' + layer.name +
                     '" is part of a cycle.')

  # Store the traversal order for layer sorting.
  if layer not in layer_indices:
    layer_indices[layer] = len(layer_indices)

  # Propagate to all previous tensors connected to this node.
  nodes_in_progress.add(node)
  if not node.is_input:
    for tensor in node.keras_inputs:
      _build_map_helper(tensor, finished_nodes, nodes_in_progress,
                        nodes_in_decreasing_depth, layer_indices)

  finished_nodes.add(node)
  nodes_in_progress.remove(node)
  nodes_in_decreasing_depth.append(node)


def _map_subgraph_network(inputs, outputs):
  """Returns the nodes and layers in the topology from `inputs` to `outputs`.

  Args:
    inputs: List of input tensors.
    outputs: List of output tensors.

  Returns:
    A tuple of List{Node] and List[Layer].
  """
  base_layer_utils.create_keras_history(outputs)
  # Keep only nodes and layers in the topology between inputs and outputs.
  _, nodes_by_depth, layers, _ = _map_graph_network(inputs, outputs)
  return nest.flatten([nodes for nodes in nodes_by_depth.values()]), layers


def _should_skip_first_node(layer):
  """Returns True if the first layer node should not be saved or loaded."""
  # Networks start with a pre-existing node linking their input to output.
  return issubclass(layer.__class__, Network) and layer._is_graph_network


def _deserialize_keras_tensors(kwargs, layer_map):
  """Deserializes Keras Tensors passed to `call`.."""

  def _deserialize_keras_tensor(t):
    """Deserializes a single Keras Tensor passed to `call`."""
    if isinstance(t, tf_utils.ListWrapper):
      t = t.as_list()
      layer_name = t[0]
      node_index = t[1]
      tensor_index = t[2]

      layer = layer_map[layer_name]
      node = layer._inbound_nodes[node_index]
      return nest.flatten(node.outputs)[tensor_index]
    return t

  kwargs = tf_utils.convert_inner_node_data(kwargs, wrap=True)
  return nest.map_structure(_deserialize_keras_tensor, kwargs)


def connect_ancillary_layers(model, created_layers):
  """Adds layers that are not connected to the outputs to the model."""
  # Layers not connected to outputs, such as those added in `add_loss`.
  ancillary_layers = [
      layer for layer in created_layers.values() if layer not in model.layers
  ]
  if ancillary_layers:
    relevant_nodes = nest.flatten([
        layer.inbound_nodes[1:]
        if _should_skip_first_node(layer) else layer.inbound_nodes
        for layer in created_layers.values()
    ])
    model._insert_layers(ancillary_layers, relevant_nodes)
  return model


def reconstruct_from_config(config, custom_objects=None, created_layers=None):
  """Reconstructs graph from config object.

  Args:
    config: Dictionary returned from Network.get_config()
    custom_objects: Optional dictionary mapping names (strings) to custom
      classes or functions to be considered during deserialization.
    created_layers: Optional dictionary mapping names to Layer objects. Any
      layer not in this dictionary will be be created and added to the dict.
      This function will add new nodes to all layers (excluding InputLayers),
      instead of re-using pre-existing nodes in the layers.

  Returns:
    Tuple of (input tensors, output tensors, dictionary of created layers)
  """
  # Layer instances created during the graph reconstruction process.
  created_layers = created_layers or collections.OrderedDict()

  # Maps input data (tuple of inbound layer name, node index) from the config
  # to node indices in the newly generated model. The node indices may be
  # different if the layers have already been called previously.
  node_index_map = {}
  node_count_by_layer = {}

  # Dictionary mapping layer instances to
  # node data that specifies a layer call.
  # It acts as a queue that maintains any unprocessed
  # layer call until it becomes possible to process it
  # (i.e. until the input tensors to the call all exist).
  unprocessed_nodes = {}

  def add_unprocessed_node(layer, node_data):
    if layer not in unprocessed_nodes:
      unprocessed_nodes[layer] = [node_data]
    else:
      unprocessed_nodes[layer].append(node_data)

  def get_node_index(layer, config_node_index):
    """Returns node index in layer (might differ from config_node_index)."""
    if isinstance(layer, input_layer_module.InputLayer):
      return 0
    return node_index_map.get((layer.name, config_node_index), None)

  def process_node(layer, node_data):
    """Deserialize a node.

    Arguments:
        layer: layer instance.
        node_data: Nested structure of `ListWrapper`.

    Raises:
        ValueError: In case of improperly formatted `node_data`.
    """
    input_tensors = []
    for input_data in nest.flatten(node_data):
      input_data = input_data.as_list()
      inbound_layer_name = input_data[0]
      inbound_node_index = input_data[1]
      inbound_tensor_index = input_data[2]
      if len(input_data) == 3:
        kwargs = {}
      elif len(input_data) == 4:
        kwargs = input_data[3]
        kwargs = _deserialize_keras_tensors(kwargs, created_layers)
      else:
        raise ValueError('Improperly formatted model config.')

      inbound_layer = created_layers[inbound_layer_name]
      inbound_node_index = get_node_index(inbound_layer, inbound_node_index)

      if inbound_node_index is None:
        add_unprocessed_node(layer, node_data)
        return
      inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
      input_tensors.append(
          nest.flatten(inbound_node.outputs)[inbound_tensor_index])
    input_tensors = nest.pack_sequence_as(node_data, input_tensors)
    # Call layer on its inputs, thus creating the node
    # and building the layer if needed.
    if input_tensors is not None:
      input_tensors = base_layer_utils.unnest_if_single_tensor(input_tensors)
      output_tensors = layer(input_tensors, **kwargs)

      # Update node index map.
      output_index = nest.flatten(output_tensors)[0]._keras_history.node_index
      node_index_map[(layer.name, node_count_by_layer[layer])] = output_index
      node_count_by_layer[layer] += 1

  def process_layer(layer_data):
    """Deserializes a layer, then call it on appropriate inputs.

    Arguments:
        layer_data: layer config dict.

    Raises:
        ValueError: In case of improperly formatted `layer_data` dict.
    """
    layer_name = layer_data['name']

    if layer_name in created_layers:
      layer = created_layers[layer_name]
    else:
      # Instantiate layer.
      from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top

      layer = deserialize_layer(layer_data, custom_objects=custom_objects)
      created_layers[layer_name] = layer

    node_count_by_layer[layer] = int(_should_skip_first_node(layer))

    # Gather layer inputs and convert to `ListWrapper` objects.
    inbound_nodes_data = layer_data['inbound_nodes']
    inbound_nodes_data = tf_utils.convert_inner_node_data(
        inbound_nodes_data, wrap=True)
    for node_data in inbound_nodes_data:
      # We don't process nodes (i.e. make layer calls)
      # on the fly because the inbound node may not yet exist,
      # in case of layer shared at different topological depths
      # (e.g. a model such as A(B(A(B(x)))))
      add_unprocessed_node(layer, node_data)

  # First, we create all layers and enqueue nodes to be processed
  for layer_data in config['layers']:
    process_layer(layer_data)
  # Then we process nodes in order of layer depth.
  # Nodes that cannot yet be processed (if the inbound node
  # does not yet exist) are re-enqueued, and the process
  # is repeated until all nodes are processed.
  while unprocessed_nodes:
    for layer_data in config['layers']:
      layer = created_layers[layer_data['name']]
      if layer in unprocessed_nodes:
        for node_data in unprocessed_nodes.pop(layer):
          process_node(layer, node_data)

  input_tensors = []
  output_tensors = []

  input_layers = tf_utils.convert_inner_node_data(
      config['input_layers'], wrap=True)
  for layer_data in nest.flatten(input_layers):
    layer_name, node_index, tensor_index = layer_data.as_list()
    assert layer_name in created_layers
    layer = created_layers[layer_name]
    node_index = get_node_index(layer, node_index)
    layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
    input_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])

  output_layers = tf_utils.convert_inner_node_data(
      config['output_layers'], wrap=True)
  for layer_data in nest.flatten(output_layers):
    layer_name, node_index, tensor_index = layer_data.as_list()
    assert layer_name in created_layers
    layer = created_layers[layer_name]
    node_index = get_node_index(layer, node_index)
    layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
    output_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])

  input_tensors = nest.pack_sequence_as(input_layers, input_tensors)
  output_tensors = nest.pack_sequence_as(output_layers, output_tensors)
  return input_tensors, output_tensors, created_layers


def get_network_config(network, serialize_layer_fn=None):
  """Builds the config, which consists of the node graph and serialized layers.

  Args:
    network: A Network object.
    serialize_layer_fn: Function used to serialize layers.

  Returns:
    Config dictionary.
  """
  serialize_layer_fn = (
      serialize_layer_fn or generic_utils.serialize_keras_object)
  config = {
      'name': network.name,
  }
  node_conversion_map = {}
  for layer in network.layers:
    kept_nodes = 1 if _should_skip_first_node(layer) else 0
    for original_node_index, node in enumerate(layer._inbound_nodes):
      node_key = _make_node_key(layer.name, original_node_index)
      if node_key in network._network_nodes:
        node_conversion_map[node_key] = kept_nodes
        kept_nodes += 1
  layer_configs = []
  for layer in network.layers:  # From the earliest layers on.
    filtered_inbound_nodes = []
    for original_node_index, node in enumerate(layer._inbound_nodes):
      node_key = _make_node_key(layer.name, original_node_index)
      if node_key in network._network_nodes and not node.is_input:
        # The node is relevant to the model:
        # add to filtered_inbound_nodes.
        node_data = node.serialize(_make_node_key, node_conversion_map)
        filtered_inbound_nodes.append(node_data)

    layer_config = serialize_layer_fn(layer)
    layer_config['name'] = layer.name
    layer_config['inbound_nodes'] = filtered_inbound_nodes
    layer_configs.append(layer_config)
  config['layers'] = layer_configs

  # Gather info about inputs and outputs.
  model_inputs = []
  for i in range(len(network._input_layers)):
    layer, node_index, tensor_index = network._input_coordinates[i]
    node_key = _make_node_key(layer.name, node_index)
    if node_key not in network._network_nodes:
      continue
    new_node_index = node_conversion_map[node_key]
    model_inputs.append(
        tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
  model_inputs = nest.pack_sequence_as(network._nested_inputs, model_inputs)
  # Preserve external Keras compat for Models with single input.
  if not nest.is_sequence(model_inputs):
    model_inputs = [model_inputs]
  model_inputs = tf_utils.convert_inner_node_data(model_inputs)
  config['input_layers'] = model_inputs

  model_outputs = []
  for i in range(len(network._output_layers)):
    layer, node_index, tensor_index = network._output_coordinates[i]
    node_key = _make_node_key(layer.name, node_index)
    if node_key not in network._network_nodes:
      continue
    new_node_index = node_conversion_map[node_key]
    model_outputs.append(
        tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
  model_outputs = nest.pack_sequence_as(network._nested_outputs, model_outputs)
  # Preserve external Keras compat for Models with single output.
  if not nest.is_sequence(model_outputs):
    model_outputs = [model_outputs]
  model_outputs = tf_utils.convert_inner_node_data(model_outputs)
  config['output_layers'] = model_outputs
  return config
