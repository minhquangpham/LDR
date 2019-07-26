import tensorflow as tf

from subprocess import Popen

import numpy as np
import time

from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn import *

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

home_dir = "/home/pham/Neural_Align/"

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class ConditionalGRUCell(LayerRNNCell):
  """Conditional Gated Recurrent Unit cell (cf. https://www.aclweb.org/anthology/E17-3017).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().
  """
  def __init__(self,
               num_units,
               input_depth=512,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None,
               dtype=None,
               **kwargs):
    super(ConditionalGRUCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = num_units
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh
    self._kernel_initializer = initializers.get(kernel_initializer)
    self._bias_initializer = initializers.get(bias_initializer)
    self.input_depth = input_depth
  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    #input_depth = inputs_shape[-1]
    input_depth = self.input_depth
    # GRU cell 1 
    self._gate_kernel_1 = self.add_variable(
        "gates_1/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias_1 = self.add_variable(
        "gates_1/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel_1 = self.add_variable(
        "candidate_1/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias_1 = self.add_variable(
        "candidate_1/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))
    # GRU cell 2
    self._gate_kernel_2 = self.add_variable(
        "gates_2/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[2 * self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias_2 = self.add_variable(
        "gates_2/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel_2 = self.add_variable(
        "candidate_2/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[2 * self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias_2 = self.add_variable(
        "candidate_2/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))
    
    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    # Using Attention Wrapper
    print("self.input_depth:",self.input_depth)
    inputs, context = array_ops.split(value=inputs, num_or_size_splits=[int(self.input_depth), self._num_units], axis=-1)
    #_, _, state = array_ops.split(value=state, num_or_size_splits=[int(self.input_depth), self._num_units, self._num_units], axis=-1)
    
    # First round
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel_1)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias_1)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = math_ops.matmul(
        array_ops.concat([inputs, r_state], 1), self._candidate_kernel_1)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias_1)

    c = self._activation(candidate)
    new_h_prime = u * state + (1 - u) * c
    
    # Final round

    gate_inputs = math_ops.matmul(
        array_ops.concat([context, new_h_prime], 1), self._gate_kernel_2)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias_2)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * new_h_prime

    candidate = math_ops.matmul(
        array_ops.concat([context, r_state], 1), self._candidate_kernel_2)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias_2)

    c = self._activation(candidate)
    new_h = u * new_h_prime + (1 - u) * c

    return new_h, new_h

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "kernel_initializer": initializers.serialize(self._kernel_initializer),
        "bias_initializer": initializers.serialize(self._bias_initializer),
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(ConditionalGRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))



