"""
Author: Anthony Atkinson and Darshan Lakshminarayanan
Modified: 2024.03.05

Contains the core flow model necessities. Building a flow, compiling one from
scratch, the loss function, retrieving the intermediate flows, and the MADE
block itself.

It will likely contain other types of blocks/flows, in which case PyTorch is
more favorable due to its exposed interface and customizability.
"""

import json
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import TransformedDistribution
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import RealNVP
from tensorflow_probability.python.bijectors import AutoregressiveNetwork
from typing import Any, Callable

from ..io._path import LOG_WARN, LOG_FATAL, print_msg
from .._loaders import DataManager
from ._loss import NLL

class MLPRealNVP(AutoregressiveNetwork):

    def __init__(self,
                 params,
                 event_shape=None,
                 conditional=False,
                 conditional_event_shape=None,
                 conditional_input_layers='all_layers',
                 hidden_units=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 validate_args=False,
                 **kwargs):
        """Constructs the RealNVP layer. Duplicate of AutoregressiveNetwork
        from Tensorflow Probability with the masks removed.

        Args:
        params: Python integer specifying the number of parameters to output
            per input.
        event_shape: Python `list`-like of positive integers (or a single int),
            specifying the shape of the input to this layer, which is also the
            event_shape of the distribution parameterized by this layer.  Currently
            only rank-1 shapes are supported.  That is, event_shape must be a single
            integer.  If not specified, the event shape is inferred when this layer
            is first called or built.
        conditional: Python boolean describing whether to add conditional inputs.
        conditional_event_shape: Python `list`-like of positive integers (or a
            single int), specifying the shape of the conditional input to this layer
            (without the batch dimensions). This must be specified if `conditional`
            is `True`.
        conditional_input_layers: Python `str` describing how to add conditional
            parameters to the autoregressive network. When "all_layers" the
            conditional input will be combined with the network at every layer,
            whilst "first_layer" combines the conditional input only at the first
            layer which is then passed through the network
            autoregressively. Default: 'all_layers'.
        hidden_units: Python `list`-like of non-negative integers, specifying
            the number of units in each hidden layer.
        hidden_degrees: Method for assigning degrees to the hidden units:
            'equal', 'random'.  If 'equal', hidden units in each layer are allocated
            equally (up to a remainder term) to each degree.  Default: 'equal'.
        activation: An activation function.  See `tf.keras.layers.Dense`. Default:
            `None`.
        use_bias: Whether or not the dense layers constructed in this layer
            should have a bias term.  See `tf.keras.layers.Dense`.  Default: `True`.
        kernel_initializer: Initializer for the `Dense` kernel weight
            matrices.  Default: 'glorot_uniform'.
        bias_initializer: Initializer for the `Dense` bias vectors. Default:
            'zeros'.
        kernel_regularizer: Regularizer function applied to the `Dense` kernel
            weight matrices.  Default: None.
        bias_regularizer: Regularizer function applied to the `Dense` bias
            weight vectors.  Default: None.
        kernel_constraint: Constraint function applied to the `Dense` kernel
            weight matrices.  Default: None.
        bias_constraint: Constraint function applied to the `Dense` bias
            weight vectors.  Default: None.
        validate_args: Python `bool`, default `False`. When `True`, layer
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
        **kwargs: Additional keyword arguments passed to this layer (but not to
            the `tf.keras.layer.Dense` layers constructed by this layer).
        """
        super().__init__(**kwargs)

        self._params = params
        self._event_shape = _list(event_shape) if event_shape is not None else None
        self._conditional = conditional
        self._conditional_event_shape = (
            _list(conditional_event_shape)
            if conditional_event_shape is not None else None)
        self._conditional_layers = conditional_input_layers
        self._hidden_units = hidden_units if hidden_units is not None else []
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = bias_constraint
        self._validate_args = validate_args
        self._kwargs = kwargs

        if self._event_shape is not None:
            self._event_size = self._event_shape[-1]
            self._event_ndims = len(self._event_shape)

            if self._event_ndims != 1:
                raise ValueError('Parameter `event_shape` must describe a rank-1 '
                                'shape. `event_shape: {!r}`'.format(event_shape))

        if self._conditional:
            if self._event_shape is None:
                raise ValueError('`event_shape` must be provided when '
                                '`conditional` is True')
            if self._conditional_event_shape is None:
                raise ValueError('`conditional_event_shape` must be provided when '
                                '`conditional` is True')
            self._conditional_size = self._conditional_event_shape[-1]
            self._conditional_ndims = len(self._conditional_event_shape)
            if self._conditional_ndims != 1:
                raise ValueError('Parameter `conditional_event_shape` must describe a '
                                'rank-1 shape')
            if not ((self._conditional_layers == 'first_layer') or
                    (self._conditional_layers == 'all_layers')):
                raise ValueError('`conditional_input_layers` must be '
                                '"first_layers" or "all_layers"')
        else:
            if self._conditional_event_shape is not None:
                raise ValueError('`conditional_event_shape` passed but `conditional` '
                                'is set to False.')

        # To be built in `build`.
        self._network = None
    

    def build(self, input_shape):
        """See tfkl.Layer.build."""
        if self._event_shape is None:
            # `event_shape` wasn't specied at __init__, so infer from `input_shape`.
            self._event_shape = [tf.compat.dimension_value(input_shape[-1])]
            self._event_size = self._event_shape[-1]
            self._event_ndims = len(self._event_shape)
            # Should we throw if input_shape has rank > 2?

        if input_shape[-1] != self._event_shape[-1]:
            raise ValueError('Invalid final dimension of `input_shape`. '
                            'Expected `{!r}`, but got `{!r}`'.format(
                                self._event_shape[-1], input_shape[-1]))

        outputs = [tf.keras.Input((self._event_size,), dtype=self.dtype)]
        inputs = outputs[0]
        if self._conditional:
            conditional_input = tf.keras.Input((self._conditional_size,),
                                            dtype=self.dtype)
            inputs = [inputs, conditional_input]

        # Input-to-hidden, hidden-to-hidden, and hidden-to-output layers:
        #  [..., self._event_size] -> [..., self._hidden_units[0]].
        #  [..., self._hidden_units[k-1]] -> [..., self._hidden_units[k]].
        #  [..., self._hidden_units[-1]] -> [..., event_size * self._params].
        layer_output_sizes = self._hidden_units + [self._event_size * self._params]
        for k in range(len(self._hidden_units)):
            autoregressive_output = tf.keras.layers.Dense(
                layer_output_sizes[k],
                activation=None,
                use_bias=self._use_bias,
                kernel_initializer=self._kernel_initializer,
                bias_initializer=self._bias_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                kernel_constraint=self._kernel_constraint,
                bias_constraint=self._bias_constraint,
                dtype=self.dtype)(outputs[-1])
            if (self._conditional and
                ((self._conditional_layers == 'all_layers') or
                ((self._conditional_layers == 'first_layer') and (k == 0)))):
                conditional_output = tf.keras.layers.Dense(
                    layer_output_sizes[k],
                    activation=None,
                    use_bias=False,
                    kernel_initializer=self._kernel_initializer,
                    bias_initializer=None,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=None,
                    kernel_constraint=self._kernel_constraint,
                    bias_constraint=None,
                    dtype=self.dtype)(conditional_input)
                outputs.append(tf.keras.layers.Add()([
                    autoregressive_output,
                    conditional_output]))
            else:
                outputs.append(autoregressive_output)
            if k + 1 < len(self._hidden_units):
                outputs.append(
                    tf.keras.layers.Activation(self._activation)
                    (outputs[-1]))
        self._network = tf.keras.models.Model(
            inputs=inputs,
            outputs=outputs[-1])
        # Allow network to be called with inputs of shapes that don't match
        # the specs of the network's input layers.
        self._network.input_spec = None
        # Record that the layer has been built.
        super().build(input_shape)
    

    def call(self, x, conditional_input=None):
        if self.params == 1:
            return shift
        result = super().call(x, conditional_input=conditional_input)
        shift, log_scale = tf.unstack(result, num=2, axis=-1)
        return shift, tf.math.tanh(log_scale)
    
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({"params": self.params,
                       "event_shape": self.event_shape,
                       "conditional": self.conditional,
                       "conditional_event_shape": self.conditional_event_shape,
                       "hidden_units": self.hidden_units,
                       "activation": self.activation,
                       "name": self.name})
        
        return config


def _list(xs):
    """Convert the given argument to a list."""
    try:
        return list(xs)
    except TypeError:
        return [xs]
