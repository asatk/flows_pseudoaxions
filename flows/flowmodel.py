"""
Author: Anthony Atkinson
Modified: 2023.07.14

Contains the core flow model necessities. Building a flow, compiling one from
scratch, the loss function, retrieving the intermediate flows, and the MADE
block itself.

It will likely contain other types of blocks/flows, in which case PyTorch is
more favorable due to its exposed interface and customizability.
"""

from typing import Any
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.distributions import TransformedDistribution
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.bijectors import MaskedAutoregressiveFlow as MAF
import numpy as np

class Made(tfb.AutoregressiveNetwork):
    """
    A duplicate of tfp.bijectors.AutoregressiveNetwork class with tanh applied
    on the output log-scape. This is important for improved regularization and
    "inf" and "nan" values that would otherwise often occur during training.
    """

    def __init__(self, params=None, event_shape=None, conditional=True,
                 conditional_event_shape=None,
                 conditional_input_layers="all_layers", hidden_units=None,
                 input_order="left-to-right", hidden_degrees="equal",
                 activation=None, use_bias=True,
                 kernel_initializer="glorot_uniform", bias_initializer="zeros",
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 validate_args=False, **kwargs):
        
        super().__init__(
            params=params, event_shape=event_shape, conditional=conditional,
            conditional_event_shape=conditional_event_shape,
            conditional_input_layers=conditional_input_layers,
            hidden_units=hidden_units, input_order=input_order,
            hidden_degrees=hidden_degrees, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, validate_args=validate_args,
            **kwargs)
        
        self.conditional = conditional
        self.conditional_event_shape = conditional_event_shape
        self.hidden_units = hidden_units
        self.activation = activation
    
    def call(self, x, conditional_input=None):
        
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

def build_distribution(made_list: list, num_inputs: int, num_made: int=10,
        hidden_layers: int=1, hidden_units: int=128,
        cond_event_shape: tuple=None)-> tuple[TransformedDistribution, list]:

    if cond_event_shape is None:
        cond_event_shape = (num_inputs, )
    
    permutation = tfb.Permute(np.arange(0, num_inputs)[::-1])
    hidden_units_list = hidden_layers * [hidden_units]
    if len(made_list) == 0:
        for i in range(num_made):
            made_list.append(MAF(shift_and_log_scale_fn=Made(
                            params=2, hidden_units=hidden_units_list,
                            event_shape=(num_inputs,), conditional=True,
                            conditional_event_shape=cond_event_shape,
                            activation="relu", name=f"made_{i}"),
                            name=f"maf_{i}"))
            made_list.append(permutation)
    else:
        made_list_temp = []
        for i, made in enumerate(made_list):
            made_list_temp.append(MAF(shift_and_log_scale_fn=made, name=f"maf_{i}"))
            made_list_temp.append(permutation)

        made_list = made_list_temp

    # make bijection from made layers; remove final permute layer
    made_chain = tfb.Chain(list(reversed(made_list[:-1])))

    # we want to transform to gaussian distribution with mean 0 and std 1 in latent space
    distribution = TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[num_inputs]),
        bijector=made_chain)
    
    return distribution, made_list

def compile_MAF_model(num_made: int, num_inputs: int,
        num_cond_inputs: int=None, hidden_layers: int=1, hidden_units: int=128,
        lr_tuple: tuple=(1e-3, 1e-4, 100)) -> tuple[tfk.Model, Any, list[Any]]:
    # TODO add docstring though this is mostly an internal method - do later
    """
    Build new model from scratch
    """

    # Define optimizer/learning rate function
    base_lr, end_lr, decay_steps = lr_tuple
    learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=base_lr, decay_steps=decay_steps,
            end_learning_rate=end_lr, power=0.5, cycle=True)
    
    # Build model layers and compile
    made_list = []
    distribution, made_list = build_distribution(made_list, num_inputs,
            hidden_layers=hidden_layers, hidden_units=hidden_units,
            cond_event_shape=(num_cond_inputs, ), num_made=num_made)
    
    # Data input layers
    x_ = tfk.layers.Input(shape=(num_inputs,), name="aux_input")
    input_list = [x_]

    # Conditional data input laters
    c_ = tfk.layers.Input(shape=(num_cond_inputs,), name="cond_input")
    input_list.append(c_)

    # Feed conditional data to MAFs
    current_kwargs = {}
    for i in range(num_made):
        current_kwargs[f"maf_{i}"] = {"conditional_input" : c_}
    
    # Log-likelihood output of distribution is network output (loss)
    log_prob_ = distribution.log_prob(x_, bijector_kwargs=current_kwargs)
    model = tfk.Model(input_list, log_prob_)
    model.compile(
            optimizer=tfk.optimizers.Adam(learning_rate=learning_rate_fn),
            loss=lossfn)

    return model, distribution, made_list

def lossfn(x, logprob):
    return -logprob

def intermediate_flows_chain(made_list):
    """
    Separate each step of the flow into individual distributions in order to
    samples from and test each bijection's output.
    """

    # reverse the list of made blocks to unpack in generating direction
    made_list_rev = list(reversed(made_list[:-1]))

    feat_extraction_dists = []

    made_chain = tfb.Chain([])
    dist = TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[2]),
        bijector=made_chain)
    feat_extraction_dists.append(dist)

    for i in range(1, len(made_list_rev), 2):
        made_chain = tfb.Chain(made_list_rev[0:i])
        dist = TransformedDistribution(
            distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[2]),
            bijector=made_chain)
        feat_extraction_dists.append(dist)

    return feat_extraction_dists