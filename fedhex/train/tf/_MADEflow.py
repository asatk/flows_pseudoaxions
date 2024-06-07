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
import os
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.bijectors import \
    MaskedAutoregressiveFlow as MAF
from tensorflow_probability.python.distributions import TransformedDistribution

from ..._loaders import DataManager
from ...io._path import LOG_FATAL, LOG_WARN, print_msg


@tfk.saving.register_keras_serializable(package="fedhex.train.tf", name="loss_MADE")
def loss_MADE(x, logprob):
    return -logprob


@tfk.saving.register_keras_serializable(name="Made")
class MADE(tfb.AutoregressiveNetwork):
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
    
    # @classmethod
    # def from_config(cls, config):
    #     sublayer_config = config.pop("")
    #     sublayer = tfk.saving.deserialize_keras_object(sublayer_config)
    #     return cls(sublayer, **config)


def build_MADE(num_inputs: int,
               num_made: int,
               hidden_units: list[int],
               activation: str="relu",
               cond_event_shape: tuple=None)-> tuple[
                   TransformedDistribution,
                   list]:
    """
    Construct a flow.
    """

    if cond_event_shape is None:
        cond_event_shape = (num_inputs, )
    
    # construct a list of all flows and permute input dimensions btwn each
    maf_list = []
    for i in range(num_made):
        # 2 params indicates learning one param for the scale and the shift
        made = MADE(params=2, hidden_units=hidden_units,
            event_shape=(num_inputs,), conditional=True,
            conditional_event_shape=cond_event_shape,
            activation=activation, name=f"made_{i}")
        maf = MAF(shift_and_log_scale_fn=made, name=f"maf_{i}")
        perm = tfb.Permute(np.arange(0, num_inputs)[::-1])
        
        maf_list.append(maf)
        # there is no permute on the output layer
        if i < num_made - 1:
            maf_list.append(perm)

    # chain the flows together to complete bijection
    chain = tfb.Chain(list(reversed(maf_list)))

    # transform a distribution of joint std normals to our target distribution
    distribution = TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[num_inputs]),
        bijector=chain)
    
    return distribution, maf_list


def compile_MADE(num_made: int,
                 num_inputs: int,
                 hidden_units: list[int],
                 num_cond_inputs: int=None,
                 activation: str="relu",
                 loss: Callable=loss_MADE,
                 opt: tfk.optimizers.Optimizer=None) -> tuple[
                     tfk.Model,
                     TransformedDistribution,
                     list[Any]]:
    """
    Build new model from scratch
    """

    # Define optimizer/learning rate function
    if opt is None or not isinstance(opt, tfk.optimizers.Optimizer):
        print_msg("Using default optimizer and learning rate schedule: Adam w/ learning rate=1e-3")
        # learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(
        #         initial_learning_rate=1e-3, decay_steps=200,
        #         end_learning_rate=1e-4, power=1.0, cycle=False)
        opt = tfk.optimizers.Adam()

    # Build model layers and compile
    distribution, maf_list = build_MADE(num_inputs,
            hidden_units=hidden_units, cond_event_shape=(num_cond_inputs, ),
            num_made=num_made, activation=activation)
    
    # Data and Conditional Data input layers
    x_ = tfk.layers.Input(shape=(num_inputs,), name="aux_input")
    c_ = tfk.layers.Input(shape=(num_cond_inputs,), name="cond_input")
    input_list = [x_, c_]

    # Feed conditional data to MAFs
    current_kwargs = {}
    for i in range(num_made):
        current_kwargs[f"maf_{i}"] = {"conditional_input" : c_}
    
    # Log-likelihood output of distribution is network output (loss)
    log_prob_ = distribution.log_prob(x_, bijector_kwargs=current_kwargs)
    model = tfk.Model(input_list, log_prob_)
    model.compile(optimizer=opt,
                  loss=loss)

    return model, distribution, maf_list

def maskOutside(arr: np.ndarray, ranges: list[list[float]]):
    mask = np.zeros(shape=arr.shape[0], dtype=bool)
    for i in range(len(ranges)):
        mask |= (arr[:,i] < ranges[i][0]) | (arr[:,i] > ranges[i][1])
    return np.array(mask)

def eval_MADE(cond,
              made_list: list,
              dist: TransformedDistribution,
              dm: DataManager=None,
              criteria: Callable=None,
              ranges: list[list[float]]=None,
              seed: int = 0x2024,
              *args) -> np.ndarray:
    
    if dm is not None:
        cond = dm.norm(samples=cond, is_cond=True)

    # Pass the flow the conditional inputs (labels)
    current_kwargs = {}
    for i in range(len(made_list) // 2):
        current_kwargs[f"maf_{i}"] = {"conditional_input" : cond}
    
    # Evaluate the flow at each of the provided conditional inputs
    gen_data_norm = np.array(dist.sample(len(cond), bijector_kwargs=current_kwargs, seed=seed))
    # Use DM to transform data into problem space where criteria are defined
    if dm is not None:
        gen_data = dm.denorm(gen_data_norm, is_cond=False)
   
    #If there is no rejection criterion, return the generated data
    if criteria is None and ranges is None:
        return gen_data
    
    # Handle `None` as +/- inifity when `ranges` is used
    if ranges is not None:
        criteria = maskOutside
        for r in ranges:
            if r[0] is None:
                r[0] = -np.inf
            if r[1] is None:
                r[1] = np.inf
        args = (ranges,)
    
    # Create a mask based on rejection criteria applied to generated data
    mask = criteria(gen_data, *args)

    # Re-sample until no elements are rejected
    while(np.any(mask)):
        
        # Pass the flow the new conditional inputs (labels)
        current_kwargs = {}
        for i in range(len(made_list) // 2):
            current_kwargs[f"maf_{i}"] = {"conditional_input" : cond[mask]}
        
        # Re-sample
        gen_resample_norm = np.array(dist.sample(np.sum(mask), bijector_kwargs=current_kwargs, seed=seed))
        if dm is not None:
            gen_data[mask] = dm.denorm(gen_resample_norm, is_cond=False)
        else:
            gen_data[mask] = gen_resample_norm
        
        # Update mask based on rejection criteria applied to new data
        mask = criteria(gen_data, *args)
    
    return gen_data


def intermediate_MADE(made_list):

    """
    Separate each step of the flow into individual distributions in order to
    samples from and test each bijection's output.
    """

    # TODO check that model is being built correctly, esp when each MADE has
    # more than one layer

    # reverse the list of made blocks to unpack in generating direction
    made_list_rev = list(reversed(made_list[:-1]))

    feat_extraction_dists = []

    made_chain = tfb.Chain([])
    dist = TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[2]), # TODO shape is not 2 in general
        bijector=made_chain)
    feat_extraction_dists.append(dist)

    for i in range(1, len(made_list_rev), 2):
        made_chain = tfb.Chain(made_list_rev[0:i])
        dist = TransformedDistribution(
            distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[2]),
            bijector=made_chain)
        feat_extraction_dists.append(dist)

    return feat_extraction_dists


def load_MADE(flow_path: str|None=None)-> tuple[
                  tfk.Model,
                  TransformedDistribution,
                  list[Any],
                  dict[str, Any]]:
    """
    Retrieve the Keras SavedModel, the tfb TransformedDistribution, and the
    list of MADE blocks from the desired model. Either a new model and its
    parts or those contained in the given model directory are returned.

    flow_path, str or None
        Path pointing to a Keras SavedModel instance on disk. If None then
        a new model is constructed. Otherwise, a pre-built model is loaded from
        the file at `flow_path`.

    Returns
        keras.SavedModel : model with the specified parameters
        TFDistribution.TransformedDistribution : transformation from the
            normalizing distribution to the data (what is trained and sampled)
        list[tf.Module] : list of the MAF layers with Permute layers in between
    """

    if not os.path.isdir(flow_path):
        print_msg(f"The model at '{flow_path}' does not exist.", level=LOG_WARN)
        return tuple(None, None, None, None)
    
    # Define model's custom objects
    # custom_objects = {"loss_MADE": loss_MADE, "Made": MADE}
    # Load a model and extract its skeleton of MAFs
    # model = tfk.models.load_model(flow_path, custom_objects=custom_objects)

    model: tfk.Model= tfk.models.load_model(flow_path)
    layer_names = [layer.name for layer in model.submodules if isinstance(layer, MADE)]

    num_made = len(layer_names)
    ninputs = model.get_layer("aux_input").input_shape[0][-1]
    ncinputs = model.get_layer("cond_input").input_shape[0][-1]
    
    made0: MADE= model.get_layer(layer_names[0])
    activation = made0.activation
    hidden_units = list(made0.hidden_units)
    
    made_list = []
    maf_list = []

    for i in range(num_made):
        made = model.get_layer(name=f"made_{i}")
        maf = MAF(shift_and_log_scale_fn=made, name=f"maf_{i}")
        perm = tfb.Permute(np.arange(0, ninputs)[::-1])
        
        made_list.append(made)
        maf_list.append(maf)
        # there is no permute on the output layer
        if i < num_made - 1:
            maf_list.append(perm)

    # chain the flows together to complete bijection
    chain = tfb.Chain(list(reversed(maf_list)))

    # transform a distribution of joint std normals to our target distribution
    distribution = TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[ninputs]),
        bijector=chain)
    
    cfg = dict(nmade=num_made,
               ninputs=ninputs,
               ncinputs=ncinputs,
               hidden_units=hidden_units,
               activation=activation)
    
    return model, distribution, made_list, cfg