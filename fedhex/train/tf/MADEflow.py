"""
Author: Anthony Atkinson
Modified: 2023.07.14

Contains the core flow model necessities. Building a flow, compiling one from
scratch, the loss function, retrieving the intermediate flows, and the MADE
block itself.

It will likely contain other types of blocks/flows, in which case PyTorch is
more favorable due to its exposed interface and customizability.
"""
import json
import os
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.keras.models import load_model as kload_model, Model
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import TransformedDistribution
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import MaskedAutoregressiveFlow as MAF
from typing import Any
import numpy as np

from ...io._path import print_msg, LOG_WARN, LOG_FATAL

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

def build_MADE(made_blocks: list, num_inputs: int, num_made: int=10,
        hidden_layers: int=1, hidden_units: int=128, activation: str="relu",
        cond_event_shape: tuple=None)-> tuple[TransformedDistribution, list]:

    if cond_event_shape is None:
        cond_event_shape = (num_inputs, )
    
    permutation = tfb.Permute(np.arange(0, num_inputs)[::-1])
    
    # Since we later want to use a Gaussian as our autoregressive distribution, we need to set
    # "params" to 2, such that the MADE network can parameterize its mean and logarithmic standard deviation.

    if made_blocks is None or len(made_blocks) == 0:
        made_blocks = []
        hidden_units_list = hidden_layers * [hidden_units]
        for i in range(num_made):
            #TODO check that params should be 2 ^ refer to above from tutorial
            made = MADE(params=2, hidden_units=hidden_units_list,
                event_shape=(num_inputs,), conditional=True,
                conditional_event_shape=cond_event_shape,
                activation=activation, name=f"made_{i}")
            made_blocks.append(made)
    
    made_list = []
    for i, made in enumerate(made_blocks):
        made_list.append(MAF(shift_and_log_scale_fn=made, name=f"maf_{i}"))
        made_list.append(permutation)

    # make bijection from made layers; remove final permute layer
    made_chain = tfb.Chain(list(reversed(made_list[:-1])))

    # we want to transform to gaussian distribution with mean 0 and std 1 in latent space
    distribution = TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[num_inputs]),
        bijector=made_chain)
    
    return distribution, made_list

def compile_MADE_model(num_made: int, num_inputs: int,
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
    distribution, made_list = build_MADE(made_list, num_inputs,
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
            loss=lossfn_MADE)

    return model, distribution, made_list

def lossfn_MADE(x, logprob):
    return -logprob


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


def load_MADE(flow_path: str|None=None, newmodel: bool=True) -> tuple[Model, Any, list[Any]]:
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

    config_path = flow_path + "/config.json"
    if not os.path.isfile(config_path):
        print_msg(f"The model at '{flow_path}' is missing the model config " +
                  f"file at {config_path}. A new model is going to be " +
                  f"created at '{flow_path}'.", level=LOG_WARN)
    with open(config_path, "r") as config_file:
        config_dict = json.load(config_file)

    if not isinstance(config_dict, dict):
        print_msg(f"Config file {config_path} is incorrectly formatted." +
                  "It must only consist of one dictionary of all relevant" +
                  "parameters.", level=LOG_FATAL)
    # Retrieve configuration parameters

    # TODO try the t/e block once things are working smoothly
    
    """
    build_model_keys = ["nmade", "ndim", "ndim_label", "hidden_layers", "hidden_unts"]

    try:
        for i, key in enumerate(build_model_keys):
            eval(f"{key} = config_dict[{key}]")
            if not isinstance(eval(f"{key}"), int):
                raise TypeError()
    except KeyError:
        print_msg(f"Config file {config_path} lacks entry `{key}",
                  level=LOG_FATAL)
    except TypeError:
        print_msg(f"Config file {config_path} does not have the correct type" +
                  f"for entry `{key}`", level=LOG_FATAL)
    """
    # until then, allow the keyerror and exit
    nmade = config_dict["nmade"]
    ndim = config_dict["ndim"]
    ndim_label = config_dict["ndim_label"]
    hidden_layers = config_dict["hidden_layers"]
    hidden_units = config_dict["hidden_units"]

    # nmade = config_dict.get("nmade", None)
    # ndim = config_dict.get("ndim", None)
    # ndim_label = config_dict.get("ndim_label", None)
    # hidden_layers = config_dict.get("hidden_layers", None)
    # hidden_units = config_dict.get("hidden_units", None)
    
    # Build a model from scratch
    if newmodel:
        model, distribution, made_list = compile_MADE_model(
            nmade, num_inputs=ndim, num_cond_inputs=ndim_label,
            hidden_layers=hidden_layers, hidden_units=hidden_units)
        
        print("-----------------------------------------------------")

        print(made_list)

    # Load a model and extract its skeleton of MAFs
    else:
        # Define model's custom objects
        custom_objects = {"lossfn_MADE": lossfn_MADE, "Made": MADE}
        model = kload_model(flow_path, custom_objects=custom_objects)
        made_blocks = []
        for i in range(nmade):
            made_blocks.append(model.get_layer(name=f"made_{i}"))

        print(made_blocks)

        print("-----------------------------------------------------")

        distribution, made_list = build_MADE(made_blocks,
            ndim, num_made=nmade, hidden_layers=hidden_layers,
            hidden_units=hidden_units)
        
        print(made_list)

        print("-----------------------------------------------------")

    return model, distribution, made_list