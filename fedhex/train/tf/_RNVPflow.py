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
from tensorflow import keras as tfk
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import TransformedDistribution
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import RealNVP
from typing import Any, Callable

from ...io._path import LOG_WARN, LOG_FATAL, print_msg
from ..._loaders import DataManager


@tfk.saving.register_keras_serializable(name="lossfn_RNVP")
def lossfn_RNVP(x, logprob, lam: np.float64=0.0):
    return -logprob + lam * tf.reduce_sum(tf.square(x), axis=None)


        # config.update({"params": self.params,
        #                "event_shape": self.event_shape,
        #                "conditional": self.conditional,
        #                "conditional_event_shape": self.conditional_event_shape,
        #                "hidden_units": self.hidden_units,
        #                "activation": self.activation,
        #                "name": self.name})


def build_RNVP(blocks: list,
               num_inputs: int,
               num_blocks: int=10,
               hidden_layers: int|list=1,
               hidden_units: int=128,
               activation: str="relu",
               cond_event_shape: tuple=None)-> tuple[
                   TransformedDistribution,
                   list]:

    if cond_event_shape is None:
        cond_event_shape = (num_inputs, )
    
    # Since we later want to use a Gaussian as our autoregressive distribution, we need to set
    # "params" to 2, such that the MADE network can parameterize its mean and logarithmic standard deviation.

    if blocks is None or len(blocks) == 0:
        blocks = []

        if isinstance(hidden_layers, list):
            hidden_units_list = hidden_layers
        else:
            hidden_units_list = hidden_layers * [hidden_units]
        
        for i in range(num_blocks):
            # 2 params indicates 1 param for each the scale and the shift
            block = RealNVP(num_masked=num_inputs-1,
                            shift_and_log_scale_fn=tfb.real_nvp_default_template(
                                hidden_layers=hidden_units_list,
                                activation=activation),
                event_shape=(num_inputs,), conditional=True,
                conditional_event_shape=cond_event_shape,
                activation=activation, name=f"rnvp_{i}")
            blocks.append(block)
            blocks.append(tfb.Permute(np.arange(0, num_inputs)[::-1]))

    # make bijection from made layers; remove final permute layer
    chain = tfb.Chain(list(reversed(blocks[:-1])))

    # we want to transform to gaussian distribution with mean 0 and std 1 in latent space
    distribution = TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[num_inputs]),
        bijector=chain)
    
    return distribution, blocks


def compile_MADE_model(num_made: int,
                       num_inputs: int,
                       num_cond_inputs: int=None,
                       hidden_layers: int|list=1,
                       hidden_units: int=128,
                       activation: str="relu",
                       loss: Callable=lossfn_MADE,
                       opt: tfk.optimizers.Optimizer=None) -> tuple[
                           tfk.Model,
                           TransformedDistribution,
                           list[Any]]:
    """
    Build new model from scratch
    """

    # Define optimizer/learning rate function
    if opt is None or not isinstance(opt, tfk.optimizers.Optimizer):
        learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=1e-3, decay_steps=100,
                end_learning_rate=1e-4, power=1.0, cycle=False)
        opt = tfk.optimizers.Adam(learning_rate=learning_rate_fn)
    
    # Build model layers and compile
    made_list = []
    distribution, made_list = build_RNVP(made_list, num_inputs,
            hidden_layers=hidden_layers, hidden_units=hidden_units,
            cond_event_shape=(num_cond_inputs, ), num_blocks=num_made,
            activation=activation)
    
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

    return model, distribution, made_list

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


def load_MADE(flow_path: str|None=None,
              newmodel: bool=True)-> tuple[
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

    config_path = flow_path + "/config.json"
    if not os.path.isfile(config_path):
        print_msg(f"The model at '{flow_path}' is missing the model config " +
                  f"file at {config_path}. A new model is going to be " +
                  f"created at '{flow_path}'.", level=LOG_WARN)
        return tuple(None, None, None, None)
    
    with open(config_path, "r") as config_file:
        config_dict = json.load(config_file)

    if not isinstance(config_dict, dict):
        print_msg(f"Config file {config_path} is incorrectly formatted." +
                  "It must only consist of one dictionary of all relevant" +
                  "parameters.", level=LOG_FATAL)
        return tuple(None, None, None, None)
    # Retrieve configuration parameters

    # TODO try the t/e block once things are working smoothly
    
    """
    build_model_keys = ["nmade", "ndim", "ndim_label", "hidden_layers", "hidden_unts"]

    try:
        for i, key in enumerate(build_model_keys):
            eval(f"{key} = config_dict[{key}]")
            if not isinstance(eval(key), int):
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
    ndim = config_dict["ninputs"]
    ndim_label = config_dict["ncinputs"]
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
        model = tfk.models.load_model(flow_path, custom_objects=custom_objects)
        made_blocks = []
        for i in range(nmade):
            made_blocks.append(model.get_layer(name=f"made_{i}"))

        print(made_blocks)

        print("-----------------------------------------------------")

        distribution, made_list = build_RNVP(made_blocks,
            ndim, num_blocks=nmade, hidden_layers=hidden_layers,
            hidden_units=hidden_units)
        
        print(made_list)

        print("-----------------------------------------------------")

    return model, distribution, made_list, config_dict