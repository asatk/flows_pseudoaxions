"""
Author: Anthony Atkinson and Darshan Lakshminarayanan
Modified: 2024.03.05

Contains the core flow model necessities. Building a flow, compiling one from
scratch, the loss function, retrieving the intermediate flows, and the MADE
block itself.

It will likely contain other types of blocks/flows, in which case PyTorch is
more favorable due to its exposed interface and customizability.
"""

import os
import types
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import Distribution, TransformedDistribution

from .._loaders import DataManager
from ..io._path import LOG_WARN, print_msg
from ._loss import NLL


# TODO turn build/compile/eval _MAF into a MAF(keras.Model) class

def build_flow(flow_class,
               bijector_class,
               num_flows: int,
               len_event: int,
               len_cond_event: int|None,
               hidden_units: list[int],
               activation: str="relu",
               prior: Distribution=None,
               bijector_kwargs: dict|None=None,
               flow_kwargs: dict|None=None
               )-> tuple[
                   TransformedDistribution,
                   list]:
    """Construct a flow using a chain of bijectors.

    Args:
        num_flows (int): number of transformations to the base distribution
        len_event (int): number of features in the vectorized input data
        len_cond_event (int | None): number of features in the vectorized\
            conditional input data. If None, the flow will not learn a\
            conditional density but instead learn to model the entire data set.
        hidden_units (list[int]): list of parameters in each layer of the MADE
        activation (str, optional): activation function between each layer in\
            each made. Defaults to "relu".
        prior (Distribution, optional): base distribution from which samples\
            are drawn and then transformed by the flow. If None, the base\
            distribution is `len_event` standard normals. Defaults to None.
        bijector_kwargs (dict): additional arguments for the Tensorflow\
            AutoregressiveNetwork class. Defaults to None.
        flow_kwargs (dict): additional arguments for the Tensorflow\
            MaskedAutoregressiveFlows class. Defaults to None.
            
    Returns:
        distribution (TransformedDistribution): distribution whose parameters\
            are learned during training and whose samples of the base\
            distribution are transformed through the flow to match the target\
            distribution
        flow_list (list): list of each flow in the chain
    """

    if prior is None or not isinstance(prior, Distribution):
        print_msg(f"Using default prior/base distribution: {len_event} joint Standard Normals")
        prior = tfd.Sample(tfd.Normal(loc=0., scale=1.),
                           sample_shape=[len_event])

    event_shape = (len_event, )
    if len_cond_event is None:
        conditional = False
        cond_event_shape = None
    else:
        conditional = True
        cond_event_shape = (len_cond_event,)

    if bijector_kwargs is None:
        bijector_kwargs = {}

    if flow_kwargs is None:
        flow_kwargs = {}

    # construct a list of all flows and permute input dimensions btwn each
    flow_list = []
    for i in range(num_flows):
        
        # TODO do bijector.property for the ones used in load/all func signatures

        # 2 params indicates learning one param for the scale and the shift
        bijector = bijector_class(params=2,
                    hidden_units=hidden_units,
                    event_shape=event_shape,
                    conditional=conditional,
                    conditional_event_shape=cond_event_shape,
                    activation=activation,
                    name=f"bijector_{i}",
                    **bijector_kwargs)
        
        flow = flow_class(shift_and_log_scale_fn=bijector,
                          name=f"flow_{i}",
                          **flow_kwargs)
        
        perm = tfb.Permute(np.arange(0, len_event)[::-1])
        
        flow_list.append(flow)
        # there is no permute on the output layer
        if i < num_flows - 1:
            flow_list.append(perm)

    # chain the flows together to complete bijection
    chain = tfb.Chain(list(reversed(flow_list)))

    # transform a distribution of joint std normals to our target distribution
    distribution = TransformedDistribution(
        distribution=prior,
        bijector=chain)
    
    return distribution, flow_list


def compile_flow(flow_class,
                 bijector_class,
                 num_flows: int,
                 len_event: int,
                 len_cond_event: int|None,
                 hidden_units: list[int],
                 activation: str="relu",
                 prior: Distribution=None,
                 optimizer: keras.optimizers.Optimizer=None,
                 loss: Callable|tf.losses.Loss=None,
                 bijector_kwargs: dict|None=None,
                 flow_kwargs: dict|None=None,
                 **kwargs
                 ) -> tuple[
                     keras.Model,
                     TransformedDistribution,
                     list[Any]]:
    """Compile a new flow from scratch

    Args:
        num_flows (int): number of transformations to the base distribution
        len_event (int): number of features in the vectorized input data
        len_cond_event (int | None): number of features in the vectorized\
            conditional input data. If None, the flow will not learn a\
            conditional density but instead learn to model the entire data set.
        hidden_units (list[int]): list of parameters in each layer of the MADE
        activation (str, optional): activation function between each layer in\
            each made. Defaults to "relu".
        prior (Distribution, optional): base distribution from which samples\
            are drawn and then transformed by the flow. If None, the base\
            distribution is `len_event` standard normals. Defaults to None.
        optimizer (keras.optimizers.Optimizer, optional): optimizer used for\
            learning the parameters of the transformations. If None, the\
            default Tensorflow ADAM implementation is used. Defaults to None.
        loss (Callable | tf.losses.Loss, optional): training objective to be\
            minimized. Defaults to loss_MAF.
        bijector_kwargs (dict): additional arguments for the Tensorflow\
            AutoregressiveNetwork class. Defaults to None.
        flow_kwargs (dict): additional arguments for the Tensorflow\
            MaskedAutoregressiveFlows class. Defaults to None.
        kwargs: keras.Model.compile keyword arguments

    Returns:
        model (keras.Model): traininable Keras flow model
        distribution (TransformedDistribution): distribution whose parameters\
            are learned during training and whose samples of the base\
            distribution are transformed through the flow to match the target\
            distribution
        flow_list (list): list of each flow in the chain
    """

    # Define optimizer/learning rate function
    if optimizer is None or not isinstance(optimizer, keras.optimizers.Optimizer):
        print_msg("Using default optimizer and learning rate schedule: Adam " \
                  "with learning rate=1e-3")
        optimizer = keras.optimizers.Adam()

    if loss is None or not isinstance(loss, types.FunctionType|tf.losses.Loss):
        print_msg("Using default loss function: NLL")
        loss = NLL()
    else:
        if isinstance(loss, types.FunctionType):
            _name = loss.__name__ 
        else:
            _name = loss.__class__.__name__
        print_msg(f"Registering custom loss '{_name}' as 'Custom>custom_loss'")
        keras.saving.register_keras_serializable(package="Custom",
                                                 name="custom_loss")(loss)

    # Build model layers and compile
    distribution, flow_list = build_flow(
        flow_class=flow_class,
        bijector_class=bijector_class,
        num_flows=num_flows,
        len_event=len_event,
        len_cond_event=len_cond_event,
        hidden_units=hidden_units,
        activation=activation,
        prior=prior,
        bijector_kwargs=bijector_kwargs,
        flow_kwargs=flow_kwargs)
    
    # Data and Conditional Data input layers
    x_ = keras.layers.Input(shape=(len_event,), name="aux_input")
    input_list = [x_]
    current_kwargs = {}
    if len_cond_event is not None:
        c_ = keras.layers.Input(shape=(len_cond_event,), name="cond_input")
        input_list.append(c_)

        # Provide input tensors/placeholders for conditional data to the bijectors
        for i in range(num_flows):
            current_kwargs[f"flow_{i}"] = {"conditional_input" : c_}

    # Log-likelihood output of distribution is network output
    log_prob_ = distribution.log_prob(x_, bijector_kwargs=current_kwargs)
    model = keras.Model(input_list, log_prob_)
    model.compile(optimizer=optimizer, loss=loss, **kwargs)

    return model, distribution, flow_list


def mask_outside(arr: np.ndarray, ranges: list[list[float]]):
    mask = np.zeros(shape=arr.shape[0], dtype=bool)
    for i in range(len(ranges)):
        mask |= (arr[:,i] < ranges[i][0]) | (arr[:,i] > ranges[i][1])
    return np.array(mask)


def eval_flow(cond,
              made_list: list,
              dist: TransformedDistribution,
              dm: DataManager=None,
              criteria: Callable=None,
              ranges: list[list[float]]=None,
              seed: int=0x2024,
              *args) -> np.ndarray:
    
    if dm is not None:
        cond = dm.norm(samples=cond, is_cond=True)

    # Pass the flow the conditional inputs (labels)
    current_kwargs = {}
    for i in range(len(made_list) // 2):
        current_kwargs[f"maf_{i}"] = {"conditional_input" : cond}
    
    # Evaluate the flow at each of the provided conditional inputs
    gen_data_norm = np.array(dist.sample(len(cond),
                                         bijector_kwargs=current_kwargs,
                                         seed=seed))
    
    # Use DM to transform data into problem space where criteria are defined
    if dm is not None:
        gen_data = dm.denorm(gen_data_norm, is_cond=False)
   
    #If there is no rejection criterion, return the generated data
    if criteria is None and ranges is None:
        return gen_data
    
    # Handle `None` as +/- inifity when `ranges` is used
    if ranges is not None:
        criteria = mask_outside
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
        gen_resample_norm = np.array(dist.sample(np.sum(mask),
                                                 bijector_kwargs=current_kwargs,
                                                 seed=seed))
        if dm is not None:
            gen_data[mask] = dm.denorm(gen_resample_norm, is_cond=False)
        else:
            gen_data[mask] = gen_resample_norm
        
        # Update mask based on rejection criteria applied to new data
        mask = criteria(gen_data, *args)
    
    return gen_data


def intermediate_flow(flow_list):

    """
    Separate each step of the flow into individual distributions in order to
    samples from and test each bijection's output.
    """

    # reverse the list of made blocks to unpack in generating direction
    flow_list_rev = list(reversed(flow_list[:-1]))

    feat_extraction_dists = []

    made_chain = tfb.Chain([])
    dist = TransformedDistribution(
        distribution=tfd.Sample(
            tfd.Normal(loc=0., scale=1.),
            sample_shape=[2]), # TODO shape is not 2 in general
        bijector=made_chain)
    feat_extraction_dists.append(dist)

    for i in range(1, len(flow_list_rev), 2):
        made_chain = tfb.Chain(flow_list_rev[0:i])
        dist = TransformedDistribution(
            distribution=tfd.Sample(
                tfd.Normal(loc=0., scale=1.),
                sample_shape=[2]),
            bijector=made_chain)
        feat_extraction_dists.append(dist)

    return feat_extraction_dists


def load_flow(flow_class,
              bijector_class,
              flow_path: str|None=None,
              loss: Callable|keras.losses.Loss=None)-> tuple[
                   keras.Model,
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
    
    if loss is None or not isinstance(loss, Callable|tf.losses.Loss):
        print_msg("Using default loss function: NLL")
    else:
        if isinstance(loss, types.FunctionType):
            _name = loss.__name__ 
        else:
            _name = loss.__class__.__name__
        print_msg(f"Registering custom loss {'_name'} as 'Custom>custom_loss'")
        keras.saving.register_keras_serializable(package="Custom",
                                                 name="custom_loss")(loss)
    

    model: keras.Model= keras.models.load_model(flow_path)
    layer_names = [layer.name for layer in model.submodules if isinstance(layer, bijector_class)]

    num_flows = len(layer_names)
    event_shape = model.get_layer("aux_input").input_shape[0][-1]
    cond_event_shape = model.get_layer("cond_input").input_shape[0][-1]
    
    bijector0 = model.get_layer(layer_names[0])
    activation = bijector0.activation
    hidden_units = list(bijector0.hidden_units)
    
    bijector_list = []
    flow_list = []

    for i in range(num_flows):
        bijector = model.get_layer(name=f"bijector_{i}")
        flow = flow_class(shift_and_log_scale_fn=bijector, name=f"flow_{i}")
        perm = tfb.Permute(np.arange(0, event_shape)[::-1])
        
        bijector_list.append(bijector)
        flow_list.append(flow)
        # there is no permute on the output layer
        if i < num_flows - 1:
            flow_list.append(perm)

    # chain the flows together to complete bijection
    chain = tfb.Chain(list(reversed(flow_list)))

    # transform a distribution of joint std normals to our target distribution
    distribution = TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[event_shape]),
        bijector=chain)
    
    cfg = dict(num_flows=num_flows,
               len_event=event_shape,
               len_cond_event=cond_event_shape,
               hidden_units=hidden_units,
               activation=activation)
    
    return model, distribution, bijector_list, cfg