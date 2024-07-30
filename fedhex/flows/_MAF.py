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
from tensorflow_probability.python.bijectors import AutoregressiveNetwork
from tensorflow_probability.python.bijectors import Chain
from tensorflow_probability.python.bijectors import Permute
from tensorflow_probability.python.bijectors import MaskedAutoregressiveFlow as MAF
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.distributions import Normal
from tensorflow_probability.python.distributions import Sample
from tensorflow_probability.python.distributions import TransformedDistribution

from fedhex.constants import DEFAULT_SEED

from ..utils import print_msg, LOG_WARN
from ._loss import NLL


@keras.saving.register_keras_serializable(name="Made")
class MADE(AutoregressiveNetwork):
    """
    A duplicate of tfp.bijectors.AutoregressiveNetwork class with tanh applied
    on the output log-scape. This is important for improved regularization and
    `inf` and `nan` values that would otherwise often occur during training.
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
        if self.params == 1:
            return result
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
    #     sublayer = keras.saving.deserialize_keras_object(sublayer_config)
    #     return cls(sublayer, **config)


def MADE_factory(params=None,
                 event_shape=None,
                 conditional=True,
                 conditional_event_shape=None,
                 conditional_input_layers="all_layers",
                 hidden_units=None,
                 input_order="left-to-right",
                 hidden_degrees="equal",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 validate_args=False,
                 **kwargs) -> Callable[..., MADE]:

    name = kwargs.pop("name", "made")

    def f(name_suffix: str=""):
        return MADE(params=params,
                    event_shape=event_shape,
                    conditional=conditional,
                    conditional_event_shape=conditional_event_shape,
                    conditional_input_layers=conditional_input_layers,
                    hidden_units=hidden_units,
                    input_order=input_order,
                    hidden_degrees=hidden_degrees,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                    validate_args=validate_args,
                    name=name + name_suffix,
                    **kwargs)
    
    return f

def MAF_factory(shift_and_log_scale_fn=None,
                bijector_fn=None,
                is_constant_jacobian=False,
                validate_args=False,
                unroll_loop=False,
                event_ndims=1,
                name="maf") -> Callable[..., MADE]:

    def f(name_suffix: str=""):
        return MAF(shift_and_log_scale_fn=shift_and_log_scale_fn,
                   bijector_fn=bijector_fn,
                   is_constant_jacobian=is_constant_jacobian,
                   validate_args=validate_args,
                   unroll_loop=unroll_loop,
                   event_ndims=event_ndims,
                   name=name + name_suffix)
    
    return f


def compile_MAF(num_flows: int,
                len_event: int,
                len_cond_event: int|None,
                hidden_units: list[int],
                activation: str="relu",
                base: Distribution=None,
                optimizer: keras.optimizers.Optimizer=None,
                loss: Callable|tf.losses.Loss=None,
                MADE_kwargs: dict|None=None,
                MAF_kwargs: dict|None=None,
                **kwargs
                ) -> tuple[
                    keras.Model,
                    TransformedDistribution,
                    list[Any]]:
    """Build new Masked Autoregressive Flow from scratch

    Args:
        num_flows (int): number of transformations to the base distribution
        len_event (int): number of features in the vectorized input data
        len_cond_event (int | None): number of features in the vectorized\
            conditional input data. If None, the flow will not learn a\
            conditional density but instead learn to model the entire data set.
        hidden_units (list[int]): list of parameters in each layer of the MADE
        activation (str, optional): activation function between each layer in\
            each made. Defaults to "relu".
        base (Distribution, optional): base distribution from which samples\
            are drawn and then transformed by the flow. If None, the base\
            distribution is `len_event` standard normals. Defaults to None.
        optimizer (keras.optimizers.Optimizer, optional): optimizer used for\
            learning the parameters of the transformations. If None, the\
            default Tensorflow ADAM implementation is used. Defaults to None.
        loss (Callable | tf.losses.Loss, optional): training objective to be\
            minimized. Defaults to loss_MAF.
        MADE_kwargs (dict): additional arguments for the Tensorflow\
            AutoregressiveNetwork class. Defaults to None.
        MAF_kwargs (dict): additional arguments for the Tensorflow\
            MaskedAutoregressiveFlows class. Defaults to None.
        kwargs: keras.Model.compile keyword arguments

    Returns:
        model (keras.Model): traininable Keras flow model
        distribution (TransformedDistribution): distribution whose parameters\
            are learned during training and whose samples of the base\
            distribution are transformed through the flow to match the target\
            distribution
        maf_list (list[MAF]): list of each MAF in the chain
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

    # Provide base distribution to be transformed into the target distribution
    if not bool(base) or not isinstance(base, Distribution):
        print_msg(f"Using default base distribution: {len_event} joint Standard Normals")
        base = Sample(Normal(loc=0., scale=1.),
                      sample_shape=[len_event])

    event_shape = (len_event, )
    if len_cond_event is None:
        conditional = False
        cond_event_shape = None
    else:
        conditional = True
        cond_event_shape = (len_cond_event,)

    if MADE_kwargs is None:
        MADE_kwargs = {}

    if MAF_kwargs is None:
        MAF_kwargs = {}

    # construct a list of all flows and permute input dimensions btwn each
    maf_list = []
    # reverse order of input elements between layers
    permutation = np.arange(len_event)[::-1]
    for i in range(num_flows):
        
        # 2 params indicates learning one param for the scale and the shift
        made = MADE(params=2,
                    hidden_units=hidden_units,
                    event_shape=event_shape,
                    conditional=conditional,
                    conditional_event_shape=cond_event_shape,
                    activation=activation,
                    name=f"made_{i}",
                    **MADE_kwargs)
        
        maf = MAF(shift_and_log_scale_fn=made,
                  name=f"maf_{i}",
                  **MAF_kwargs)
        maf_list.append(maf)

        # there is no permute on the output layer
        if i < num_flows - 1:
            perm = Permute(permutation)
            maf_list.append(perm)

    # chain the flows together to complete bijection
    chain = Chain(list(reversed(maf_list)))

    # transform a distribution of joint std normals to our target distribution
    distribution = TransformedDistribution(
        distribution=base,
        bijector=chain)
    
    # Data and Conditional Data input layers
    x_ = keras.layers.Input(shape=(len_event,), name="aux_input")
    input_list = [x_]
    current_kwargs = {}
    if len_cond_event is not None:
        c_ = keras.layers.Input(shape=(len_cond_event,), name="cond_input")
        input_list.append(c_)

        # Provide input tensors/placeholders for conditional data to the bijectors
        for i in range(num_flows):
            current_kwargs[f"maf_{i}"] = {"conditional_input" : c_}

    # Log-likelihood output of distribution is network output
    log_prob_ = distribution.log_prob(x_, bijector_kwargs=current_kwargs)
    model = keras.Model(input_list, log_prob_)
    model.compile(optimizer=optimizer, loss=loss, **kwargs)

    return model, distribution, maf_list


def mask_outside(arr: np.ndarray, ranges: list[list[float]]):
    mask = np.zeros(shape=arr.shape[0], dtype=bool)
    for i in range(len(ranges)):
        mask |= (arr[:,i] < ranges[i][0]) | (arr[:,i] > ranges[i][1])
    return np.array(mask)


def eval_MAF(cond,
             made_list: list,
             dist: TransformedDistribution,
            #  dm: DataManager=None,
             criteria: Callable=None,
             ranges: list[list[float]]=None,
             seed: int=DEFAULT_SEED,
             *args) -> np.ndarray:
    
    # if dm is not None:
    #     cond = dm.norm(samples=cond, is_cond=True)

    num_flows = len([layer.name for layer in made_list if layer.name[:3] == "maf"])

    # Pass the flow the conditional inputs (labels)
    current_kwargs = {}
    for i in range(num_flows):
        current_kwargs[f"maf_{i}"] = {"conditional_input" : cond}
    
    # Evaluate the flow at each of the provided conditional inputs
    gen_data_norm = np.array(dist.sample(len(cond),
                                         bijector_kwargs=current_kwargs,
                                         seed=seed))
    
    # Use DM to transform data into problem space where criteria are defined
    # if dm is not None:
    #     gen_data = dm.denorm(gen_data_norm, is_cond=False)
    gen_data = gen_data_norm

    #If there is no rejection criterion, return the generated data
    if not bool(criteria) and not bool(ranges):
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
        for i in range(num_flows):
            current_kwargs[f"maf_{i}"] = {"conditional_input" : cond[mask]}
        
        # Re-sample
        gen_resample_norm = np.array(dist.sample(np.sum(mask),
                                                 bijector_kwargs=current_kwargs,
                                                 seed=seed))
        # if dm is not None:
        #     gen_data[mask] = dm.denorm(gen_resample_norm, is_cond=False)
        # else:
        gen_data[mask] = gen_resample_norm
        
        # Update mask based on rejection criteria applied to new data
        mask = criteria(gen_data, *args)
    
    return gen_data


def intermediate_MAF(prior, made_list):

    """
    Separate each step of the flow into individual distributions in order to
    samples from and test each bijection's output.
    """

    num_flows = len([layer.name for layer in made_list if layer.name[:3] == "maf"])

    # reverse the list of made blocks to unpack in generating direction
    made_list_rev = list(reversed(made_list[:-1]))

    feat_extraction_dists = []

    made_chain = Chain([])
    dist = TransformedDistribution(
        distribution=prior,
        bijector=made_chain)
    feat_extraction_dists.append(dist)

    # for made_block in made_list_rev:
    #     made_chain = tfb.Chain(made_list_rev[0:i])
    #     dist = TransformedDistribution(
    #         distribution=prior,
    #         bijector=made_chain)
    #     feat_extraction_dists.append(dist)

    return feat_extraction_dists


def load_MAF(flow_path: str|None=None,
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
    layer_names = [layer.name for layer in model.submodules if isinstance(layer, MADE)]

    num_flows = len(layer_names)
    event_shape = model.get_layer("aux_input").input_shape[0][-1]
    cond_event_shape = model.get_layer("cond_input").input_shape[0][-1]
    
    made0 = model.get_layer(layer_names[0])
    activation = made0.activation
    hidden_units = list(made0.hidden_units)
    
    made_list = []
    maf_list = []

    permutation = np.arange(0, event_shape)[::-1]
    for i in range(num_flows):
        made = model.get_layer(name=f"made_{i}")
        maf = MAF(shift_and_log_scale_fn=made, name=f"maf_{i}")
        perm = Permute(permutation)
        
        made_list.append(made)
        maf_list.append(maf)
        # there is no permute on the output layer
        if i < num_flows - 1:
            maf_list.append(perm)

    # chain the flows together to complete bijection
    chain = Chain(list(reversed(maf_list)))

    # transform a distribution of joint std normals to our target distribution
    distribution = TransformedDistribution(
        distribution=Sample(Normal(loc=0., scale=1.), sample_shape=[event_shape]),
        bijector=chain)
    
    cfg = dict(num_flows=num_flows,
               len_event=event_shape,
               len_cond_event=cond_event_shape,
               hidden_units=hidden_units,
               activation=activation)
    
    return model, distribution, made_list, cfg