import abc
import numpy as np
import keras
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.bijectors import AutoregressiveNetwork, MaskedAutoregressiveFlow, Bijector
from typing import Callable

from fedhex.constants import DEFAULT_SEED


class FlowComponent(metaclass=abc.ABCMeta):

    _is_flow: bool=False
    
    def __init__(self,
                 name: str="comp"):
        self._name = name
    
    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def make(self):
        ...

    def __add__(self, value):
        if isinstance(value, FlowComponent):
            return FlowBuilder(components=[self, value])
        else:
            return NotImplemented

    def __call__(self):
        return self.make()


class NormalizingFlowComponent(FlowComponent):

    _is_flow: bool = True

    def __init__(self,
                 conditional: bool=False,
                 name: str="norm_flow"):
        super().__init__(name=name)
        self._conditional = conditional

    @property
    def conditional(self):
        return self._conditional


class Flow(keras.Model):

    def __init__(self,
                 base,
                 chain,
                 num_flows: int,
                 conditional: bool=False,
                 bijector_name: str="flow"):
        super().__init__()
        self._base = base
        self._chain = chain
        self._num_flows = num_flows
        self._dist = tfd.TransformedDistribution(
            distribution=base,
            bijector=chain
        )
        self._preprocess = None
        self._conditional = conditional
        self._bijector_name = bijector_name

        self._data_mean = 0
        self._data_std = 1
        if conditional:
            self._conditional_data_mean = 0
            self._conditional_data_std = 1

    @property
    def num_flows(self) -> int:
        return self._num_flows
    
    @property
    def bijector_name(self) -> str:
        return self._bijector_name

    def adapt(self, x):
        if self._conditional:
            if not isinstance(x, tuple(list, tuple)) or len(x) != 2:
                return ValueError("Conditional flow must be provided `conditional_input`")
            # x = tf.concat(x, axis=0)
            x, c = x
            self._conditional_data_mean = np.mean(c, axis=-1)
            self._conditional_data_std = np.std(c, axis=-1)
        self._data_mean = np.mean(x, axis=-1)
        self._data_std = np.std(x, axis=-1)

    def sample(self,
               num_samples: int=None,
               c=None,
               seed: int=DEFAULT_SEED):
        if c is not None:
            if bool(num_samples) and num_samples != c.shape[0]:
                return ValueError("`num_samples` differs from first dimension"\
                                  "of provided conditional data.")
            c = (c - self._conditional_data_mean) / self._conditional_data_std
            bijector_kwargs = {
                f"{self.bijector_name}_{i}": {"conditional_input": c}\
                    for i in range(self.num_flows)
            }
        else:
            bijector_kwargs = {}
        
        y = self._dist.sample(sample_shape=num_samples,
                              seed=seed,
                              bijector_kwargs=bijector_kwargs)
        return y * self._data_std + self._data_mean
        
    def call(self, x):
        bijector_kwargs = {}
        if self._conditional:
            if not isinstance(x, (list, tuple)) or len(x) != 2:
                return ValueError("Conditional flow must be provided `conditional_input`")
            x, c = x
            c = (c - self._conditional_data_mean) / self._conditional_data_std
            bijector_kwargs = {
                f"{self.bijector_name}_{i}": {"conditional_input": c}\
                    for i in range(self.num_flows)
            }
        x = (x - self._data_mean) / self._data_std
        return self._dist.log_prob(x, bijector_kwargs=bijector_kwargs)

    def get_config(self):
        model_config = super().get_config()
        # base_config = self._base.get_config()
        # dist_config = self._dist.get_config()

        config = {"num_flows": self._num_flows,
                  "conditional": self._conditional,
                  "bijector_name": self._bijector_name,
                #   "data_mean": self._data_mean,
                #   "data_std": self._data_std,
                #   "conditional_data_mean": self._conditional_data_mean,
                #   "conditional_data_std": self._conditional_data_std,
                #   **base_config,
                #   **dist_config,
                  **model_config
        }

        return config


#TODO MAKE PRESET BUILDER
class FlowBuilder():
    
    def __init__(self, components: list[FlowComponent]=None):
        self.components = [] if components is None else components
    
    def build(self, base) -> Flow:
        bijectors = []
        flow_name = None
        conditional = False
        num_blocks = 0
        for component in self.components:
            if component._is_flow:
                num_blocks += 1
                flow_name = component.name
                conditional = component.conditional
            bijectors.append(component())

        if flow_name is None:
            return ValueError("No normalizing flow was included in the body of the FlowBuilder. Please include a NormalizingFlowComponent")

        chain = tfb.Chain(list(reversed(bijectors)))
        flow = Flow(base=base,
                    chain=chain,
                    num_flows=num_blocks,
                    bijector_name=flow_name,
                    conditional=conditional)
        return flow

    def __len__(self):
        return len(self.components)

    def __add__(self, value):
        if isinstance(value, FlowComponent):
            return FlowBuilder(components=self.components + [value])
        elif isinstance(value, FlowBuilder):
            return FlowBuilder(components=self.components + value.components)
        return NotImplemented

    def __radd__(self, value):
        if isinstance(value, FlowComponent):
            return FlowBuilder(components=[value] + self.components)
        elif isinstance(value, FlowBuilder):
            return FlowBuilder(components=value.components + self.components)
        return NotImplemented

    def __mul__(self, value):
        if isinstance(value, int):
            return FlowBuilder(components=self.components * value)
        return NotImplemented

    def __rmul__(self, value):
        if isinstance(value, int):
            return FlowBuilder(components=self.components * value)
        return NotImplemented

    def __imul__(self, value):
        if isinstance(value, int):
            self.components *= value
            return self
        else:
            return NotImplemented

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= len(self.components):
                return ValueError("index out of range of component list length")
            return FlowBuilder(self.components[index])
        elif isinstance(index, slice):
            if index.stop is not None and index.stop >= len(self.components):
                return ValueError("index out of range of component list length")
            return FlowBuilder(self.components[index])
        return ValueError("only integer indexes accepted")
    
    def __setitem__(self, index, value):
        if isinstance(index, int):
            if index >= len(self.components):
                return ValueError("index out of range of list length")
            if isinstance(value, FlowComponent):
                self.components[index] = value
            else:
                return ValueError("only objects of type `FlowComponent` can be set in a FlowBuilder")
        elif isinstance(index, slice):
            if index.stop is not None and index.stop >= len(self.components):
                return ValueError("slice out of range of component list length")
            if isinstance(value, (list, tuple)):
                # self.components[index] = value
                for i, v in zip(index, value):
                    self[i] = v
        return ValueError("only integer or slice indexes accepted")
    
    def __iter__(self):
        return iter(self.components)


class MAFComponent(NormalizingFlowComponent):

    def __init__(self,
                 shift_and_log_scale_fn: keras.layers.Layer=None,
                 bijector_fn: keras.layers.Layer=None,
                 is_constant_jacobian=False,
                 validate_args=False,
                 unroll_loop=False,
                 event_ndims=1,
                 conditional: bool=False,
                 name="maf"):

        self.is_constant_jacobian = is_constant_jacobian
        self.validate_args = validate_args
        self.unroll_loop = unroll_loop
        self.event_ndims = event_ndims

        if bool(shift_and_log_scale_fn) == True:
            self._config = shift_and_log_scale_fn.get_config()
            self._layer_class = shift_and_log_scale_fn.__class__
            self._layer_name = shift_and_log_scale_fn.name
            self._use_shift_and_scale = True
        elif bool(bijector_fn) == True:
            self._config = bijector_fn.get_config()
            self._layer_class = bijector_fn.__class__
            self._layer_name = bijector_fn.name
            self._use_shift_and_scale = False
        else:
            return ValueError(
                "Either `bijector_fn_factory` or "\
                "`shift_and_log_scale_fn_factory` must be specified"
            )
        
        self._counter = 0
        
        super().__init__(conditional=conditional,
                         name=name)
        
    def make(self):
        name = f"{self.name}_{self._counter}"
        
        layer_name = f"{self._layer_name}_{self._counter}"
        config = self._config.copy()
        config["name"] = layer_name
        layer = self._layer_class.from_config(config)
        
        self._counter += 1

        if self._use_shift_and_scale:
            return MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=layer,
                bijector_fn=None,
                is_constant_jacobian=self.is_constant_jacobian,
                validate_args=self.validate_args,
                unroll_loop=self.unroll_loop,
                event_ndims=self.event_ndims,
                name=name
            )
        else:
            return MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=None,
                bijector_fn=layer,
                is_constant_jacobian=self.is_constant_jacobian,
                validate_args=self.validate_args,
                unroll_loop=self.unroll_loop,
                event_ndims=self.event_ndims,
                name=name
            )


class Permute(FlowComponent):
    
    def __init__(self,
                 permutation,
                 axis=-1,
                 validate_args: bool=False,
                 name: str="permute"):
        self._p = tfb.Permute(
            permutation=permutation,
            axis=axis,
            validate_args=validate_args,
            name=name
        )
        super().__init__(name=name)

    def make(self):
        return self._p


class BatchNorm(FlowComponent):

    def __init__(self,
                 batchnorm_layer: keras.layers.BatchNormalization|None=None,
                 training=True,
                 validate_args=False,
                 name='batch_normalization_bijector'):
        
        
        self._layer = batchnorm_layer
        if bool(batchnorm_layer):
            self._layer_name = batchnorm_layer.name
            self._config = batchnorm_layer.get_config()
        self.training = training
        self.validate_args = validate_args

        self._counter = 0

        super().__init__(name=name)

    def make(self):

        name = f"{self._name}_{self._counter}"

        if bool(self._layer):
            layer_name = f"{self._layer_name}_{self._counter}"
            config = self._config.copy()
            config["name"] = layer_name
            layer = keras.layers.BatchNormalization.from_config(config)
        else:
            layer = None

        self._counter += 1

        return tfb.BatchNormalization(
            batchnorm_layer=layer,
            training=self.training,
            validate_args=self.validate_args,
            name=name
        )
