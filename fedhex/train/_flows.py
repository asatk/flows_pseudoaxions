import numpy as np
import keras
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.bijectors import AutoregressiveNetwork, MaskedAutoregressiveFlow, Bijector
from typing import Callable

from fedhex.constants import DEFAULT_SEED
from ._loss import NLL


class FlowComponent():

    _is_flow: bool=False
    
    def __init__(self,
                 bijector: Bijector=None,
                 factory: Callable[[int], Bijector]=None,
                 at_head: bool=False,
                 at_tail: bool=False,
                 name: str="comp"):
        if at_head and at_tail:
            return ValueError("`at_head` and `at_tail` cannot both be true")
        if not bijector and not factory:
            return ValueError("One of `bijector` or `factory` must be provided")
        if bijector:
            factory = lambda i=0: bijector
        self._bijector = bijector
        self._factory = factory
        self._at_head = at_head
        self._at_tail = at_tail
        self._name = name

    @property
    def at_head(self):
        return self._at_head
    
    @property
    def at_tail(self):
        return self._at_tail
    
    @property
    def name(self):
        return self._name

    def make(self, i: int=0):
        return self._factory(i)

    def __add__(self, value):
        if isinstance(value, FlowComponent):
            if self.at_head:
                if value.at_head:
                    return FlowBuilder(num_flows=1,
                                       head=[self, value])
                elif value.at_tail:
                    return FlowBuilder(num_flows=1,
                                       head=[self],
                                       tail=[value])
                else:
                    return FlowBuilder(num_flows=1,
                                       head=[self],
                                       body=[value])
            elif self.at_tail:
                if value.at_head:
                    return FlowBuilder(num_flows=1,
                                       head=[value],
                                       tail=[self])
                elif value.at_tail:
                    return FlowBuilder(num_flows=1,
                                       tail=[self, value])
                else:
                    return FlowBuilder(num_flows=1,
                                       body=[value],
                                       tail=[self])
            else:
                if value.at_head:
                    return FlowBuilder(num_flows=1,
                                       head=[value],
                                       body=[self])
                elif value.at_tail:
                    return FlowBuilder(num_flows=1,
                                       body=[self],
                                       tail=[value])
                else:
                    return FlowBuilder(num_flows=1,
                                       body=[self, value])
        else:
            return NotImplemented

    def __call__(self, i: int=0):
        return self._factory(i)


class NormalizingFlowComponent(FlowComponent):

    _is_flow: bool = True

    def __init__(self,
                 bijector: Bijector=None,
                 factory: Callable[[int], Bijector]=None,
                 conditional: bool=False,
                 at_head: bool=False,
                 at_tail: bool=False,
                 name: str="norm_flow"):
        super().__init__(bijector=bijector,
                         factory=factory,
                         at_head=at_head,
                         at_tail=at_tail,
                         name=name)
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
                 flow_name: str="flow"):
        super().__init__()
        self._base = base
        self._chain = chain
        self._num_flows = num_flows
        self._dist = tfd.TransformedDistribution(
            distribution=base,
            bijector=chain
        )
        self._conditional = conditional
        self._flow_name = flow_name

    @property
    def num_flows(self) -> int:
        return self._num_flows
    
    @property
    def flow_name(self) -> str:
        return self._flow_name

        # self._history = {k: self._history.get(k, []) + v
        #                  for k, v in history.history.items()}


    def sample(self,
             num_samples: int=None,
             c=None,
             seed: int=DEFAULT_SEED):
        if c is not None:
            if num_samples is not None and num_samples != c.shape[0]:
                return ValueError("`num_samples` differs from first dimension"\
                                  "of provided conditional data.")
            bijector_kwargs = {
                f"{self.flow_name}_{i}": {"conditional_input": c}\
                    for i in range(self.num_flows)
            }
        else:
            bijector_kwargs = {}
        
        return self._dist.sample(sample_shape=num_samples,
                                 seed=seed,
                                 bijector_kwargs=bijector_kwargs)
        
    def call(self, x):
        bijector_kwargs = {}
        if self._conditional:
            if not isinstance(x, tuple(list, tuple)) or len(x) != 2:
                return ValueError("Conditional flow must be provided `conditional_input`")
            x, c = x
            bijector_kwargs = {
                f"{self.flow_name}_{i}": {"conditional_input": c}\
                    for i in range(self.num_flows)
            }
        return self._dist.log_prob(x, bijector_kwargs=bijector_kwargs)

#TODO MAKE PRESET BUILDER
class FlowBuilder():
    
    def __init__(self,
                 num_flows: int,
                 head: list[FlowComponent]=None,
                 body: list[FlowComponent]=None,
                 tail: list[FlowComponent]=None):
        self._num_flows = num_flows
        self._head = [] if head is None else head
        self._body = [] if body is None else body
        self._tail = [] if tail is None else tail

    @property
    def num_flows(self) -> int:
        return self._num_flows

    @property
    def head(self) -> list[FlowComponent]:
        return self._head
    
    @property
    def body(self) -> list[FlowComponent]:
        return self._body
    
    @property
    def tail(self) -> list[FlowComponent]:
        return self._tail
    
    def build(self, base) -> Flow:
        bijectors = []
        norm_flows_count = 0
        flow_name = None
        conditional = False
        for component in self.head:
            bijectors.append(component())
        for i in range(self.num_flows):
            for component in self.body:
                # skip prepended norm flows
                if norm_flows_count == 0 and not component._is_flow:
                    continue
                # count norm flows
                elif component._is_flow:
                    norm_flows_count += 1
                    flow_name = component.name
                    conditional = component.conditional
                # skip appended non-norm flows
                elif norm_flows_count == self.num_flows:
                    break
                bijectors.append(component(i))
        for component in self.tail:
            bijectors.append(component())

        if norm_flows_count == 0:
            return ValueError("No normalizing flow was included in the body of the FlowBuilder. Please include a NormalizingFlowComponent")

        chain = tfb.Chain(list(reversed(bijectors)))
        flow = Flow(base=base,
                    chain=chain,
                    num_flows=self.num_flows,
                    flow_name=flow_name,
                    conditional=conditional)
        return flow

    def __add__(self, value):
        if isinstance(value, FlowComponent):
            if value.at_tail:
                return FlowBuilder(num_flows=self.num_flows,
                                   head=self.head,
                                   body=self.body,
                                   tail=self.tail + [value])
            else:
                return FlowBuilder(num_flows=self._num_flows,
                                   head=self.head,
                                   body=self.body + [value],
                                   tail=self.tail)
        elif isinstance(value, FlowBuilder):
            return FlowBuilder(num_flows=self.num_flows + value.num_flows,
                               head=self.head,
                               body=self.body + value.body,
                               tail=self.tail)
        return NotImplemented

    def __radd__(self, value):
        if isinstance(value, FlowComponent):
            if value.at_head:
                return FlowBuilder(num_flows=self.num_flows,
                                   head=[value] + self.head,
                                   body=self.body,
                                   tail=self.tail)
            else:
                return FlowBuilder(num_flows=self._num_flows,
                                   head=self.head,
                                   body=[value] + self.body,
                                   tail=self.tail)
        elif isinstance(value, FlowBuilder):
            return FlowBuilder(num_flows=value.num_flows + self.num_flows,
                               head=self.head,
                               body=value.body + self.body,
                               tail=self.tail)
        return NotImplemented

    def __mul__(self, value):
        if isinstance(value, int):
            return FlowBuilder(num_flows=self._num_flows * value,
                               head=self.head,
                               body=self.body,
                               tail=self.tail)
        return NotImplemented

    def __rmul__(self, value):
        if isinstance(value, int):
            return FlowBuilder(num_flows=value * self._num_flows,
                               head=self.head,
                               body=self.body,
                               tail=self.tail)
        return NotImplemented

    def __imul__(self, value):
        if isinstance(value, int):
            self.num_flows *= value
        else:
            return NotImplemented


# make a MF Full, Simple, Conditional, wtv is necessary

def MADEFactory(params,
               event_shape=None,
               conditional=False,
               conditional_event_shape=None,
               conditional_input_layers='all_layers',
               hidden_units=None,
               input_order='left-to-right',
               hidden_degrees='equal',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               validate_args=False,
               name: str="made",
               **kwargs) -> Callable[..., AutoregressiveNetwork]:
    return lambda i=0: AutoregressiveNetwork(
        params,
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
        name=f"{name}_{i}",
        **kwargs)


class MAFComponent(NormalizingFlowComponent):

    def __init__(self,
                 shift_and_log_scale_fn_factory: Callable|None=None,
                 bijector_fn_factory: Callable|None=None,
                 is_constant_jacobian=False,
                 validate_args=False,
                 unroll_loop=False,
                 event_ndims=1,
                 conditional: bool=False,
                 name="maf",
                 at_head: bool=False,
                 at_tail: bool=False):
        if shift_and_log_scale_fn_factory is not None:
            f = lambda i: MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=shift_and_log_scale_fn_factory(i),
                bijector_fn=bijector_fn_factory,
                is_constant_jacobian=is_constant_jacobian,
                validate_args=validate_args,
                unroll_loop=unroll_loop,
                event_ndims=event_ndims,
                name=f"{name}_{i}"
            )
        elif bijector_fn_factory is not None:
            f = lambda i: MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=shift_and_log_scale_fn_factory,
                bijector_fn=bijector_fn_factory(i),
                is_constant_jacobian=is_constant_jacobian,
                validate_args=validate_args,
                unroll_loop=unroll_loop,
                event_ndims=event_ndims,
                name=f"{name}_{i}"
            )
        else:
            return ValueError(
                "Either `bijector_fn_factory` or "\
                "`shift_and_log_scale_fn_factory` must be specified"
            )
        super().__init__(factory=f,
                         conditional=conditional,
                         at_head=at_head,
                         at_tail=at_tail,
                         name=name)


class Permute(FlowComponent):
    
    def __init__(self,
                 permutation,
                 axis=-1,
                 validate_args: bool=False,
                 name: str="permute",
                 at_head: bool=False,
                 at_tail: bool=False):
        p = tfb.Permute(
            permutation=permutation,
            axis=axis,
            validate_args=validate_args,
            name=name
        )
        super().__init__(bijector=p,
                         at_head=at_head,
                         at_tail=at_tail)


class BatchNorm(FlowComponent):

    def __init__(self,
                 batchnorm_layer=None,
                 training=True,
                 validate_args=False,
                 name='batch_normalization',
                 at_head: bool=False,
                 at_tail: bool=False):
        f = lambda i=0: tfb.BatchNormalization(
               batchnorm_layer=batchnorm_layer,
               training=training,
               validate_args=validate_args,
               name=f"{name}_{i}"
        )
        super().__init__(factory=f,
                         at_head=at_head,
                         at_tail=at_tail)


class AffineBijector(tfb.bijector.CoordinatewiseBijectorMixin,
                     tfb.bijector.AutoCompositeTensorBijector):
    def __init__(self,
                 shift=0,
                 scale=1,
                 validate_args: bool=False,
                 name="affine"):
        
        parameters = dict(locals())
        self._shift = shift
        self._scale = scale
        self._log_scale = tf.math.log(tf.abs(scale))

        super().__init__(
            forward_min_event_ndims=0,
            is_constant_jacobian=True,
            validate_args=validate_args,
            parameters=parameters,
            name=name)

    @property
    def log_scale(self):
        """The `log_scale` term in `Y = exp(log_scale) * X`."""
        return self._log_scale

    @property
    def scale(self):
        """The `scale` term in `Y = scale * X + shift`."""
        return self._scale

    @property
    def shift(self):
        """The `shift` term in `Y = scale * X + shift`."""
        return self._shift
    
    def _is_increasing(self):
        return self.scale > 0

    def _forward(self, x):
        y = tf.identity(x)
        return y * tf.exp(self.log_scale) + self.shift

    def _inverse(self, y):
        x = tf.identity(y)
        return (x - self.shift) * tf.exp(-self.log_scale)
    
    def _forward_log_det_jacobian(self, x):
        return self.log_scale


class Whiten(FlowComponent):

    def __init__(self,
                 shift=0,
                 scale=1,
                 data=None,
                 conditional_shift=0,
                 conditional_scale=1,
                 conditional_data=None,
                 validate_args: bool=False,
                 name="whiten",
                 at_head: bool=False,
                 at_tail: bool=False):
        if data is not None:
            shift = np.mean(data, axis=-1)
            scale = np.std(data, axis=-1)
        if conditional_data is not None:
            conditional_shift = np.mean(conditional_data, axis=1)
            conditional_scale = np.std(conditional_data, axis=-1)
        w = AffineBijector(
            shift=shift,
            scale=scale,
            validate_args=validate_args,
            name=name
        )
        cond_w = AffineBijector(
            shift=conditional_shift,
            scale=conditional_scale,
            validate_args=validate_args,
            name=name
        )
        super().__init__(bijector=w,
                         at_head=at_head,
                         at_tail=at_tail)
