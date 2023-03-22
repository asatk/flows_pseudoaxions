import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow.test as tft
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
import numpy as np

# This is simply a copy of the original "AutoregressiveNetwork" class in tfp.bijectors. The only reason we need to do this is that we want to apply a tanh
# on the output log-scale when the Network is called. This allows for better regularization and helps with "inf" and "nan" values that otherwise would
# frequently occur during training.
class Made(tfb.AutoregressiveNetwork):
    def __init__(self, params, event_shape=None, conditional=False, conditional_event_shape=None, conditional_input_layers='all_layers', hidden_units=None,
                 input_order='left-to-right', hidden_degrees='equal', activation=None, use_bias=True,kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None, validate_args=False, **kwargs):
        
        super().__init__(params=params, event_shape=event_shape, conditional=conditional, conditional_event_shape=conditional_event_shape,
                         conditional_input_layers=conditional_input_layers, hidden_units=hidden_units, input_order=input_order, hidden_degrees=hidden_degrees,
                         activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                         validate_args=validate_args, **kwargs)
    
    def call(self, x, conditional_input=None):
        

        result = super().call(x, conditional_input=conditional_input)
        
        shift, log_scale = tf.unstack(result, num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)
    
    def get_config(self):
        
        config = super().get_config().copy()
        
        return config

def compile_MAF_model(num_made, num_inputs, num_cond_inputs=None, made_layers=[128], base_lr=1.0e-3, end_lr=1.0e-4, return_layer_list=False):

    if num_cond_inputs is not None:
        conditional = True
        cond_event_shape = (num_cond_inputs,)
    else:
        conditional = False
        cond_event_shape = None

    made_list = []
    for i in range(num_made):
        made_list.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=Made(params=2, hidden_units=made_layers, event_shape=(num_inputs,), conditional=conditional,
                                        conditional_event_shape=cond_event_shape, activation='relu', name=f"made_{i}"), name=f"maf_{i}"))
    
        #made_list.append(tfb.BatchNormalization())
        made_list.append(tfb.Permute(permutation=np.array(np.arange(0, num_inputs)[::-1])))
                     
    # remove final permute layer
    made_chain = tfb.Chain(list(reversed(made_list[:-1])))

    # we want to transform to gaussian distribution with mean 0 and std 1 in latent space
    distribution = tfd.TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[num_inputs]),
        bijector=made_chain)

    x_ = tfk.layers.Input(shape=(num_inputs,), name="aux_input")
    input_list = [x_]

    if conditional:
        c_ = tfk.layers.Input(shape=(num_cond_inputs,), name="cond_input")
        input_list.append(c_)

        current_kwargs = {}
        for i in range(num_made):
            current_kwargs[f"maf_{i}"] = {'conditional_input' : c_}
    
    else:
        current_kwargs = {}
  
    log_prob_ = distribution.log_prob(x_, bijector_kwargs=current_kwargs)
  
    model = tfk.Model(input_list, log_prob_)
    max_epochs = 100  # maximum number of epochs of the training
    learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(base_lr, max_epochs, end_lr, power=0.5)
    model.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate_fn),
                loss=lambda _, log_prob: -log_prob)
  
    if return_layer_list:
        return model, distribution, made_list
    else:
        return model, distribution

def intermediate_flows_chain(made_list):
    # reverse the list of made blocks to unpack in generating direction
    made_list_rev = list(reversed(made_list[:-1]))

    feat_extraction_dists = []

    made_chain = tfb.Chain([])
    dist = tfd.TransformedDistribution(
        distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[2]),
        bijector=made_chain)
    feat_extraction_dists.append(dist)

    for i in range(1, len(made_list_rev), 2):
        made_chain = tfb.Chain(made_list_rev[0:i])
        dist = tfd.TransformedDistribution(
            distribution=tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[2]),
            bijector=made_chain)
        feat_extraction_dists.append(dist)

    return feat_extraction_dists