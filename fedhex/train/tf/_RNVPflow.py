"""
Author: Anthony Atkinson
Modified: 2023.07.23

Contains the core functions for Real Non-Volume-Preserving Flows.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.api._v2.keras.models import load_model as kload_model
from keras.api._v2.keras.regularizers import l2
from tensorflow_probability import distributions as tfd


# tfb.real_nvp_default_template()


def Coupling(ninputs: int,
             nlayers: int,
             dim_layer: int,
             activation: str="relu",
             reg: float=0.01):
    
    in_layer = keras.layers.Input(shape=ninputs)

    temp_t_layer = in_layer
    temp_s_layer = in_layer
    for _ in range(nlayers - 1):
        t_layer = keras.layers.Dense(
            dim_layer, activation=activation, kernel_regularizer=l2(reg)
        )(temp_t_layer)
        s_layer = keras.layers.Dense(
            dim_layer, activation=activation, kernel_regularizer=l2(reg)
        )(temp_s_layer)

        temp_t_layer = t_layer
        temp_s_layer = s_layer

    out_t_layer = keras.layers.Dense(
        ninputs, activation="linear", kernel_regularizer=l2(reg)
    )(temp_t_layer)
    out_s_layer = keras.layers.Dense(
        ninputs, activation="tanh", kernel_regularizer=l2(reg)
    )(temp_s_layer)

    return keras.Model(inputs=in_layer, outputs=[out_s_layer, out_t_layer])


class RNVP(keras.Model):
    def __init__(self, ninputs: int, ncinputs: int, dim_coupling: int, n_coupling: int, nlayers: int):
        super().__init__()
        
        self.ninputs = ninputs
        self.ncinputs = ncinputs
        self.dim_coupling = dim_coupling
        self.n_coupling = n_coupling
        self.nlayers = nlayers
        
        self.distribution = tfd.MultivariateNormalDiag(
            loc=np.zeros(ninputs, dtype="float32"), scale_diag=np.ones(ninputs, dtype="float32"))
        self.masks = np.zeros((nlayers, ninputs), dtype="float32")
        for i in range(nlayers):
            self.masks[i,i%ninputs] = 1
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(ninputs, n_coupling, dim_coupling) for _ in range(nlayers)]

    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def call(self, x, training: bool=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.nlayers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood, axis=None)

    def train_step(self, data):
        with tf.GradientTape() as tape:

            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}