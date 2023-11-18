"""
Author: Anthony Atkinson
Modified: 2023.07.23

Contains the core functions for Real Non-Volume-Preserving Flows.
"""

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.keras.models import load_model as kload_model, Model
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import TransformedDistribution
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import MaskedAutoregressiveFlow as MAF

class RNVP():
    pass

    tfb.real_nvp_default_template()