import numpy as np
import tensorflow as tf

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

class MultivariateStandardNormal(tfd.Sample):

    def __init__(self, sample_shape: int):
        super().__init__(
            tfd.Normal(loc=0.0, scale=1.0),
            sample_shape=sample_shape
        )
