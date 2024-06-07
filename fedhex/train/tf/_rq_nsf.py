import tensorflow as tf
from tensorflow.python import keras
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability import distributions

class RQNSF(tfb.Bijectors):
    def __init__(self, validate_args=False, name="RQ_NSF"):
        super(RQNSF, self)
    
    def _forward(self, x):
        pass

    def _inverse(self, x):
        pass

    def forward_log_det_jacobian(self, x):
        pass