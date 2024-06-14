from tensorflow import keras
import tensorflow as tf

@keras.saving.register_keras_serializable(name="nll_loss")
class NLL(keras.losses.Loss):

    def __init__(self, name="nll_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, x, y):
        return tf.negative(y)
