from tensorflow import keras
import tensorflow as tf

@keras.saving.register_keras_serializable(name="nll_loss")
class NLL(keras.losses.Loss):

    def __init__(self, reduction="none", name="nll_loss", **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)

    def call(self, x, y, axis=-1):
        return tf.negative(tf.reduce_mean(y, axis=axis))
