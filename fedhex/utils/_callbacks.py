"""
Author: Anthony Atkinson
Modified: 2023.07.20

Callback definitions
"""


from datetime import datetime
import numpy as np
from tensorflow.python.keras import callbacks as cb


class LossHistory(cb.Callback):
    """
    Abstract class for callbacks that track losses of a model as it trains.
    Losses may be retrieved by accessing the class field `self.losses`, which
    is updated with the loss as a result of a training event, i.e., the end of
    a batch or an epoch.
    """

    def __init__(self, loss_path: str):
        self.path = loss_path
    
    def on_train_begin(self, logs={}, save_batch=False, save_epoch=True):
        self.losses = np.empty((0, 2))

    def on_train_end(self, logs=None):
        np.save(self.path, self.losses)


class BatchLossHistory(LossHistory):
    """
    Updates the loss of a model at the end of each training batch.
    """

    def on_batch_end(self, batch, logs={}):
        self.losses = np.concatenate((self.losses, [[batch, float(logs.get('loss'))]]))
        

class Checkpointer(cb.ModelCheckpoint):
    """
    Callback that saves keras SavedModel at regular epochs. This is mostly
    a wrapper for the keras ModelCheckpoint callback with some explicit
    defaults that cater to this architecture.
    """

    def __init__(self, filepath: str, verbose: int=1,
                 save_freq: int|str="epoch"):
        super().__init__(filepath=filepath, verbose=verbose,
                         save_freq=save_freq, save_weights_only=False)


class EpochLossHistory(LossHistory):
    """
    Updates the loss of a model at the end of each training epoch.
    """

    def on_epoch_end(self, epoch, logs={}):
        self.losses = np.concatenate((self.losses, [[epoch, float(logs.get('loss'))]]))


# TODO log the first epoch for a time baseline
class SelectiveProgbarLogger(cb.ProgbarLogger):
    """
    Progress bar that outputs at regularly-defined intervals.
    """

    def __init__(self, verbose, epoch_interval, epoch_end, tstart, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = epoch_interval
        self.epoch_end = epoch_end
        self.tsart = tstart
    
    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.verbose = (
                0 if (epoch + 1) % self.epoch_interval != 0 and \
                     (epoch + 1) != self.epoch_end
                else self.default_verbose)
        if self.verbose:
            tnow = datetime.now()
            dt = tnow - self.tsart
            print("epoch begin: " + str(tnow) + " | time elapsed: " + str(dt))
        super().on_epoch_begin(epoch, *args, **kwargs)





