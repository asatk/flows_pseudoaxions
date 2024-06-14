"""
Author: Anthony Atkinson
Modified: 2023.07.15

Train a TF MADE normalizing flow model.
"""


import numpy as np
from tensorflow import keras

from ...io._path import print_msg, LOG_FATAL


# possibly place this in init or not
def train(model: keras.Model,
          data: np.ndarray,
          cond: np.ndarray=None,
          batch_size: int=64,
          initial_epoch: int=0,
          epochs: int=1,
          validation_split: float=0.0,
          flow_path: str|None=None,
          callbacks: list[keras.callbacks.Callback]=None) -> None:
    """Performs the full training procedure on the provided model for the given
    run configuration.

    Args:
        model (keras.Model): compiled flows model
        data (np.ndarray): training data
        cond (np.ndarray): conditional data. Defaults to None.
        batch_size (int): size of the sample from the training data used in
            each training step. Defaults to 64.
        initial_epoch (int, optional): first epoch of training. Defaults to 0.
        epochs (int, optional): final epoch of training. Defaults to 1.
        flow_path (str | None, optional): _description_. Defaults to None.
        callbacks (list[keras.callbacks.Callback], optional): Collection of
            callbacks used during training. Defaults to None.
    """

    if data.shape[0] != cond.shape[0]:
        print_msg("Data and Conditional Data are not the same lengths: " + \
                  f"{data.shape[0]} and {cond.shape[0]}.", level=LOG_FATAL)

    tstart = print_msg("[Training begins]")
    
    # Run the training procedure with all data and configuration options
    model.fit(x=[data, cond],
              y=np.zeros((data.shape[0], 0)),
              shuffle=True,
              batch_size=batch_size,
              initial_epoch=initial_epoch,
              epochs=epochs,
              validation_split=validation_split,
              verbose=0,
              callbacks=callbacks)

    if flow_path is not None:
        model.save(flow_path)

    tend = print_msg("[Training ends]")
    print_msg("[Time elapsed]", time=(tend - tstart))
