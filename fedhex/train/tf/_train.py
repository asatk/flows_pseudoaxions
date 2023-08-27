"""
Author: Anthony Atkinson
Modified: 2023.07.15

Train a TF MADE normalizing flow model.
"""


import numpy as np
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import Callback

from ...io._path import print_msg


# possibly place this in init or not
def train(model: Model, data: np.ndarray, cond: np.ndarray, batch_size: int,
          starting_epoch: int=0, end_epoch: int=1, flow_path: str|None=None,
          callbacks: list[Callback]=None) -> None:
    

    # TODO update docstring
    """
    Performs the full training procedure on the provided model for the given
    run configuration.

    model : keras.Model, a fully-constructed flows model
    data : np.ndarray, the network-ready training data used by the model. Any
        pre-processing on these data must already be done.
    cond : np.ndarray, the network-ready conditional regression labels used by
        the model to condition the provided data. The data and conditional data
        vectors must correspond element-wise; that is, the i-th data entry must
        be conditioned by the i-th regression label. Any pre-processing on
        these conditional data must already be done.
    flow_path : str or None (default=None), if provided, the flow model will be
        saved at regular intervals (checkpoints) and at the end of training. If
        no path is provided, the model will not be saved after training and all
        data will be lost once the process exits.
    """

    assert data.shape[0] == cond.shape[0]

    tstart = print_msg("[Training begins]")
    
    # Run the training procedure with all data and configuration options
    model.fit(x=[data, cond],
            y=np.zeros((data.shape[0], 0), dtype=np.float32),
            shuffle=True,
            batch_size=batch_size,
            epochs=end_epoch,
            verbose=0,
            initial_epoch=starting_epoch,
            callbacks=callbacks)

    if flow_path is not None:
        model.save(flow_path)

    tend = print_msg("[Training ends]")
    print_msg("[Time elapsed]", time=(tend - tstart))
