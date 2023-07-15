"""
Author: Anthony Atkinson
Modified: 2023.07.15

Definitions of utility functions for training a normalizing flow.
"""


import json
import numpy as np
import os
from tensorflow.python.keras.models import load_model as kload_model, Model
from typing import Any

import flows.flowmodel as flowmodel
from . import io as ioutils
from .io import print_msg, LOG_WARN, LOG_FATAL


### --- MODE CONSTANTS --- ###
MODE_LINE = 0    #2-D gaussians along a line
MODE_GRID = 1    #2-D gaussians in a grid
MODE_ROOT = 2    #2-D distributions in a grid, usually


def train(model: Model, data: np.ndarray, cond: np.ndarray, nepochs: int,
          batch_size: int, save_freq: int|str="epoch", starting_epoch: int=0,
          flow_path: str|None=None, loss_path: str|None=None) -> None:
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
    loss_path : str or None (default=None), if provided, the training losses of
        the flow model will be saved after every epoch. If no path is provided,
        the losses will not be saved after training and all data will be lost
        once the process exits.
    """

    assert data.shape[0] == cond.shape[0]

    tstart = print_msg("[Training begins]")


    # ---- MAKE CALLBACKS ---- #
    # Construct a list of callbacks during training
    callbacks = []
    
    # Checkpoint callback
    if flow_path is not None:
        ckptpath = flow_path + "/cp-{epoch:04d}.ckpt"
        ckpt_cb = ioutils.Checkpointer(ckptpath, save_freq=save_freq)
        callbacks.append(ckpt_cb)
        
    # Loss history callback
    if loss_path is not None:
        losshistory_cb = ioutils.EpochLossHistory()
        callbacks.append(losshistory_cb)

    # Logger callback
    progbar_cb = ioutils.SelectiveProgbarLogger(
                    verbose=1,
                    epoch_interval=10,
                    epoch_end=nepochs,
                    tstart=tstart)
    callbacks.append(progbar_cb)
    # ---- MAKE CALLBACKS ---- #

    
    # Run the training procedure with all data and configuration options
    model.fit(x=[data, cond],
            y=np.zeros((data.shape[0], 0), dtype=np.float32),
            shuffle=True,
            batch_size=batch_size,
            epochs=nepochs,
            verbose=0,
            initial_epoch=starting_epoch,
            callbacks=callbacks)

    if flow_path is not None:
        model.save(flow_path)

    if loss_path is not None:
        np.save(loss_path, losshistory_cb.losses)

    tend = print_msg("[Training ends]")
    print_msg("[Time elapsed]", time=(tend - tstart))

def getmodel(flow_path: str|None=None, newmodel: bool=True) -> tuple[Model, Any, list[Any]]:
    """
    Retrieve the Keras SavedModel, the tfb TransformedDistribution, and the
    list of MADE blocks from the desired model. Either a new model and its
    parts or those contained in the given model directory are returned.

    flow_path, str or None
        Path pointing to a Keras SavedModel instance on disk. If None then
        a new model is constructed. Otherwise, a pre-built model is loaded from
        the file at `flow_path`.

    Returns
        keras.SavedModel : model with the specified parameters
        TFDistribution.TransformedDistribution : transformation from the
            normalizing distribution to the data (what is trained and sampled)
        list[tf.Module] : list of the MAF layers with Permute layers in between
    """

    config_path = flow_path + "/config.json"
    if not os.path.isfile(config_path):
        print_msg(f"The model at '{flow_path}' is missing the model config " +
                  f"file at {config_path}. A new model is going to be " +
                  f"created at '{flow_path}'.", level=LOG_WARN)
    with open(config_path, "r") as config_file:
        config_dict = json.load(config_file)

    # Retrieve configuration parameters
    nmade = config_dict["nmade"]
    ndim = config_dict["ndim"]
    ndim_label = config_dict["ndim_label"]
    hidden_layers = config_dict["hidden_layers"]
    hidden_units = config_dict["hidden_units"]
    
    # Build a model from scratch
    if newmodel:
        model, distribution, made_list = flowmodel.compile_MAF_model(
            nmade, num_inputs=ndim, num_cond_inputs=ndim_label,
            hidden_layers=hidden_layers, hidden_units=hidden_units)

    # Load a model and extract its skeleton of MAFs
    else:
        # Define model's custom objects
        custom_objects = {"lossfn": flowmodel.lossfn, "Made": flowmodel.Made}
        model = kload_model(flow_path, custom_objects=custom_objects)
        made_blocks = []
        for i in range(nmade):
            made_blocks.append(model.get_layer(name=f"made_{i}"))

        distribution, made_list = flowmodel.build_distribution(made_blocks,
            ndim, num_made=nmade, hidden_layers=hidden_layers,
            hidden_units=hidden_units)

    return model, distribution, made_list