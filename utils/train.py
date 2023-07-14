"""
Author: Anthony Atkinson
Modified: 2023.07.14

Definitions of utility functions for training a normalizing flow.
"""

from absl import logging as absl_logging
from datetime import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from typing import Any

import defs
import flows.flowmodel as flowmodel
from . import io as ioutils
from .io import print_msg

def train(model: tf.keras.Model, data: np.ndarray, cond: np.ndarray,
          nepochs: int, batch_size: int, flow_path: str|None=None, loss_path: str|None=None) -> None:
    # TODO update docstring
    """
    Performs the full training procedure on the provided model for the given
    run configuration.

    model : tf.keras.Model, a fully-constructed flows model
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

    # hopefully get rid of those stupid tensorflow warnings
    absl_logging.set_verbosity(absl_logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    tstart = print_msg("[Training begins]")

    # Construct a list of callbacks during training
    callbacks = []
    
    # Checkpoint callback
    if flow_path is not None:
        ckptpath = flow_path + "/cp-{epoch:04d}.ckpt"
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                    filepath=ckptpath,
                    verbose=1,
                    save_weights_only=False,
                    save_freq=int(defs.epoch_save * len(data) / batch_size)))
        
    # Loss history callback
    if loss_path is not None:
        losshistory = ioutils.EpochLossHistory()
        callbacks.append(losshistory)

    # Logger callback
    callbacks.append(ioutils.SelectiveProgbarLogger(
                    verbose=1,
                    epoch_interval=10,
                    epoch_end=nepochs,
                    tstart=tstart))
    
    # Run the training procedure with all data and configuration options
    model.fit(x=[data, cond],
            y=np.zeros((data.shape[0], 0), dtype=np.float32),
            shuffle=True,
            batch_size=batch_size,
            epochs=nepochs,
            verbose=0,
            initial_epoch=0 if defs.newmodel else defs.epoch_resume,
            callbacks=callbacks)

    if flow_path is not None:
        model.save(flow_path)

    if loss_path is not None:
        np.save(loss_path, losshistory.losses)

    tend = print_msg("[Training ends]")
    print_msg("[Time elapsed]", time=(tend - tstart))

def getmodel(flow_path: str=None) -> tuple[keras.Model, Any, list[Any]]:
    """
    Retrieve the Keras SavedModel, the tfb TransformedDistribution, and the
    list of MADE blocks from the desired model. Either a new model and its
    parts or those contained in the given model directory are returned.

    flow_path, str
        Path pointing to a Keras SavedModel instance on disk.

    Returns
        keras.SavedModel : model with the specified parameters
        TFDistribution.TransformedDistribution : transformation from the
            normalizing distribution to the data (what is trained and sampled)
        list[tf.Module] : list of the MAF layers with Permute layers in between
    """
    
    # Build a model from scratch
    if defs.newmodel or flow_path is None:
        model, distribution, made_list = flowmodel.compile_MAF_model(defs.nmade, num_inputs=defs.ndim, num_cond_inputs=defs.ndim_label, hidden_layers=defs.hidden_layers, hidden_units=defs.hidden_units)

    # Load a model and extract its skeleton of MAFs
    else:
        model = keras.models.load_model(flow_path, custom_objects={"lossfn": flowmodel.lossfn, "Made": flowmodel.Made})
        made_blocks = []
        for i in range(defs.nmade):
            made_blocks.append(model.get_layer(name=f"made_{i}"))

        distribution, made_list = flowmodel.build_distribution(made_blocks, defs.ndim, num_made=defs.nmade, hidden_layers=defs.hidden_layers, hidden_units=defs.hidden_units)

    return model, distribution, made_list