'''
Definitions of utility functions for training a normalizing flow.

Author: Anthony Atkinson
'''
from absl import logging as absl_logging
from datetime import datetime
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from typing import Any

import defs
import flowmodel
import utils as myutils

# Train a model
def train(model: tf.keras.Model, samples: np.ndarray, labels: np.ndarray, flowpath: str) -> None:

    absl_logging.set_verbosity(absl_logging.ERROR)
    tstart = datetime.now()
    print("training begins: ", tstart)

    ckptpath = flowpath + "/cp-{epoch:04d}.ckpt"
    # as this is once again a unsupervised task, the target vector y is zeros
    model.fit(x=[samples, labels],
            y=np.zeros((samples.shape[0], 0), dtype=np.float32),
            shuffle=True,
            batch_size=defs.batch_size,
            epochs=defs.nepochs,
            verbose=0,
            initial_epoch=0 if defs.newmodel else defs.epoch_resume,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath=ckptpath,
                    verbose=1,
                    save_weights_only=False,
                    save_freq=int(defs.epoch_save * len(samples) / defs.batch_size)),
                myutils.SelectiveProgbarLogger(
                    verbose=1,
                    epoch_interval=10,
                    epoch_end=defs.nepochs)])

    model.save(flowpath)

    tend = datetime.now()
    print("training ends: ", tend, "\ntime elapsed: ", tend - tstart)

def getmodel(flow_path: str=None) -> tuple[keras.Model, Any, list[Any]]:
    """
    Retrieve the Keras SavedModel, the tfb TransformedDistribution, and the
    list of MADE blocks from the desired model. Either a new model and its
    parts or those contained in the given model directory are returned.

    flow_path, str
        Path pointing to a Keras SavedModel instance on disk.
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
        
        distribution, made_list = flowmodel.build_distribution(made_blocks, defs.ndim, hidden_layers=defs.hidden_layers, hidden_units=defs.hidden_units)

    return model, distribution, made_list