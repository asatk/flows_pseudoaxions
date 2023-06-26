'''
Definitions of utility functions for training a normalizing flow.

Author: Anthony Atkinson
'''

from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

import defs
import flowmodel
import utils as myutils

# Train a model
def train(model: tf.keras.Model, samples: np.ndarray, labels: np.ndarray, flowpath: str):

    tstart = datetime.now()
    print("training begins: ", tstart)

    ###------TRAIN MODEL------###
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

def getmodel(flowpath):
    
    # Build a model from scratch
    if defs.newmodel:
        model, distribution, made_list = flowmodel.compile_MAF_model(defs.nmade, num_inputs=defs.ndim, num_cond_inputs=defs.ndim_label)

    # Load a model and extract its skeleton of MAFs
    else:
        model = keras.models.load_model(flowpath, custom_objects={"lossfn": flowmodel.lossfn, "Made": flowmodel.Made})
        made_blocks = []
        for i in range(defs.nmade):
            made_blocks.append(model.get_layer(name=f"made_{i}"))
        
        distribution, made_list = flowmodel.build_distribution(made_blocks, defs.ndim)

    return model, distribution, made_list