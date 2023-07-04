'''
Definitions of utility functions for training normalizing flows.

Author: Anthony Atkinson
'''
# import argparse
from datetime import datetime
import json
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow import keras

import defs

def init():
    '''
    Initialize the training program. Perform all necessary checks before
    running and analyzing a model.
    '''

    if not os.path.isdir(defs.data_dir):
        print("Invalid 'data_dir' directory: %s"%defs.data_dir)
        exit()

    if not os.path.isdir(defs.model_dir):
        print("Invalid 'model_dir' directory: %s"%defs.model_dir)
        exit()
    
    if not os.path.isdir(defs.output_dir):
        if os.path.isfile(defs.output_dir):
            print("Invalid 'output_dir' directory: %s"%defs.output_dir)
            exit()        
        os.mkdir(defs.output_dir)
    elif defs.newanalysis:
        c = " " 
        while c not in ["y", "N"]:
            c = input("would you like to replace the analysis output saved at %s? [y/N]"%defs.output_dir)

        if c == "N":
            exit()
        
        print("Removing the previous output at %s..."%defs.output_dir)
        shutil.rmtree(defs.output_dir)
        os.mkdir(defs.output_dir)
    
    if not os.path.isdir(defs.root_dir):
        print("Invalid 'root_dir' directory: %s"%defs.root_dir)
        exit()

    flow_path = defs.flow_path

    # Training a new model where model dir already exists
    if os.path.isdir(flow_path) and defs.newmodel:
        c = " " 
        while c not in ["y", "N"]:
            c = input("would you like to replace the model saved at %s? [y/N]"%flow_path)
        
        if c == "N":
            exit()
        
        print("Removing the previous model at %s..."%flow_path)
        shutil.rmtree(flow_path)

    # Loading a model from a dir that does not exist
    elif not (os.path.isdir(flow_path) or defs.newmodel):
            print("saved model directory does not exist")
            exit()

    # Create directory for the new model
    if defs.newmodel:
        os.mkdir(defs.flow_path)    

    # Save configuration (state of 'defs.py') at run-time
    config = dict([item for item in defs.__dict__.items() if not item[0].startswith('_')])
    with open(flow_path + "/config.json", "w") as config_file:
        json.dump(config, config_file, indent=4)

    # Set the global random seed for tensorflow
    tf.random.set_seed(defs.seed)

class SelectiveProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    '''
    Progress bar that outputs at regularly-defined intervals.
    '''

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

class LossHistory(keras.callbacks.Callback):
    '''
    Abstract class for callbacks that track losses of a model as it trains.
    Losses may be retrieved by accessing the class field `self.losses`, which
    is updated with the loss as a result of a training event, i.e., the end of
    a batch or an epoch.
    '''
    
    def on_train_begin(self, logs={}, save_batch=False, save_epoch=True):
        self.losses = np.empty((0, 2))


class BatchLossHistory(LossHistory):
    '''
    Updates the loss of a model at the end of each training batch.
    '''

    def on_batch_end(self, batch, logs={}):
        self.losses = np.concatenate((self.losses, [[batch, float(logs.get('loss'))]]))


class EpochLossHistory(LossHistory):
    '''
    Updates the loss of a model at the end of each training epoch.
    '''

    def on_epoch_end(self, epoch, logs={}):
        self.losses = np.concatenate((self.losses, [[epoch, float(logs.get('loss'))]]))
