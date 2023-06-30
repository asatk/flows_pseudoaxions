'''
Definitions of utility functions for training normalizing flows.

Author: Anthony Atkinson
'''
# import argparse
from datetime import datetime
import json
import os
import shutil
import tensorflow as tf

import defs

def init():

    if not os.path.isdir(defs.data_dir):
        print("Invalid 'data_dir' directory: %s"%defs.data_dir)
        exit()

    if not os.path.isdir(defs.model_dir):
        print("Invalid 'model_dir' directory: %s"%defs.model_dir)
        exit()
    
    if not os.path.isdir(defs.output_dir):
        print("Invalid 'output_dir' directory: %s"%defs.output_dir)
        exit()
    
    if not os.path.isdir(defs.root_dir):
        print("Invalid 'root_dir' directory: %s"%defs.root_dir)
        exit()

    flow_path = defs.flow_path

    # Training a new model where model dir already exists
    if os.path.isdir(flow_path) and defs.newmodel:
        c = " "
        while c not in ['y', 'N']:
            c = input("would you like to replace the model saved at %s? [y/N]"%(flow_path))
        
        if c == 'N':
            exit()
        else:
            print("Removing the previous model at %s...")
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

# Progress bar which outputs at certain intervals only
class SelectiveProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, verbose, epoch_interval, epoch_end, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = epoch_interval
        self.epoch_end = epoch_end
    
    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.verbose = (
                0 if (epoch + 1) % self.epoch_interval != 0 and \
                     (epoch + 1) != self.epoch_end
                else self.default_verbose)
        if (epoch + 1) % self.epoch_interval == 0:
            print("epoch begin: ", datetime.now())
        super().on_epoch_begin(epoch, *args, **kwargs)