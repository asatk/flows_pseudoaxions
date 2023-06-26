'''
Definitions of utility functions for training normalizing flows.

Author: Anthony Atkinson
'''
# import argparse
from datetime import datetime
import os
import tensorflow as tf

import defs

def init():

    model_dir = defs.model_dir
    flow_path = defs.flow_path

    if not (os.path.isdir(defs.data_dir) and os.path.isdir(model_dir) \
            and os.path.isdir(defs.output_dir) and os.path.isdir(defs.root_dir)):
        print("invalid directories")
        exit()
    
    # run_num = 0
    # flowname = datetime.today().date()
    # flowpath = "/".join([model_dir,'flow_%s_run%02i'%(flowname, run_num)])

    # Training a new model where model dir already exists
    if os.path.isdir(flow_path) and defs.newmodel:
        c = " "
        while c not in ['y', 'N']:
            c = input("would you like to replace the model saved at %s? [y/N]"%(flow_path))
        
        if c == 'N':
            exit()
        else:
            print("Removing the previous model at %s...")
            os.rmdir(flow_path)

    # Loading a model from a dir that does not exist
    elif not (os.path.isdir(flow_path) or defs.newmodel):
            print("saved model directory does not exist")
            exit()

# Progress bar which outputs at certain intervals only
class SelectiveProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, verbose, epoch_interval, epoch_end, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = epoch_interval
        self.epoch_end = epoch_end
    
    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.verbose = (
                0 if (epoch + 1) % self.epoch_interval != 0 and (epoch + 1) != self.epoch_end
                else self.default_verbose)
        if (epoch + 1) % self.epoch_interval == 0:
            print("epoch begin: ", datetime.now())
        super().on_epoch_begin(epoch, *args, **kwargs)