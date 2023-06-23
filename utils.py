'''
Definitions of utility functions for training CCGAN.

Author: Anthony Atkinson
'''

from datetime import datetime
# import numpy as np
import os
import tensorflow as tf

# import defs as defs

def namefile(dirname: str, name: str, isdir=False, ext: str=None):

    if isdir or ext is None:
        ext = ""

    d = os.listdir(dirname)
    i = 0
    
    f = "%s_run%02i%s"%(name, i, ext)
    
    while f in d:
        f = "%s_run%02i%s"%(name, i, ext)
        i += 1

    f = dirname + f

    if isdir:
        f += '/'

    return f
    
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
        if (epoch + 1) % self.epoch_interval != 0:
            print("epoch begin: ", datetime.now())
        super().on_epoch_begin(epoch, *args, **kwargs)