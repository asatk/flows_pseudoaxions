"""
Author: Anthony Atkinson
Modified: 2023.07.15

Definitions of general-purpose utility functions. These include I/O, logging,
and initialization, for example.
"""

# import argparse
from datetime import datetime, timedelta
import json
import numpy as np
import os
import shutil
import sys
import tensorflow as tf
from tensorflow import keras
from textwrap import fill as twfill
import traceback
from typing import Any

from .constants import DEFAULT_SEED

loglevel = 0

LOG_INFO = 0
LOG_DEBUG = 1
LOG_WARN = 2
LOG_ERROR = 3
LOG_FATAL = 4


MSG_BOLD = 0
MSG_QUIET = 1
MSG_LONG = 2

level_codes: list[str] = [
    "I",
    "D",
    "W",
    "E",
    "F"
]

msgs: list[str] = [
    # A bolder message (wrapping)
    (
        "========================================"
        "========================================"
        "\n{:s}\n"
        "========================================"
        "========================================"
    ),

    # A quieter message (wrapping)
    (
        "\n{:s}\n"
    ),
    
    # A longer message (no wrapping)
    (
        "\n{:s}\n"
    )
]

def init(data_dir: str="data", model_dir: str="model",
         output_dir: str="output", root_dir: str="root",
         data_path: str=None, flow_path: str=None,
         newdata: bool=False, newmodel: bool=False, newanalysis: bool=False,
         seed: int=DEFAULT_SEED):
    
    # maybe have diff idea of init. init data-gen differently from training
    # and these different from analysis. Prevents making directories pre-
    # emptively and then deleting them later if interrupted or crashed.
    
    # TODO update docstring
    """
    Initialize the training program workspace. Perform all necessary checks
    before running and analyzing a model.
    """

    # TODO make a function to repeat these same checking/deletion procedures

    if not os.path.isdir(data_dir):
        print_msg(f"Invalid directory for 'data_dir' argument: {data_dir}",
                  level=LOG_FATAL)


    if not os.path.isdir(model_dir):
        print_msg(f"Invalid directory for 'model_dir' argument: {model_dir}",
                  level=LOG_FATAL)
        

    # TODO possibly add ability to overwrite/add to dir
    if not os.path.isdir(output_dir):
        if os.path.isfile(output_dir):
            print_msg("Invalid directory for 'output_dir' argument: " +
                      f"{output_dir}", level=LOG_FATAL)
        os.mkdir(output_dir)
    elif newanalysis:
        c = " " 
        while c not in ["y", "N"]:
            c = input("would you like to replace the analysis output saved at " +
                      f"{output_dir}? [y/N]")

        if c == "N":
            print_msg(f"Not overwriting the output directory {output_dir}",
                      level=LOG_FATAL)
        
        print_msg(f"Removing the previous output directory {output_dir}",
                      level=LOG_INFO)
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    

    if not os.path.isdir(root_dir):
        print_msg(f"Invalid directory for 'root_dir' argument: {root_dir}",
                  level=LOG_FATAL)


    if data_path == "":
        data_path = str(datetime.today())

    # Making new data
    if os.path.isdir(data_path) and newdata:
        c = " " 
        while c not in ["y", "N"]:
            c = input("would you like to replace the data saved at " +
                      f"{data_path}? [y/N]")
        
        if c == "N":
            print_msg(f"Not overwriting the data directory {data_path}",
                      level=LOG_FATAL)
        
        print_msg(f"Removing the previous data at {data_path}",
                      level=LOG_INFO)
        shutil.rmtree(data_path)
        os.mkdir(data_path)


    if flow_path == "":
        flow_path = str(datetime.today())

    # Training a new model where model dir already exists
    if os.path.isdir(flow_path) and newmodel:
        c = " " 
        while c not in ["y", "N"]:
            c = input("would you like to replace the model saved at " +
                      f"{flow_path}? [y/N]")
        
        if c == "N":
            print_msg(f"Not overwriting the flow model at {flow_path}",
                      level=LOG_FATAL)
        
        print_msg(f"Removing the previous model at {flow_path}",
                      level=LOG_INFO)
        shutil.rmtree(flow_path)
        os.mkdir(flow_path)

    # Make a new model
    elif not os.path.isdir(flow_path) and newmodel:
        os.mkdir(flow_path)
    
    # Loading a model from a dir that does not exist - fatal error
    elif not os.path.isdir(flow_path) and not newmodel:
        print_msg(f"Saved model directory {flow_path} does not exist",
                level=LOG_FATAL)

    # Set the global random seed for tensorflow
    tf.random.set_seed(seed)


def save_config(path: str, params_dict: dict):
    # Save configuration (state of 'defs.py') at run-time
    config = dict([item for item in params_dict.items() if not item[0].startswith('_')])
    with open(path, "w") as config_file:
        json.dump(config, config_file, indent=4)

def print_msg(msg: str, level: int=LOG_INFO, m_type: int=MSG_BOLD, time: datetime=None) -> datetime:
    """
    Utility method for logging output. Returns a datetime object indicating
    the time of logging if no time was provided. Otherwise, the time argument
    is returned to the call. If the global logging level is higher than that
    provided, the message is not printed but a time is still returned.

    msg : str, the message to be printed.
    level : int (default=LOG_INFO), the importance of the logged message.
        Messages with a level of `LOG_FATAL` will always print the stacktrace
        and exit.
    m_type : int (default=MSG_BOLD), the format of the logged message.
    time : datetime (default=None), the relevant time for the logged message.
        If no time is provided, the current time as given by `datetime.now()`
        will be used as the log time. The time will always be printed in the
        format "%H:%M:%S.%2f"

    Returns
        The time relevant to the log message. If `time` is None, the current
        time is returned.
    """

    # TODO add an open filestream to have output to a log file
    # TODO nicely print stack (not the terrible python printout)
    # TODO perhaps specify time str-conversion format as argument?
    if time is None:
        time = datetime.now()
    
    if isinstance(time, datetime):
        timestr = time.strftime("%H:%M:%S.%f")[:-4]
    elif isinstance(time, timedelta):
        timestr = str(time)[-15:-4]
    else:
        # TODO figure something out here later
        print("AAAAAAA TIME MESSED UP")

    if level >= loglevel:
            
        if m_type in [MSG_BOLD, MSG_QUIET]:
            s = twfill("{:<s} <{:>s}> {:>s}".format(timestr,
                level_codes[level], msg), width=80)
        elif m_type is MSG_LONG:
            s = "{:s}\t<{:s}> {:s}".format(timestr, level_codes[level], msg)

        print(msgs[m_type].format(s))

    if level == LOG_FATAL:
        traceback.print_stack(file=sys.stdout)
        exit()
    
    return time


class SelectiveProgbarLogger(keras.callbacks.ProgbarLogger):
    """
    Progress bar that outputs at regularly-defined intervals.
    """

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


class Checkpointer(keras.callbacks.ModelCheckpoint):
    """
    Callback that saves keras SavedModel at regular epochs. This is mostly
    a wrapper for the keras ModelCheckpoint callback with some explicit
    defaults that cater to this architecture.
    """

    def __init__(self, filepath: str, verbose: int=1,
                 save_freq: int|str="epoch"):
        super().__init__(filepath=filepath, verbose=verbose,
                         save_freq=save_freq, save_weights_only=False)


class LossHistory(keras.callbacks.Callback):
    """
    Abstract class for callbacks that track losses of a model as it trains.
    Losses may be retrieved by accessing the class field `self.losses`, which
    is updated with the loss as a result of a training event, i.e., the end of
    a batch or an epoch.
    """
    
    def on_train_begin(self, logs={}, save_batch=False, save_epoch=True):
        self.losses = np.empty((0, 2))


class BatchLossHistory(LossHistory):
    """
    Updates the loss of a model at the end of each training batch.
    """

    def on_batch_end(self, batch, logs={}):
        self.losses = np.concatenate((self.losses, [[batch, float(logs.get('loss'))]]))


class EpochLossHistory(LossHistory):
    """
    Updates the loss of a model at the end of each training epoch.
    """

    def on_epoch_end(self, epoch, logs={}):
        self.losses = np.concatenate((self.losses, [[epoch, float(logs.get('loss'))]]))
