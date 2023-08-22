"""
Author: Anthony Atkinson
Modified: 2023.07.14

Flows-Enriched Data generation for High-energy EXperiment.
"""

__author__ = "Anthony Atkinson"
__version__ = "0.1.0"


import abc
import tensorflow as tf

from .constants import *
from .io import *
from .posttrain import *
from .pretrain import *
from .train import *
from .utils import *

from .io import save_config


class DataManager(metaclass=abc.ABCMeta):

    def __str__(self):
        return f"<<{self.__class__.__name__}>>"
    
    def get_data(self):
        return self._data
    
    def get_cond(self):
        return self._cond
    
    def get_whiten_data(self):
        return self._whiten_data
    
    def get_whiten_cond(self):
        return self._whiten_cond

    def save(self, config_path: str) -> True:

        prefix = f"_{self.__class__.__name__}"
        len_prefix = len(prefix)
        
        d = {k: v for k, v in self.__dict__.items() if k[0] != "_"}
        d.update({k: v for k, v in self.__dict__.items() if k[len_prefix] == prefix})
        
        return save_config(config_path, d)
    
    # TODO save config, preproc, recover


def set_seed(seed: int=constants.DEFAULT_SEED):
    # Set the global random seed for tensorflow
    tf.random.set_seed(seed)

set_seed()

__all__ = [
    "DataManager",
    "set_seed"
]

