"""
Author: Anthony Atkinson
Modified: 2023.07.14

Flows-Enriched Data generation for High-energy EXperiment.
"""

__author__ = "Anthony Atkinson"
__version__ = "0.1.0"


import tensorflow as tf

from .train import *
from .posttrain import *
from .pretrain import *
from .train import *
from .utils import *
from .constants import *


def set_seed(seed: int=constants.DEFAULT_SEED):
    # Set the global random seed for tensorflow
    tf.random.set_seed(seed)

set_seed()

__all__ = [
    "set_seed"
]

