"""
Author: Anthony Atkinson
Modified: 2024.06.07

Flows-Enriched Data generation for High-energy EXperiment.
"""


__author__ = "Anthony Atkinson"
__version__ = "0.1.0"


from . import constants
from . import flows
from . import utils

# tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(seed)


__all__ = [
    "constants",
    "flows",
    "utils"
]
