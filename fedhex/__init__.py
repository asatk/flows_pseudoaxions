"""
Author: Anthony Atkinson
Modified: 2024.06.07

Flows-Enriched Data generation for High-energy EXperiment.
"""


__author__ = "Anthony Atkinson"
__version__ = "0.1.0"


from ._generators import CircleGaussGenerator, LineGaussGenerator, GridGaussGenerator
from ._loaders import Loader, NumpyLoader, RootLoader
from ._managers import DataManager, ModelManager
from ._modelmanagers import MADEManager

# tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(seed)


__all__ = [
    "CircleGaussGenerator",
    "LineGaussGenerator",
    "GridGaussGenerator",
    "Loader",
    "NumpyLoader",
    "RootLoader",
    "DataManager",
    "ModelManager",
    "MADEManager",
    "constants",
    "io",
    "posttrain",
    "pretrain",
    "train",
    "utils"
]
