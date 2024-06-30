"""
Author: Anthony Atkinson
Modified: 2024.06.07

Flows-Enriched Data generation for High-energy EXperiment.
"""


__author__ = "Anthony Atkinson"
__version__ = "0.1.0"


from ._generators import CircleGaussGenerator, LineGaussGenerator, GridGaussGenerator, UnitCubeGaussGenerator
from ._loaders import Loader, NumpyLoader, RootLoader
from ._managers import DataManager, ModelManager
from ._modelmanagers import MAFManager

from . import train
from . import io
from . import utils
from . import pretrain
from . import posttrain

# tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(seed)


__all__ = [
    "CircleGaussGenerator",
    "LineGaussGenerator",
    "GridGaussGenerator",
    "UnitCubeGaussGenerator",
    "Loader",
    "NumpyLoader",
    "RootLoader",
    "DataManager",
    "ModelManager",
    "MAFManager",
    "constants",
    "io",
    "posttrain",
    "pretrain",
    "train",
    "utils"
]
