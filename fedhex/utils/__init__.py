"""
Author: Anthony Atkinson
Modified: 2023.07.15

Utility functions subpackage
"""


from ._callbacks import BatchLossHistory, Checkpointer, EpochLossHistory, \
    SelectiveProgbarLogger
from ._logging import *
from ._logging import get_loglevel, print_msg, set_loglevel


logging_consts = [c for c in _logging if c[:3] == "LOG"]
logging_consts.append([c for c in _logging if c[:3] == "MSG"])

__all__ = [
    "BatchLossHistory",
    "Checkpointer",
    "EpochLossHistory",
    "SelectiveProgbarLogger",
    "get_loglevel",
    "print_msg",
    "set_loglevel"
]

__all__.append(logging_consts)

del logging_consts