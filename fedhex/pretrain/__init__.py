"""
Author: Anthony Atkinson
Modified: 2023.07.20

Provides all necessary functionality setting up training.
"""


from ._data import dewhiten, whiten

from . import generation


__all__ = [
    "dewhiten",
    "whiten"
]