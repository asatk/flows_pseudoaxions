"""
Author: Anthony Atkinson
Modified: 2023.07.23

Data generation
"""


import numpy as np
from .. import whiten
from ... import DEFAULT_SEED
from ...io import save_config
from ...utils import LOG_FATAL, print_msg

from ._gaussian import CovStrategy, CovModStrategy, DiagCov, FullCov, SampleCov, RepeatStrategy

class Generator():

    def __init__(self, ndist: tuple[int], nsamp: int=1e3, seed: int=DEFAULT_SEED, config: dict|None=None):
        if config is None:
            self.nsamp = nsamp
            self.ndist = ndist
            self.seed = seed
            self._rng = np.random.default_rng(seed)

            self.is_gen = False
            self._samples = np.zeros(shape=(nsamp, 2))
            self._labels = np.zeros(shape=(nsamp, 2))
            self._whiten_data = dict()
            self._whiten_cond = dict()
        else:
            self.nsamp = config.get("nsamp", 1e3)
            self.ndist = config.get("ndist", 10)
            self.seed = config.get("seed", DEFAULT_SEED)

    def preproc(self, save_path: str=None, ret_whiten: bool=False):
        data, whiten_data = whiten(self._samples)
        cond, whiten_cond = whiten(self._labels)

        self._whiten_data = whiten_data
        self._whiten_cond = whiten_cond
        self._data = data
        self._cond = cond

        if save_path is not None:
            data_dict = {"data": data, "cond": cond, "whiten_data": whiten_data, "whiten_cond": whiten_cond}
            np.save(save_path, data_dict, allow_pickle=True)

        if ret_whiten:
            return data, cond, whiten_data, whiten_cond
        return data, cond

    def save(self, config_path: str) -> True:

        # name = self.__class__.__name__
        # d = {k: v for k, v in self.__dict__.items() if name not in k}
        d = {k: v for k, v in self.__dict__.items() if k[0] != "_"}
        
        return save_config(config_path, d)


from ._gaussian import LineGaussGenerator, GridGaussGenerator


__all__ = [
    "CovStrategy",
    "CovModStrategy",
    "DiagCov",
    "FullCov",
    "SampleCov",
    "RepeatStrategy",
    "LineGaussGenerator",
    "GridGaussGenerator"
]