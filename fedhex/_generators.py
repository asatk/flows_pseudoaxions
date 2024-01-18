import abc
import numpy as np
from numpy import ndarray
from typing import Self

from .constants import DEFAULT_SEED, WHITEN_EPSILON
from .pretrain.generation import CovModStrategy, NoneMod, sample_gaussian
from .utils import LOG_ERROR, print_msg

from ._managers import DataManager

# TODO move nsamp to gen method, not instantiation


class Generator(DataManager, metaclass=abc.ABCMeta):

    def __init__(self, ndist: tuple[int], seed: int=DEFAULT_SEED, config: dict|None=None):
        super().__init__()
        if config is None:
            self.ndist = ndist
            self.seed = seed
            self._rng = np.random.default_rng(seed)
        else:
            self.ndist = config.get("ndist", 10)
            self.seed = config.get("seed", DEFAULT_SEED)

        self.has_generated = False

    @abc.abstractmethod
    def generate(self, nsamp: int=1e3):
        ...

    # @property
    # def has_generated(self):
    #     return self._has_generated

    def preproc(self, epsilon: float=WHITEN_EPSILON) -> tuple[ndarray, ndarray]:
        if not self.has_generated:
            print_msg("Data have not yet been generated. Run `generate()` " + \
                      "this `Generator` instance before running `preproc()`.",
                      level=LOG_ERROR)
            return tuple()
        return super().preproc(epsilon=epsilon)
    
    def recover(self) -> tuple[ndarray, ndarray]:
        if not self.has_generated:
            print_msg("Data have not yet been generated. Run `generate()` " + \
                      "this `Generator` instance before running `recover()`.",
                      level=LOG_ERROR)
            return tuple()
        return super().recover()


class GaussGenerator(Generator, metaclass=abc.ABCMeta):

    def __init__(self, ngaus: int|tuple, cov_strat: CovModStrategy|None=None, seed: int=DEFAULT_SEED):
        super().__init__(ndist=ngaus, seed=seed)
        self.cov_strat = NoneMod if cov_strat is None else cov_strat


class CircleGaussGenerator(GaussGenerator):

    def __init__(self, cov_strat: CovModStrategy=None, ngaus: int=10,
                 rad: float=1.0, seed: int=DEFAULT_SEED, lims: tuple=(0.0, 360.0)):
        super().__init__(ngaus=(ngaus, ), cov_strat=cov_strat, seed=seed)
        self.rad = rad
        self.lims = lims
    
    def generate(self, nsamp: int=1e3) -> tuple[ndarray]:

        thetamin, thetamax = self.lims
        ngaus = self.ndist[0]
        radius = self.rad

        arc = np.array([np.linspace(thetamin, thetamax, ngaus, endpoint=False)]).T
        arc_rad = arc / 180. * np.pi

        cond = np.multiply([np.sin(arc_rad), np.cos(arc_rad)], radius).T
        covs = self.cov_strat.create(cond)
        samples, labels = sample_gaussian(nsamples=nsamp, labels_unique=arc, means=cond, covs=covs, rng=self._rng)

        self._samples = samples
        self._labels = labels
        self.has_generated = True
        self.has_original = True

        return samples, labels


class LineGaussGenerator(GaussGenerator):

    def __init__(self, cov_strat: CovModStrategy=None,
                 ngaus: int=10, val: float=0.5, seed: int=DEFAULT_SEED, lims: tuple=(0.0, 1.0)):
        super().__init__(ngaus=(ngaus, ), cov_strat=cov_strat, seed=seed)
        self.val = val
        self.lims = lims

    def generate(self, nsamp: int=1e3) -> tuple[ndarray]:

        xmin, xmax = self.lims
        ngaus = self.ndist[0]

        xax = np.linspace(xmin, xmax, ngaus + 1, endpoint=False)[:-1]
        cond = np.array([xax + (xmax - xmin) / (ngaus + 1)]).T

        centers = np.concatenate((cond, np.array([np.repeat(self.val, len(cond))]).T), axis=1)
        covs = self.cov_strat.create(cond)
        samples, labels = sample_gaussian(nsamples=nsamp, labels_unique=cond, means=centers, covs=covs, rng=self._rng)
        
        self._samples = samples
        self._labels = labels
        self.has_generated = True
        self.has_original = True

        return samples, labels


class GridGaussGenerator(GaussGenerator):

    def __init__(self, cov_strat: CovModStrategy=None,
                 ngausx: int=10, ngausy: int=10, seed: int=DEFAULT_SEED, lims: tuple=((0.0, 1.0), (0.0, 1.0))):
        super().__init__(ngaus=(ngausx, ngausy), seed=seed)
        self.cov_strat = cov_strat
        self.lims = lims

    def generate(self, nsamp: int=1e3) -> tuple[ndarray]:

        lims = self.lims
        xmin, xmax = lims[0]
        ymin, ymax = lims[1]
        ngausx = self.ndist[0]
        ngausy = self.ndist[1]
        
        xax = np.linspace(xmin, xmax, ngausx + 1, endpoint=False)[:-1] + (xmax - xmin) / (ngausx + 1)
        yax = np.linspace(ymin, ymax, ngausy + 1, endpoint=False)[:-1] + (ymax - ymin) / (ngausy + 1)
        x, y = np.meshgrid(xax, yax)
        cond = np.array([x.ravel(), y.ravel()]).T

        covs = self.cov_strat.create(cond)
        samples, labels = sample_gaussian(nsamples=nsamp, labels_unique=cond, means=cond, covs=covs, rng=self._rng)
        
        self._samples = samples
        self._labels = labels
        self.has_generated = True
        self.has_original = True

        return samples, labels
    
    @classmethod
    def import_cfg(cls: Self, config_path: str):
        print(f"{str(cls)}:{config_path}")
