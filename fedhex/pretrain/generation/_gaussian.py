"""
Author: Anthony Atkinson
Modified: 2023.07.21

Generating gaussian data.
"""


import abc
import numpy as np
from numpy import ndarray


class CovStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create(self) -> ndarray:
        pass


class CovModStrategy(metaclass=abc.ABCMeta):

    def __init__(self, s: CovStrategy):
        self.s = s

    @abc.abstractmethod
    def create(self, xy: tuple|ndarray) -> ndarray:
        pass


class DiagCov(CovStrategy):
    def __init__(self, ndim: int, sigma: float|np.ndarray):
        self.cov = self.__create(ndim=ndim, sigma=sigma)

    def __create(self, ndim, sigma) -> ndarray:
        cov = np.zeros((ndim, ndim))
        d = np.diag_indices(ndim)
        if isinstance(sigma, float):
            sigma = sigma * np.ones(ndim)
        elif isinstance(sigma, np.ndarray) and sigma.shape[0] != ndim:
            return None
        elif not isinstance(sigma, np.ndarray):
            return None
            
        cov[d] = np.square(sigma)
        return cov

    def create(self) -> ndarray:
        return self.cov


class FullCov(CovStrategy):
    def __init__(self, sigmas: ndarray, corrs: ndarray|None=None):
        """
        sigmas : ndarray
            list of variances
        corrs : ndarray
            list of covariances listed in order of diagonals of the covariance
            matrix from 1 to `ndim = len(sigmas)`. The matrix must be symmetric
            so the correlations on diagonals 1 to ndim are the same as those
            from -1 to -ndim.
        """
        self.cov = self.__create(sigmas=sigmas, corrs=corrs)

    def __create(self, sigmas, corrs) -> ndarray:

        ndim = len(sigmas)

        cov = np.diag(sigmas, k=0)
        
        if corrs is not None:
            len_corrs = ndim * (ndim - 1) / 2
            assert len_corrs == len(corrs)

            start_idx = 0
            for i in range(1, ndim):
                end_idx = start_idx + ndim - i
                vals = corrs[start_idx:end_idx]
                cov += np.diag(vals, k=i)
                cov += np.diag(vals, k=-i)
                start_idx = end_idx
        
        return cov

    def create(self) -> ndarray:
        return self.cov


class SampleCov(CovStrategy):
    def __init__(self, samples: ndarray):
        self.cov = self.__create(samples)

    def __create(self, samples: ndarray) -> ndarray:
        return np.cov(samples)

    def create(self) -> ndarray:
        return self.cov


class RepeatStrategy(CovModStrategy):
    def __init__(self, s: CovStrategy):
        super().__init__(s)

    def create(self, xy: tuple|ndarray) -> ndarray:
        return np.repeat([self.s.create()], len(xy), axis=0)


class RadialMod(CovModStrategy):
    def __init__(self, s: CovStrategy, growth_rate: float):
        super().__init__(s)
        self.growth_rate = growth_rate

    def create(self, xy: ndarray) -> ndarray:
        gr = self.growth_rate
        cov = self.s.create()
        return np.array([((1 + gr) ** i * cov) for i in range(len(xy))])


class DiagMod(CovModStrategy):
    def __init__(self, s: CovStrategy, growth_rates: ndarray):
        super().__init__(s)
        self.growth_rates = growth_rates

    def create(self, xy: ndarray) -> ndarray:
        grs = self.growth_rates
        cov = self.s.create()

        assert grs.shape[0] == xy.shape[1]

        mod_arr = np.diag(1. + grs)
        return np.array([(mod_arr ** i * cov) for i in range(len(xy))])


class SkewMod(CovModStrategy):
    def __init__(self, s: CovStrategy):
        super().__init__(s)

    def create(self, xy: ndarray):
        return xy


class NoneMod(CovModStrategy):
    def create(self, xy) -> ndarray:
        return xy


### COV VARIATION
def cov_change_skew(labels: np.ndarray, cov: np.ndarray) -> list[np.ndarray]:
    """
    Makes a list of covariance matrices that become increasingly skewed
    according to their label.
    """
    n_labels = len(labels)
    return [np.dot(cov, np.array([[np.cos(i * np.pi/n_labels), np.sin(i * np.pi/n_labels)],
        [-np.sin(i * np.pi/n_labels), np.cos(i * np.pi/n_labels)]])) for i in range(n_labels)]


def sample_gaussian(nsamples: int, labels_unique: ndarray,
                    means: ndarray, covs: ndarray,
                    rng: np.random.Generator) -> tuple[ndarray, ndarray]:
    """
    Generates training samples distributed according to the Gaussians described
    by `means` and `cov_mtxs` which each correspond to a given training label

    n_samples
        number of gaussian samples drawn for each label
    labels
        list of labels for which samples are made
    means
        list of means corresponding to each label
    cov_mtxs
        list of cov. matrix corresponding to each label
    """

    assert len(labels_unique) == len(means)

    ndim = covs[0].shape[-1]
    ndim_label = labels_unique.shape[-1]

    # vector of samples from 'ndim'-dimension gaussians
    samples = np.empty((0, ndim), dtype=float)
    # vector of labels for each sample taken
    labels = np.empty((0, ndim_label), dtype=float)

    # create data and label vectors for training
    for i in range(len(means)):
        samples_i = rng.multivariate_normal(means[i], covs[i], size=nsamples)
        samples = np.concatenate((samples, samples_i), axis=0)

        labels_i = np.repeat(labels_unique[i], nsamples).reshape(ndim_label, -1).T
        labels = np.concatenate((labels, labels_i), axis=0)

    return samples, labels
