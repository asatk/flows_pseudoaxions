import abc
import numpy as np
import tensorflow as tf

class AbstractGaussians(metaclass=abc.ABCMeta):
    def __init__(self,
                 n_dim: int=2,
                 seed: int=0x1ace,
                 dtype: np.dtype=np.float32):
        self.n_dim = n_dim
        self.dtype = dtype
        self._rng = tf.random.Generator.from_seed(seed=seed)

    @abc.abstractmethod
    def generate(self, n_observations: int) -> tuple[tf.Tensor, tf.Tensor]:
        ...

class RandomGaussians(AbstractGaussians):
    """
    Draws samples from Gaussian distributions randomly located in the `n_dim`-
    dimensional unit hypercube.
    """

    def __init__(self,
                 scale: float,
                 n_dim: int=2,
                 seed: int=0x1ace,
                 dtype: np.dtype=np.float32):
        super().__init__(n_dim=n_dim, seed=seed, dtype=dtype)
        self.scale = scale

    def generate(self, n_observations: int):
        shape = (n_observations, self.n_dim)
        labels = self._rng.uniform(shape=shape, dtype=self.dtype)
        sample = labels + self._rng.normal(shape=shape, dtype=self.dtype)
        return sample, labels

class GridGaussians(AbstractGaussians):
    """
    Drwas samples from Gaussian distributions evenly spaced within the `n_dim`-
    dimensional unit hypercube.
    """

    def __init__(self,
                 scale: float,
                 n_locs: int=10,
                 n_dim: int=2,
                 seed=0x1ace,
                 dtype: np.dtype=np.float32):
        super().__init__(n_dim=n_dim, seed=seed, dtype=dtype)
        self.scale = scale
        self.n_locs = n_locs
        
        step = 1. / (self.n_locs + 1)
        locs_along_axis = np.arange(start=step,
                                    stop=1. - 1e-6,
                                    step=step,
                                    dtype=self.dtype)
        arrs = np.meshgrid(*([locs_along_axis] * self.n_dim))
        self.labels = np.array(list(map(np.ravel, arrs)), dtype=self.dtype).T

    def generate(self, n_observations: int, draw_per_label: bool=False):
        shape = (n_observations * self.n_locs ** self.n_dim, self.n_dim)
        labels = np.repeat(self.labels, repeats=n_observations, axis=0)
        return (labels + self._rng.normal(shape=shape, dtype=self.dtype),
                labels + tf.zeros_like(labels))

class CircleGaussians(AbstractGaussians):
    """
    Draws samples from Gaussian distributions evenly spaced on the unit circle.
    """

    def __init__(self,
                 scale: float,
                 n_locs: int=12,
                 use_angle: bool=False,
                 seed=0x1ace,
                 dtype: np.dtype=np.float32):
        super().__init__(n_dim=2, seed=seed, dtype=dtype)
        self.scale = scale
        self.n_locs = n_locs
        self.use_angle = use_angle

        step = 2 * np.pi / n_locs
        self.angles = np.arange(start=0,
                           stop=2 * np.pi - 1e-6,
                           step=step,
                           dtype=self.dtype)
        self.locs = np.stack(
            (np.sin(self.angles), np.cos(self.angles)),
            axis=-1,
            dtype=self.dtype)
        
        self.labels = self.angles if use_angle else self.locs

    def generate(self, n_observations: int):
        shape = (n_observations, self.n_dim)
        return (self.locs + self._rng.normal(shape=shape, dtype=self.dtype),
                self.labels + tf.zeros_like(self.labels))