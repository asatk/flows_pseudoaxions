'''
Definitions of utility functions for training CCGAN.

Author: Anthony Atkinson
'''

import numpy as np
import os
import tensorflow as tf

import defs as defs

def sample_real_gaussian(n_samples: int, labels: np.ndarray, gaus_points: np.ndarray, cov_mtxs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    '''

    n_samp_per_gaussian: number of samples drawn from each gaussian
    labels: list of labels for which samples are made
    gaus_points:    point where the i-th label is sampled from a circular gaussian
    cov_mtxs:       spread of the i-th gaussian in x and y (2dim)

    '''

    dim = cov_mtxs[0].shape[0]
    nvars = labels.shape[1]

    # n_samples samples from 'dim'-dimension gaussian
    samples = np.empty((0, dim), dtype=float)
    # a labels corresponding to each sample taken
    sampled_labels = np.empty((0,nvars), dtype=float)

    for i in range(len(gaus_points)):
        point_i = gaus_points[i]
        label_i = labels[i]
        cov_mtx_i = cov_mtxs[i]

        samples_i = np.random.multivariate_normal(point_i, cov_mtx_i, size=n_samples)
        samples = np.concatenate((samples, samples_i), axis=0)

        sampled_labels_i = np.repeat(label_i, n_samples).reshape(nvars, -1).T
        sampled_labels = np.concatenate((sampled_labels, sampled_labels_i), axis=0)

    return samples, sampled_labels


def train_labels_circle(n_train: int) -> np.ndarray:
    return np.array([np.linspace(0, 2*np.pi, n_train, endpoint=False)]).T

def train_labels_line_1d(n_train: int) -> np.ndarray:
    return np.array([np.add(np.linspace(defs.xmin, defs.xmax, n_train + 1, endpoint=False)[:-1], (defs.xmax - defs.xmin) / (n_train + 1))]).T

def train_labels_grid_2d(ngausx: int, ngausy: int) -> np.ndarray:
    xax = np.linspace(defs.xmin, defs.xmax, ngausx + 1, endpoint=False)[:-1] + (defs.xmax - defs.xmin) / (ngausx + 1)
    yax = np.linspace(defs.ymin, defs.ymax, ngausy + 1, endpoint=False)[:-1] + (defs.ymax - defs.ymin) / (ngausy + 1)
    x, y = np.meshgrid(xax, yax)
    return np.array([x.ravel(), y.ravel()]).T

def train_labels_psuedo_phi(n_train: int) -> np.ndarray:
    return np.linspace(defs.phi_min, defs.phi_max, n_train, endpoint=False)

def train_labels_psuedo_omega(n_train: int) -> np.ndarray:
    return np.linspace(defs.omega_min, defs.omega_max, n_train, endpoint=False)

def train_labels_pseudo_2d(ndistx: int, ndisty: int) -> np.ndarray:
    xax = np.linspace(defs.phi_min, defs.omega_max, ndistx + 1, endpoint=False)[:-1] + \
        (defs.phi_max - defs.phi_min) / (ndistx + 1)
    yax = np.linspace(defs.omega_min, defs.omega_max, ndisty + 1, endpoint=False)[:-1] + \
        (defs.omega_max - defs.omega_min) / (ndisty + 1)
    x, y = np.meshgrid(xax, yax)
    return np.array([x.ravel(), y.ravel()]).T

def normalize_labels_circle(labels: np.ndarray) -> np.ndarray:
    return np.divide(labels, 2*np.pi)

def recover_labels_circle(labels: np.ndarray) -> np.ndarray:
    return np.multiply(labels, 2*np.pi)

def normalize_labels_line_1d(labels: np.ndarray) -> np.ndarray:
    return np.divide(np.subtract(labels, defs.xmin), (defs.xmax - defs.xmin))

def recover_labels_line_1d(labels: np.ndarray) -> np.ndarray:
    return np.add(np.multiply(labels, (defs.xmax - defs.xmin)), defs.xmin)

def normalize_labels_grid_2d(labels: np.ndarray) -> np.ndarray:
    return np.divide(np.subtract(labels, [defs.xmin, defs.ymin]),
            np.max([defs.xmax - defs.xmin, defs.ymax - defs.ymin]))

def recover_labels_grid_2d(labels: np.ndarray) -> np.ndarray:
    return np.add(np.multiply(labels,
            np.max([defs.xmax - defs.xmin, defs.ymax - defs.ymin])), [defs.xmin, defs.ymin])

def normalize_labels_pseudo_phi(labels: np.ndarray) -> np.ndarray:
    return np.divide(np.subtract(labels, defs.phi_min), (defs.phi_max - defs.phi_min))

def recover_labels_pseudo_phi(labels: np.ndarray) -> np.ndarray:
    return np.add(np.multiply(labels, (defs.phi_max - defs.phi_min)), defs.phi_min)

def normalize_labels_pseudo_omega(labels: np.ndarray) -> np.ndarray:
    return np.divide(np.subtract(labels, defs.omega_min), (defs.omega_max - defs.omega_min))

def recover_labels_pseudo_omega(labels: np.ndarray) -> np.ndarray:
    return np.add(np.multiply(labels, (defs.omega_max - defs.omega_min)), defs.omega_min)

def normalize_labels_pseudo_2d(labels: np.ndarray) -> np.ndarray:
    return np.divide(np.subtract(labels, [defs.phi_min, defs.omega_min]),
            np.max([defs.phi_max - defs.phi_min, defs.omega_max - defs.omega_min]))

def recover_labels_pseudo_2d(labels: np.ndarray) -> np.ndarray:
    return np.add(np.multiply(labels,
            np.max([defs.phi_max - defs.phi_min, defs.omega_max - defs.omega_min])), [defs.phi_min, defs.omega_min])

def dist_center_circle(labels: np.ndarray, radius: float) -> np.ndarray:
    return np.multiply([np.sin(labels), np.cos(labels)], radius).T

def dist_center_line_1d(labels: np.ndarray, yval: float) -> np.ndarray:
    return np.concatenate((labels, np.array([np.repeat(yval, len(labels))]).T), axis=1)

def dist_center_grid_2d(labels: np.ndarray) -> np.ndarray:
    # return np.stack(labels, axis=1)
    return labels

def plot_lims_circle(radius: float) -> np.ndarray:
    return np.multiply(np.ones((2,2)), radius)

def plot_lims_line_1d() -> np.ndarray:
    return np.array([[defs.xmin, defs.xmax], [defs.ymin, defs.ymax]])

def plot_lims_grid() -> np.ndarray:
    return np.array([[defs.xmin, defs.xmax], [defs.ymin, defs.ymax]])

def plot_lims_line_pseudo() -> np.ndarray:
    return np.array([[defs.phi_min, defs.phi_max], [defs.omega_min, defs.omega_max]])

def cov_xy(sigma1: float, sigma2: float=None) -> np.ndarray:
    if sigma2 is None:
        sigma2 = sigma1
    return np.array([[sigma1**2, 0.],[0., sigma2**2]])

def cov_skew(cov11: float, cov12: float, cov21: float=None, cov22: float=None) -> np.ndarray:
    if cov21 is None:
        cov21 = cov12
    if cov22 is None:
        cov22 = cov11
    return np.power(np.array([[cov11, cov12],[cov21, cov22]]), 2.)
    
def cov_change_none(labels: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.repeat([cov], len(labels), axis=0)

def cov_change_radial(labels: np.ndarray, cov: np.ndarray):
    return [np.dot(cov, 
        np.eye(2) * (1 + (defs.xcov_change_linear_max_factor - 1) * label[0] / defs.xmax)) for label in labels]

def cov_change_linear(labels: np.ndarray, cov: np.ndarray) -> list[np.ndarray]:
    return [np.dot(cov, 
        np.array([[(1 + (defs.xcov_change_linear_max_factor - 1) * label[0] / defs.xmax), 0.], 
        [0., (1 + (defs.ycov_change_linear_max_factor - 1) * label[1] / defs.ymax)]])) for label in labels]

def cov_change_skew(labels: np.ndarray, cov: np.ndarray) -> list[np.ndarray]:
    '''
    NOT IMPLEMENTED PROPERLY
    '''
    n_labels = len(labels)
    return [np.dot(cov, np.array([[np.cos(i * np.pi/n_labels), np.sin(i * np.pi/n_labels)], [-np.sin(i * np.pi/n_labels), np.cos(i * np.pi/n_labels)]])) for i in range(n_labels)]

def namefile(dirname: str, name: str, isdir=False, ext: str=None):

    if isdir or ext is None:
        ext = ""

    d = os.listdir(dirname)
    i = 1
    f = name + ext
    
    while f in d:
        f = "%s_%i%s"%(name, i, ext)
        i += 1

    f = dirname + f

    if isdir:
        f += '/'

    return f
    
class SelectiveProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, verbose, epoch_interval, epoch_end, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = epoch_interval
        self.epoch_end = epoch_end
    
    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.verbose = (
                0 if (epoch + 1) % self.epoch_interval != 0 and (epoch + 1) != self.epoch_end
                else self.default_verbose)
        super().on_epoch_begin(epoch, *args, **kwargs)