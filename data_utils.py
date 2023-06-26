'''
Utility functions for generating, loading, and manipulating training data for
a normalizing flow.

Author: Anthony Atkinson
'''

from datetime import datetime
import numpy as np
import os
import uproot as up

import defs
import utils as myutils

up.default_library = "np"
cutstr = "CBL_Region == 1"
phistr = "CBL_RecoPhi_mass"
omegastr = "TwoProng_massPi0"
omegaidxstr = "CBL_RecoPhi_twoprongindex"
ptstr = "Photon_pt"
ptidxstr = "CBL_RecoPhi_photonindex"
labelphistr = "GenPhi_mass"
labelomegastr = "GenOmega_mass"
numworkers = 4

def makedata(mode, data_path=None, normalize=False):

    if mode == defs.ROOT and data_path is None:
        print("no path provided to load data")
        exit()

    data_dir = defs.data_dir
    run_name = defs.run_name
    
    samples_path = "%s/%s_samples.npy"%(data_dir, run_name)
    labels_path = "%s/%s_labels.npy"%(data_dir, run_name)
    normdata_path = "%s/%s_normdata.npy"%(data_dir, run_name)

    # >>>> NORMALIZE DATA!!! ONLY NORMALIZING LABELS

    if mode == defs.LINE:
        
        # assign the 1D labels for each gaussian
        uniquelabels = normalize_labels_line_1d(
            train_labels_line_1d(defs.ngaus))
        # calculate expected center of each gaussian
        ctrs = dist_center_line_1d(uniquelabels, defs.yval)

        # generate cov mtx for each gaussian
        # gaus_cov_indv = myutils.cov_skew(gaus_width, gaus_width/2., gaus_width/2., gaus_width)
        gaus_cov_indv = cov_xy(defs.sigma_gaus)
        gaus_covs = cov_change_none(ctrs, gaus_cov_indv)
        # samples each gaussian for n_samples points, each with an associated label
        samples, labels = \
            sample_gaussian(defs.nsamp, uniquelabels, ctrs, gaus_covs)
    
    elif mode == defs.GRID:

        uniquelabels = normalize_labels_grid_2d(
            train_labels_grid_2d(defs.ngausx, defs.ngausy))
        ctrs = dist_center_grid_2d(uniquelabels)

        # generate cov mtx for each gaussian
        # gaus_cov_indv = myutils.cov_skew(gaus_width, gaus_width/2., gaus_width/2., gaus_width)
        gaus_cov_indv = cov_xy(defs.sigma_gaus)
        gaus_covs = cov_change_none(ctrs, gaus_cov_indv)
        # samples each gaussian for n_samples points, each with an associated label
        samples, labels = \
            sample_gaussian(defs.nsamp, uniquelabels, ctrs, gaus_covs)

    elif mode == defs.ROOT:

        samples, labels = _loadallroot(data_path)
    
    else:
        print("this type of training data generation is not implemented")
        return None
    
    # Normalize/whiten data
    if normalize:
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        data = (samples - mean) / std
        np.save(normdata_path, np.array([mean, std]))
    else:
        data = samples
    
    # Save the ready-for-training data and their labels
    np.save(labels_path, labels)
    np.save(samples_path, data)
    
    return samples, labels

# Recursively loads all of the .ROOT files in the entire subdirectory tree
# located at `datadir` into a numpy array
def _loadallroot(datadir: str) -> np.ndarray:
    samples = np.empty((0, 2))
    labels = np.empty((0, 2))
    with os.scandir(datadir) as d:
        for entry in d:
            if entry.is_dir():
                samples_temp, labels_temp = _loadallroot(datadir + '/' + entry.name)
                samples = np.concatenate((samples_temp, samples), axis=0)
                labels = np.concatenate((labels_temp, labels), axis=0)
            else:
                print(datadir + '/' + entry.name)
                samples, labels = _loadoneroot(datadir + '/' + entry.name)

    return samples, labels

# Loads all of the events from a single .ROOT file at `datapath` that pass the
# cut determined within this function. Special care is taken to index the event
# arrays according to variables like 'twoprongindex' - this is taken care of in
# this function and this function only. To define another cut or variable to be
# indexed, this function should be modified or a new one provided.
def _loadoneroot(datapath: str) -> np.ndarray:
    if os.stat(datapath).st_size == 0:
        print("--- ^ empty file ^ ---")
        samples = np.empty((0, 2))
        labels = np.empty((0, 2))
        return samples, labels

    # load data
    datafile = up.open(datapath)
    events = datafile["Events;1"]

    # fetch desired columns
    arrs = events.arrays([phistr, omegastr, omegaidxstr, ptstr, ptidxstr,
                          labelphistr, labelomegastr], cut=cutstr, library="np")

    
    # cut out events that don't have a valid omega/pt to index
    omegaidxarr = arrs[omegaidxstr]
    ptidxarr = arrs[ptidxstr]

    # -1 & -1 == -1 only way to get -1 b/c -1 is all bits
    idxcutarr = ((omegaidxarr & ptidxarr) != -1)

    phi = arrs[phistr][idxcutarr]
    omega = arrs[omegastr][idxcutarr]
    omegaidx = arrs[omegaidxstr][idxcutarr]
    pt = arrs[ptstr][idxcutarr]
    ptidx = arrs[ptidxstr][idxcutarr]
    labelphi = arrs[labelphistr][idxcutarr]
    labelomega = arrs[labelomegastr][idxcutarr] 

    omega_temp = np.empty_like(omega, dtype=np.float32)
    labelphi_temp = np.empty_like(labelphi, dtype=np.float32)
    labelomega_temp = np.empty_like(labelomega, dtype=np.float32)

    # perform cut and extract correct element with index
    for i in range(len(omega)):
        if pt[i][ptidx[i]] > 220:
            omega_temp[i] = omega[i][omegaidx[i]]
            labelphi_temp[i] = labelphi[i][0]
            labelomega_temp[i] = labelomega[i][0]
        else:
            omega_temp[i] = np.nan

    cutarr = np.isfinite(omega_temp)
    newphi = phi[cutarr].copy()
    newomega = omega_temp[cutarr].copy()
    newlabelphi = labelphi_temp[cutarr].copy()
    newlabelomega = labelomega_temp[cutarr].copy()

    # for 10k dataset, this is 1.1%
    if (len(newphi) < 110):
        samples = np.empty((0, 2), dtype=np.float32)
        labels = np.empty((0, 2), dtype=np.float32)
    
    # compile samples and labels array
    else:
        samples = np.stack((newphi, newomega), axis=1)
        labels = np.stack((newlabelphi, newlabelomega), axis=1)

    datafile.close()

    return samples, labels

# Generates training samples distributed according to the Gaussians described
# by `means` and `cov_mtxs` which each correspond to a given training label
def sample_gaussian(n_samples: int, labels_unique: np.ndarray, means: np.ndarray, cov_mtxs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    '''
    n_samples:      number of gaussian samples drawn for each label
    labels:         list of labels for which samples are made
    means:          list of means corresponding to each label
    cov_mtxs:       list of cov. matrix corresponding to each label
    '''

    assert len(labels_unique) == len(means)
    assert len(labels_unique) == len(cov_mtxs)

    ndim = cov_mtxs[0].shape[-1]
    ndim_label = labels_unique.shape[-1]

    # vector of samples from 'ndim'-dimension gaussians
    samples = np.empty((0, ndim), dtype=float)
    # vector of labels for each sample taken
    labels = np.empty((0, ndim_label), dtype=float)

    # create data and label vectors for training
    for i in range(len(means)):
        samples_i = np.random.multivariate_normal(means[i], cov_mtxs[i], size=n_samples)
        samples = np.concatenate((samples, samples_i), axis=0)

        labels_i = np.repeat(labels_unique[i], n_samples).reshape(ndim_label, -1).T
        labels = np.concatenate((labels, labels_i), axis=0)

    return samples, labels

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