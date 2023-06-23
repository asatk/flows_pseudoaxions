'''
Definitions of utility functions for training CCGAN.

Author: Anthony Atkinson
'''

from datetime import datetime
import multiprocessing as mp
import numpy as np
import os
import tensorflow as tf
import uproot as up

import defs as defs
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

def makedata(mode, datarunname=None, rootdatapath=None):

    if datarunname is None:
        datarunname = datetime.today().date()

    datadir = "data/"

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
            sample_real_gaussian(defs.nsamp, uniquelabels, ctrs, gaus_covs)

        labelfile = 'gaussian_labels_%s_%02i'%(datarunname, defs.ngaus)
        datafile = 'gaussians_%s_%02i'%(datarunname, defs.ngaus)
        labelpath = myutils.namefile(datadir, labelfile, ext=".npy")
        datapath = myutils.namefile(datadir, datafile, ext=".npy")
    
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
            sample_real_gaussian(defs.nsamp, uniquelabels, ctrs, gaus_covs)

        labelfile = 'gaussian_labels_%s_%02ix%02i'%(datarunname, defs.ngausx, defs.ngausy)
        datafile = 'gaussians_%s_%02ix%02i'%(datarunname, defs.ngausx, defs.ngausy)
        labelpath = myutils.namefile(datadir, labelfile, ext=".npy")
        datapath = myutils.namefile(datadir, datafile, ext=".npy")

    elif mode == defs.ROOT:
        samples, labels = _loadallroot(rootdatapath)

        labelfile = 'root_labels_%s_%02ix%02i'%(datarunname, defs.ndistx, defs.ndisty)
        datafile = 'root_%s_%02ix%02i'%(datarunname, defs.ndistx, defs.ndisty)
        labelpath = myutils.namefile(datadir, labelfile, ext=".npy")
        datapath = myutils.namefile(datadir, datafile, ext=".npy")
    
    else:
        print("this type of training data generation is not implemented")
        return None
    
    np.save(labelpath, labels)
    np.save(datapath, samples)
    return samples, labels

def _loadallroot(datadir):
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

def _cutdata(i, omega, labelphi, labelomega, pt, ptidx, omegaidx):
    # index and cut criterion
    # for i in range(len(omega)):
    if pt[i][ptidx[i]] > 220:
        omega[i] = omega[i][omegaidx[i]]
        labelphi[i] = labelphi[i][0]
        labelomega[i] = labelomega[i][0]
    else:
        omega[i] = 0
    if pt[ptidx] > 220:
        return (omega[omegaidx], labelphi[0], labelomega[0])
    else:
        return None

def _loadoneroot(rootdatapath):
    if os.stat(rootdatapath).st_size == 0:
        print("--- ^ empty file ^ ---")
        samples = np.empty((0, 2))
        labels = np.empty((0, 2))
        return samples, labels

    datafile = up.open(rootdatapath)
    events = datafile["Events;1"]

    arrs = events.arrays([phistr, omegastr, omegaidxstr, ptstr, ptidxstr,
                          labelphistr, labelomegastr], cut=cutstr, library="np")

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

    # print(len(omega))

    omega_temp = np.empty_like(omega, dtype=np.float32)
    labelphi_temp = np.empty_like(labelphi, dtype=np.float32)
    labelomega_temp = np.empty_like(labelomega, dtype=np.float32)

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

    # print(len(newomega))

    # print(np.where(cutarr))
    # print(newphi)
    # print(newomega)
    # print(newlabelphi)
    # print(newlabelomega)

    # phi, omega, labelphi, labelomega = data

    samples = np.stack((newphi, newomega), axis=1)
    labels = np.stack((newlabelphi, newlabelomega), axis=1)

    datafile.close()

    # print(samples)
    # print(labels)

    # exit()

    return samples, labels

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