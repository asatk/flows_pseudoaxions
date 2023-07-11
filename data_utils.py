'''
Utility functions for generating, loading, and manipulating training data for
a normalizing flow.

Author: Anthony Atkinson
'''

import numpy as np
import os
import uproot as up

import defs
import utils as myutils

# Strings necessary for reading data from a ROOT TTree
up.default_library = "np"
cutstr = "CBL_Region == 1"
phistr = "CBL_RecoPhi_mass"
omegastr = "TwoProng_massPi0"
omegaidxstr = "CBL_RecoPhi_twoprongindex"
ptstr = "Photon_pt"
ptidxstr = "CBL_RecoPhi_photonindex"
labelphistr = "GenPhi_mass"
labelomegastr = "GenOmega_mass"

def makedata(mode: int, data_path: str=None, normalize: bool=False) -> tuple[np.ndarray, np.ndarray]:
    '''
    Make data for a given training scenario. Returns both training samples 
    and conditional data (labels) associated with each sample.

    mode, int
        The scenario for which data are needed for training or testing
    data_path, str
        The path to non-numpy data that are loaded and compiled, e.g., .ROOT
    normalize, bool
        Whether or not to normalize the data before they are saved or used
        for training.
    '''

    if mode == defs.ROOT and data_path is None:
        print("no path provided to load data")
        exit()

    data_dir = defs.data_dir
    run_name = defs.data_name
    
    data_path = "%s/%s_data.npy"%(data_dir, run_name)
    cond_path = "%s/%s_cond.npy"%(data_dir, run_name)
    normdata_path = "%s/%s_data_wtn.npy"%(data_dir, run_name)
    normdatacond_path = "%s/%s_cond_wtn.npy"%(data_dir, run_name)

    if mode == defs.LINE:
        
        # assign the 1D labels for each gaussian
        labels_unique = train_labels_line_1d(defs.ngausx)
        # determine the center of each gaussian that will be sampled
        means = means_line_1d(labels_unique, defs.val)

        # generate cov mtx for each gaussian
        cov_indv = cov_xy(defs.sigma_gaus)
        covs = cov_change_none(means, cov_indv)

        # sample each gaussian and pair each sample with its corresponding label
        samples, labels = \
            sample_gaussian(defs.nsamp, labels_unique, means, covs)
    
    elif mode == defs.GRID:

        # assign the 2D labels for each gaussian
        labels_unique = train_labels_grid_2d(defs.ngausx, defs.ngausy)
        # determine the center of each gaussian that will be sampled
        means = means_grid_2d(labels_unique)

        # generate cov mtx for each gaussian
        # gaus_cov_indv = myutils.cov_skew(defs.sigma_gaus, defs.sigma_gaus/2., defs.sigma_gaus/2., defs.sigma_gaus)
        cov_indv = cov_xy(defs.sigma_gaus)
        covs = cov_change_none(means, cov_indv)
        
        # sample each gaussian and pair each sample with its corresponding label
        samples, labels = \
            sample_gaussian(defs.nsamp, labels_unique, means, covs)

    elif mode == defs.ROOT:

        samples, labels = _loadallroot(data_path, defs.event_threshold)
    
    else:

        print("This type of training data generation is not implemented")
        exit()
    
    # >>> NEED TO WHITEN or center+scale REGRESSION LABELS!

    # Whiten data (shift and scale s.t. data has 0 mean and unit width)
    if normalize:
        data = whiten(samples, normdata_path)
        data_cond = whiten(labels, normdatacond_path)
    else:
        data = samples
        data_cond = labels
    
    # Save the ready-for-training data and their labels
    np.save(cond_path, data_cond)
    np.save(data_path, data)
    
    return data, data_cond


def _loadallroot(data_dir: str, event_threshold: float=0.01) -> np.ndarray:
    """
    Recursively loads all of the .ROOT files in the entire subdirectory tree
    located at `data_dir` into a numpy array
    """
    samples = np.empty((0, 2))
    labels = np.empty((0, 2))
    with os.scandir(data_dir) as d:
        for entry in d:
            if entry.is_dir():
                samples_temp, labels_temp = _loadallroot(data_dir + "/" + entry.name, event_threshold=event_threshold)
                samples = np.concatenate((samples_temp, samples), axis=0)
                labels = np.concatenate((labels_temp, labels), axis=0)
            else:
                print(data_dir + "/" + entry.name)
                samples, labels = _loadoneroot(data_dir + "/" + entry.name, event_threshold=event_threshold)

    return samples, labels


def _loadoneroot(data_path: str, event_threshold: float=0.01) -> np.ndarray:
    """
    Loads all of the events from a single .ROOT file at `datapath` that pass the
    cut determined within this function. Special care is taken to index the event
    arrays according to variables like 'twoprongindex' - this is taken care of in
    this function and this function only. To define another cut or variable to be
    indexed, this function should be modified or a new one provided.

    data_path
        the path to the .ROOT file where the data are located
    event_threshold **default 0.01**
        The proportion of events out of the total that a set of
        samples must have in order to be viable to be trained. 
    """
    if os.stat(data_path).st_size == 0:
        print("--- ^ empty file ^ ---")
        samples = np.empty((0, 2))
        labels = np.empty((0, 2))
        return samples, labels

    # load data
    datafile = up.open(data_path)
    nevents = datafile["Metadata;1/evtWritten"].array()[0]
    events = datafile["Events;1"]

    # fetch desired columns
    arrs = events.arrays([phistr, omegastr, omegaidxstr, ptstr, ptidxstr,
                          labelphistr, labelomegastr], cut=cutstr, library="np")

    
    # cut out events that don't have a valid omega/pt to index
    omegaidxarr = arrs[omegaidxstr]
    ptidxarr = arrs[ptidxstr]

    # -1 & -1 == -1 only way to get -1 b/c -1 is all 1 bits
    idxcutarr = ((omegaidxarr & ptidxarr) != -1)

    # select the events with valid indexes for each variable
    phi = arrs[phistr][idxcutarr]
    omega = arrs[omegastr][idxcutarr]
    omegaidx = arrs[omegaidxstr][idxcutarr]
    pt = arrs[ptstr][idxcutarr]
    ptidx = arrs[ptidxstr][idxcutarr]
    labelphi = arrs[labelphistr][idxcutarr]
    labelomega = arrs[labelomegastr][idxcutarr] 

    # arrays to store indexed data that satisfy the cut
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

    # copy only the events that satisfy the cut
    cutarr = np.isfinite(omega_temp)
    newphi = phi[cutarr].copy()
    newomega = omega_temp[cutarr].copy()
    newlabelphi = labelphi_temp[cutarr].copy()
    newlabelomega = labelomega_temp[cutarr].copy()

    # exclude data sets that have too few statistics after cuts
    if (len(newphi) < event_threshold * nevents):
        samples = np.empty((0, 2), dtype=np.float32)
        labels = np.empty((0, 2), dtype=np.float32)
    
    # compile samples and labels array
    else:
        samples = np.stack((newphi, newomega), axis=1)
        labels = np.stack((newlabelphi, newlabelomega), axis=1)

    datafile.close()

    return samples, labels


def sample_gaussian(nsamples: int, labels_unique: np.ndarray, means: np.ndarray, cov_mtxs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
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
    assert len(labels_unique) == len(cov_mtxs)

    rng = np.random.default_rng(defs.seed)

    ndim = cov_mtxs[0].shape[-1]
    ndim_label = labels_unique.shape[-1]

    # vector of samples from 'ndim'-dimension gaussians
    samples = np.empty((0, ndim), dtype=float)
    # vector of labels for each sample taken
    labels = np.empty((0, ndim_label), dtype=float)

    # create data and label vectors for training
    for i in range(len(means)):
        samples_i = rng.multivariate_normal(means[i], cov_mtxs[i], size=nsamples)
        samples = np.concatenate((samples, samples_i), axis=0)

        labels_i = np.repeat(labels_unique[i], nsamples).reshape(ndim_label, -1).T
        labels = np.concatenate((labels, labels_i), axis=0)

    return samples, labels


def whiten(data: np.ndarray, normdata_path: str, load_norm: bool=False) -> np.ndarray:
    '''
    Standardize the data to make it more 'network-friendly'. Whitened data
    are significantly easier to train on and show benefits in results. The
    statistics used to whiten the data are stored, to be used later for
    inverting this transformation back into data that is in the desired form.
    The mean and std. dev. stored as the "norm data" are not that of the raw
    data but just shift and scale parameters in the last transformation of the
    whitening transformation. Conversely, the min and max stored as the "norm
    data" are the minimum and maxmimum values of the raw data along each
    coordinate axis.

    data: np.ndarray
        raw N-D coordinates that have not been pre-processed
    normdata_path: str, None
        path to file where whitening data are stored. If `load_norm` is `True`,
        the whitening constants are loaded from the file specified at the path.
        Otherwise, constants are saved at the specified location unless
        `nomrdata_path` is None.
    load_norm: bool
        flag indicating whether or not to load the whitening constants from the
        path specified. If False, these constants are determined in calculation
        of the whitening transformation and saved at normdata_path.
    '''

    if load_norm:
        normdata = np.load(normdata_path)
        min_norm = normdata["min"]
        max_norm = normdata["max"]
    else:
        min_norm = np.min(data, axis=0) - 1e-5
        max_norm = np.max(data, axis=0) + 1e-5

    # confine data to the unit interval [0., 1.]
    data_unit = (data - min_norm) / (max_norm - min_norm)

    # apply 'logit' transform to map unit interval to (-inf, +inf)
    data_logit = np.log(1 / ((1 / data_unit) - 1))

    if load_norm:
        mean_norm = normdata["mean"]
        std_norm = normdata["std"]
    else:
        mean_norm = np.mean(data_logit, axis=0)
        std_norm = np.std(data_logit, axis=0)

    # standardize the data to have 0 mean and unit variance
    data_norm = (data_logit - mean_norm) / std_norm
 
    if not load_norm:
        normdata = {"min": min_norm, "max": max_norm, "mean": mean_norm, "std": std_norm}
        if normdata_path is not None:
            np.save(normdata_path, normdata)

    return data_norm

def dewhiten(data_norm: np.ndarray, normdata_path) -> np.ndarray:
    '''
    Invert the standardized data that were output from the network into values
    that are interpretable to the end user. The inverse transformation must use
    the same min/max/mean/std values that were used to first whiten the data
    for an accurate representation of the samples.

    data: np.ndarray
        raw N-D coordinates that have not been pre-processed
    normdata_path: str, None
        path to file where whitening data are stored for unwhitening. Returns
        array of zeroes with same shape as data_norm if not a valid path.
    '''

    if normdata_path is None or not os.path.isfile(normdata_path):
        print("`normdata_path` cannot be None - must be a valid path to" + \
              "unwhitening data. Returning zeroes.")
        return np.zeros_like(data_norm)
    normdata = np.load(normdata_path, allow_pickle=True).item()

    min_norm = normdata["min"]
    max_norm = normdata["max"]
    mean_norm = normdata["mean"]
    std_norm = normdata["std"]

    # invert the standardized values from the mean and std
    data_logit = data_norm * std_norm + mean_norm

    # apply inverse 'logit' transform to map from (-inf, +inf) to unit interval
    data_unit = 1 / (1 + np.exp(-data_logit))
    
    # shift data from unit interval back to its intended interval
    data = data_unit * (max_norm - min_norm) + min_norm

    return data


def train_labels_circle(n_train: int) -> np.ndarray:
    return np.array([np.linspace(0, 2*np.pi, n_train, endpoint=False)]).T

def train_labels_line_1d(n_train: int, xmin: float=0., xmax: float=1.) -> np.ndarray:
    return np.array([np.add(np.linspace(xmin, xmax, n_train + 1, endpoint=False)[:-1], (xmax - xmin) / (n_train + 1))]).T

def train_labels_grid_2d(ngausx: int, ngausy: int, xmin: float=0., xmax: float=1., ymin: float=0., ymax: float=1.) -> np.ndarray:
    xax = np.linspace(xmin, xmax, ngausx + 1, endpoint=False)[:-1] + (xmax - xmin) / (ngausx + 1)
    yax = np.linspace(ymin, ymax, ngausy + 1, endpoint=False)[:-1] + (ymax - ymin) / (ngausy + 1)
    x, y = np.meshgrid(xax, yax)
    return np.array([x.ravel(), y.ravel()]).T

def means_circle(labels: np.ndarray, radius: float) -> np.ndarray:
    return np.multiply([np.sin(labels), np.cos(labels)], radius).T

def means_line_1d(labels: np.ndarray, yval: float) -> np.ndarray:
    return np.concatenate((labels, np.array([np.repeat(yval, len(labels))]).T), axis=1)

def means_grid_2d(labels: np.ndarray) -> np.ndarray:
    return labels

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