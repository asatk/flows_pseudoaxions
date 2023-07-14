"""
Author: Anthony Atkinson
Modified: 2023.07.14

Utility functions for generating, loading, and manipulating training data for
a normalizing flow.
"""

import numpy as np
import os
import uproot as up

import defs
from .io import LOG_WARN, LOG_FATAL
from .io import print_msg

# Strings necessary for reading data from a ROOT TTree
# These are likely to change - highly contingent upon what the cuts and vars
# are for each run.
# >>>> find more flexible way to load ROOT data. Maybe a config file or parser
up.default_library = "np"
cutstr = "CBL_Region == 1"
phistr = "CBL_RecoPhi_mass"
omegastr = "TwoProng_massPi0"
omegaidxstr = "CBL_RecoPhi_twoprongindex"
ptstr = "Photon_pt"
ptidxstr = "CBL_RecoPhi_photonindex"
labelphistr = "GenPhi_mass"
labelomegastr = "GenOmega_mass"

def makedata(mode: int, load_data_path: str=None, save_data_path: str=None,
             use_whiten: bool=True, overwrite: bool=False) -> tuple[np.ndarray, np.ndarray]:
    # TODO update docstring
    """
    Make data for a given training scenario. Returns both training samples 
    and conditional data (labels) associated with each sample.

    mode, int
        The scenario for which data are needed for training or testing
    load_data_path, str
        The path to non-numpy data that are loaded and compiled, e.g., .ROOT
    save_data_path, str
        The path to the location where the newly-made data are stored
    use_whiten, bool
        Whether or not to use whitening in pre-processing the before they are
        saved or used for training. This is PREFFERED and will almost always
        yield signicantly improved results and quicker, stabler learning.
    """

    if mode == defs.ROOT and load_data_path is None:
        print_msg("No path provided to load data (mode=ROOT)", level=LOG_FATAL)
    
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
        samples, labels = _loadallroot(load_data_path, defs.event_threshold)
    
    else:
        print_msg("This type of training data generation is not implemented",
                  level=LOG_FATAL)

    # Whiten data (shift and scale s.t. data has 0 mean and unit width)
    if use_whiten:
        data, whiten_data = whiten(samples)
        cond, whiten_cond = whiten(labels)
    else:
        data = samples
        cond = labels
        whiten_data = {}
        whiten_cond = {}
    
    # Save the ready-for-training data, their rergression labels and whitening data
    save_data_dict(save_data_path, data=data, cond=cond,
                   whiten_data=whiten_data, whiten_cond=whiten_cond,
                   overwrite=overwrite)
    
    return data, cond


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


def whiten(data: np.ndarray, whiten_data: dict|None=None) -> tuple[np.ndarray, dict[str, float]]:
    # TODO update docstring
    """
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
    """

    if whiten_data is not None:
        min_norm = whiten_data["min"]
        max_norm = whiten_data["max"]
    else:
        # perhaps make epsilon propto max - min
        epsilon = 1e-5
        min_norm = np.min(data, axis=0) - epsilon
        max_norm = np.max(data, axis=0) + epsilon

    # confine data to the unit interval [0., 1.]
    data_unit = (data - min_norm) / (max_norm - min_norm)

    # apply 'logit' transform to map unit interval to (-inf, +inf)
    data_logit = np.log(1 / ((1 / data_unit) - 1))

    if whiten_data is not None:
        mean_norm = whiten_data["mean"]
        std_norm = whiten_data["std"]
    else:
        mean_norm = np.mean(data_logit, axis=0)
        std_norm = np.std(data_logit, axis=0)
        whiten_data = {"min": min_norm, "max": max_norm, "mean": mean_norm, "std": std_norm}

    # standardize the data to have 0 mean and unit variance
    data_norm = (data_logit - mean_norm) / std_norm
 
    return data_norm, whiten_data


def dewhiten(data_norm: np.ndarray, whiten_data: dict) -> np.ndarray:
    # TODO update docstring
    """
    Invert the standardized data that were output from the network into values
    that are interpretable to the end user. The inverse transformation must use
    the same min/max/mean/std values that were used to first whiten the data
    for an accurate representation of the samples.

    data: np.ndarray
        raw N-D coordinates that have not been pre-processed
    normdata_path: str, None
        path to file where whitening data are stored for unwhitening. Returns
        array of zeroes with same shape as data_norm if not a valid path.
    """

    min_norm = whiten_data["min"]
    max_norm = whiten_data["max"]
    mean_norm = whiten_data["mean"]
    std_norm = whiten_data["std"]

    # invert the standardized values from the mean and std
    data_logit = data_norm * std_norm + mean_norm

    # apply inverse 'logit' transform to map from (-inf, +inf) to unit interval
    data_unit = 1 / (1 + np.exp(-data_logit))
    
    # shift data from unit interval back to its intended interval
    data = data_unit * (max_norm - min_norm) + min_norm

    return data


def load_data_dict(data_dict_path: str, ret_dict: bool=False) -> tuple[np.ndarray, np.ndarray, dict, dict]|dict:
    # TODO docstring
    """
    Loads any data present in the provided path. These data are limited to
    training or generated data, conditional data, and whitening constants for
    both of these. Data are loaded for each keyword from the set "data",
    "cond", "whiten_data", and "whiten_cond" that is found. If the path does
    not point to a valid .npy file, an empty dictionary is returned. If any of
    the data corresponding to each keyword cannot be found, an empty analog of
    each type is returned. If specified by the option `ret_dict`, just the
    dictionary loaded from the file is returned.

    data_dict_path: str, path to the .npy file containing a dictionary of,
        at most, data, conditional data, and whitening data for both. If the
        path is not valid, an empty dictionary is returned.

    ret_dict : bool (default=False), returns the dictionary loaded from the
        specified path rather than a tuple of all of the data. May be useful
        if custom data are stored and wished to be retrieved later in a file
        that also holds the usual `data_dict` data.
    """


    if not os.path.isfile(data_dict_path):
        print_msg(f"The path '{data_dict_path}' does not exist or is not a file..." +
                  "returning an empty dictionary...", level=LOG_WARN)
        return {}
    
    data_dict: dict= np.load(data_dict_path, allow_pickle=True).item()
    
    if ret_dict:
        return data_dict
    else:
        kws = data_dict.keys()

        if "data" in kws:
            data = data_dict["data"]
        else:
            data = np.empty((0, ))
            print_msg("Data not found " +
                      f"in file {data_dict_path}... returning an empty " +
                      "array...", level=LOG_WARN)
        
        if "cond" in kws:
            cond = data_dict["cond"]
        else:
            cond = np.empty((0, ))
            print_msg("Conditional data not found " +
                      f"in file {data_dict_path}... returning an empty " +
                      "array...", level=LOG_WARN)

        if "whiten_data" in kws:
            whiten_data = data_dict["whiten_data"]
        else:
            whiten_data = {}
            print_msg("Whitening constants for data not found " +
                      f"in file {data_dict_path}... returning an empty " +
                      "dictionary...", level=LOG_WARN)

        if "whiten_cond" in kws:
            whiten_cond = data_dict["whiten_cond"]
        else:
            whiten_cond = {}
            print_msg("Whitening constants for conditional data not found " +
                      f"in file {data_dict_path}... returning an empty " +
                      "dictionary...", level=LOG_WARN)
        
        return data, cond, whiten_data, whiten_cond


def save_data_dict(data_dict_path: str, data: np.ndarray, cond: np.ndarray,
                   whiten_data: dict, whiten_cond: dict, overwrite: bool=False,
                   **kwargs) -> None:
    # TODO update docstring
    # TODO change any positional args to args w default values. No data
    # have to be saved necessarily.
    """
    Saves any data present in the provided path. 
    
    data_dict_path: str, path to the .npy file where the dictionary containing
        the specified data will all together be stored
    """

    if not overwrite and os.path.isfile(data_dict_path):
        print_msg(f"The path '{data_dict_path}' exists already and will not"
                  " be overwritten. No data will be saved...", level=LOG_WARN)
        return
    
    training_data = {"data": data, "cond": cond, "whiten_data": whiten_data,
                     "whiten_cond": whiten_cond, **kwargs}

    np.save(data_dict_path, training_data,allow_pickle=True)
    

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
    """
    REVIEW IMPLEMENTATION FOR ACCURACY
    """
    n_labels = len(labels)
    return [np.dot(cov, np.array([[np.cos(i * np.pi/n_labels), np.sin(i * np.pi/n_labels)], [-np.sin(i * np.pi/n_labels), np.cos(i * np.pi/n_labels)]])) for i in range(n_labels)]