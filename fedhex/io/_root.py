"""
Author: Anthony Atkinson
Modified: 2023.07.20

I/O for .ROOT files.
"""


import numpy as np
import os
import re
from typing import Callable
import uproot as up

from ..utils import print_msg, LOG_WARN


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
expressions = [
    phistr, omegastr, omegaidxstr, ptstr, ptidxstr, labelphistr, labelomegastr]


def _find_root(data_dir, max_depth: int=3):
    """
    Locate the paths to all of the .ROOT files in a given directory. This
    function is recursive and will only descend the number of subtrees as given
    by `max_depth`.
    """
    p = re.compile(".+\.ROOT$", re.IGNORECASE)
    file_list = []
    with os.scandir(data_dir) as d:
        for entry in d:
            if entry.is_dir():
                if max_depth <= 0:
                    print_msg("Maximum recursion depth reached.", level=LOG_WARN)
                    return file_list
                else:
                    list_subtree = _find_root(data_dir + "/" + entry.name, max_depth=max_depth-1)
                    file_list.extend(list_subtree)
            elif p.match(entry.name) is not None:
                path = data_dir + "/" + entry.name
                if os.stat(path).st_size == 0:
                    print_msg(f"--- {path}: empty file ---", level=LOG_WARN)
                    return []
                return [path + ":Events;1"]
            
    return file_list


def _evt_sel_1(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    # cut out events that don't have a valid omega/pt to index
    omegaidxarr = arr[omegaidxstr]
    ptidxarr = arr[ptidxstr]

    # -1 & -1 == -1 only way to get -1 b/c -1 is all 1 bits
    # idxcutarr = ((omegaidxarr | ptidxarr) != -1)
    idxcutarr = ((omegaidxarr & ptidxarr) != -1)

    # select the events with valid indexes for each variable
    phi = arr[phistr][idxcutarr]
    omega = arr[omegastr][idxcutarr]
    omegaidx = arr[omegaidxstr][idxcutarr]
    pt = arr[ptstr][idxcutarr]
    ptidx = arr[ptidxstr][idxcutarr]
    labelphi = arr[labelphistr][idxcutarr]
    labelomega = arr[labelomegastr][idxcutarr]

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

    samples = np.stack((newphi, newomega), axis=1)
    labels = np.stack((newlabelphi, newlabelomega), axis=1)

    return samples, labels


def _load_root(root_paths, event_selection_fn: Callable[[np.ndarray],
        tuple[np.ndarray]], expressions: list[str], cutstr: str,
        event_thresh: int=100):
    """
    Load events from all provided .ROOT files using a given event selection
    scheme. 
    """
        
    num_workers = 4
    arr = up.concatenate(root_paths, expressions=expressions, cut=cutstr,
        num_workers=num_workers, begin_chunk_size="2 MB", library="np")

    samples_temp, labels_temp = event_selection_fn(arr,
        expressions=expressions, cutstr=cutstr)

    # separate samples/labels into groups per label
    labels_unique, inverse_unique = np.unique(labels_temp, return_inverse=True,
        axis=0)
    samples_grp = [
        samples_temp[inverse_unique == i] for i in range(len(labels_unique))
    ]

    # compile samples and labels array
    samples = np.zeros(shape=(0, samples_temp.shape[-1]))
    labels = np.zeros(shape=(0, labels_temp.shape[-1]))

    for i, sample_i in enumerate(samples_grp):
        label_i = labels_unique[i]
        # only include labels with sufficiently many statistics after cuts
        if len(sample_i) >= event_thresh:
            samples = np.r_[samples, sample_i]
            labels = np.r_[labels, [label_i]]

    return samples, labels


def _loadallroot(data_dir: str, event_threshold: float=0.01) -> np.ndarray:
    """
    Recursively loads all of the .ROOT files in the entire subdirectory tree
    located at `data_dir` into a numpy array
    """
    # TODO test recursive vs iterative open and compile of .ROOT files. I/O is
    # expensive, but so can syscalls, i.e., dir traversal. Perhaps simply
    # locate absolute paths of all files matching pattern up to some N. Then
    # load files into numpy in chunks of size M, concatenating, applying cuts
    # and the other stuff as a whole, and THEN returning (sub)array.
    # This comes in later update
    samples = np.empty((0, 2))
    labels = np.empty((0, 2))
    p = re.compile(".+\.ROOT$", re.IGNORECASE)
    with os.scandir(data_dir) as d:
        for entry in d:
            if entry.is_dir():
                samples_temp, labels_temp = _loadallroot(data_dir + "/" + entry.name, event_threshold=event_threshold)
                samples = np.concatenate((samples_temp, samples), axis=0)
                labels = np.concatenate((labels_temp, labels), axis=0)
            elif p.match(entry.name) is not None:
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

    # TODO check that this should be bitwise or
    # -1 & -1 == -1 only way to get -1 b/c -1 is all 1 bits
    idxcutarr = ((omegaidxarr | ptidxarr) != -1)

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

    # TODO check no nan slips thru cracks
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

    # TODO separate by label
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