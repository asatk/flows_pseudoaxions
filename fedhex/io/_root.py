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

from ._data import threshold_data


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


def find_root(data_dir, max_depth: int=3):
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
                    list_subtree = find_root(data_dir + "/" + entry.name, max_depth=max_depth-1)
                    file_list.extend(list_subtree)
            elif p.match(entry.name) is not None:
                path = data_dir + "/" + entry.name
                if os.stat(path).st_size == 0:
                    print_msg(f"--- {path}: empty file ---", level=LOG_WARN)
                    return []
                return [path + ":Events;1"]
            
    return file_list


def evt_sel_1(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
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


def load_root(root_paths, event_selection_fn: Callable[[np.ndarray],
        tuple[np.ndarray]], expressions: list[str], cutstr: str,
        event_thresh: int=100, num_workers: int=1):
    """
    Load events from all provided .ROOT files using a given event selection
    scheme. 
    """
    
    arr = up.concatenate(root_paths, expressions=expressions, cut=cutstr,
        num_workers=num_workers, begin_chunk_size="2 MB", library="np")

    samples, labels = event_selection_fn(arr)

    return threshold_data(samples, labels, event_thresh=event_thresh)
