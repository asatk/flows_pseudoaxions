"""
Author: Anthony Atkinson
Modified: 2023.07.20

I/O for .ROOT files.
"""


import numpy as np
import os
import re
import uproot as up


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