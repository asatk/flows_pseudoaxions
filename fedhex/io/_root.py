"""
Author: Anthony Atkinson
Modified: 2024.01.17

I/O for .ROOT files.
"""


import numpy as np
import os
import re

import ROOT
ROOT.EnableImplicitMT(numthreads=0)
RDataFrame = ROOT.RDataFrame

from ..utils import print_msg, LOG_WARN
from ._data import threshold_data


def find_root(data_dir: str, max_depth: int=3):
    """
    Locate the paths to all of the .ROOT files in a given directory. This
    function is recursive and will only descend the number of sub directories
    as given by `max_depth`. Appends the name of the TTree given by `tree_name`
    to the end of the path.

    data_dir: str, directory being searched for .ROOT files.
    max_depth: str, the maximum depth searched to from the top directory
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
                else:
                    file_list.append(path)
            
    return file_list


def load_root(root_dir: str|list[str],
              tree_name: str,
              data_vars: list[str],
              cond_vars: list[str],
              defs: dict[str, str],
              cutstr: str,
              event_thresh: int=100,
              max_depth: int=3) -> tuple[np.ndarray, np.ndarray]:
    """
    Load events from all provided .ROOT files using a given event selection
    scheme. 

    root_dir: str or list[str], either the location of a directory on the file
     system or list of .ROOT file paths.
    tree_name: str, name of TTree in TFile
    data_vars: list[str], names of variables for the data. The array returned
     has columns in the order that the names are listed.
    cond_vars: list[str], names of variables for the conditional data. The
     array returned has columns in the order that the names are listed.
    defs: dict[str, str], pairs of aliases and expressions (algebraic or C++ syntax)
    cutstr: str, string defining cuts on data
    event_thresh: int, number of events required for a given label to be represented in data
    max_depth: int, number of directory levels traversed until the search for root files is stopped

    Returns: tuple[np.ndarray, np.ndarray], tuple containing the array of
     samples and labels, the samples first and labels second.
    """

    # Collect the paths of ROOT files used in analysis
    if isinstance(root_dir, str):
        root_paths = find_root(root_dir, max_depth=max_depth)
    elif isinstance(root_dir, list):
        root_paths = root_dir
    else:
        return (None, None)

    # Load data from the files using the given tree name
    df = RDataFrame(tree_name, root_paths)
    if cutstr is not None:
        df = df.Filter(cutstr)
    
    # Define an alias for each expression defined in defs
    for key, exp in defs.items():
        df = df.Define(key, exp)
    
    # Convert RDF to dict of numpy arrays, one for each variable name
    both_vars = data_vars + cond_vars
    arr_np = df.AsNumpy(both_vars)

    # Construct array of data in order of listed variable names
    list_data = []
    for var_name in data_vars:
        list_data.append(arr_np[var_name])
    # Make array of events such that each event is contiguous in memory
    arr_data = np.array(list_data, copy=True, order="F").T
    

    # Construct array of conditional data in order of listed label names
    list_cond = []
    for var_name in cond_vars:
        list_cond.append(arr_np[var_name])
    # Make array of labels such that each label is contiguous in memory
    arr_cond = np.array(list_cond, copy=True, order="F").T

    # Require the count of data associated with each label to reach a threhsold
    if event_thresh > 0:
        threshold_data(arr_data, arr_cond, event_thresh=event_thresh)

    return arr_data, arr_cond


def save_root(path: str,
              tree_name: str,
              gen_samples: np.ndarray=None,
              gen_labels: np.ndarray=None,
              trn_samples: np.ndarray=None,
              trn_labels: np.ndarray=None) -> bool:
    """
    Save a collection of generated and training data

    path: str, file system location to which the data are saved
    tree_name: str, name of the TTree where data are saved
    gen_samples: np.ndarray=None, generated samples to be saved
    gen_labels: np.ndarray=None, generated labels to be saved
    trn_samples: np.ndarray=None, training samples to be saved
    trn_labels: np.ndarray=None, training labels to be saved

    Returns: bool, True if the save is successful, False if the data were not
    saved because of missing parameters.
    """
    
    if not path.lower().endswith(".root"):
        path = path + "root_out.root"
    
    labels_count = 0

    save_gen = gen_samples is not None and gen_labels is not None
    save_trn = trn_samples is not None and trn_labels is not None

    if not save_gen and not save_trn:
        return False
    
    # Hard-coded labels of output
    s_numLabels = "numLabels"
    s_isGen = "isGen"
    s_labelIndex = "labelIndex"
    s_data = "data"
    s_label = "label"
    cols = [s_numLabels, s_isGen, s_labelIndex, s_data, s_label]

    #If generated data is in data dict, write it to the root file
    if save_gen:
        #Get unique generated labels
        gen_labels_unique = np.unique(gen_labels, axis=0)

        ngensamples = len(gen_samples)
        dimsample = gen_samples.shape[-1]
        ngenlabels = len(gen_labels_unique)
        dimlabel = gen_labels.shape[-1]
        dict_gen_labels = {
            str(label): index
            for index, label in enumerate(gen_labels_unique)
        }
        
        arr_numLabels = dimlabel * np.ones(ngensamples)
        arr_isGen = np.ones(ngensamples)
        arr_labelIndex = np.empty(ngensamples)

        # Iterate through each generated event's data
        for i in range(ngensamples):
            arr_labelIndex[i] = dict_gen_labels[str(gen_labels[i])]

        labels_count = ngenlabels

    #If training data is in data dict, write it to the root file
    if save_trn:
        
        # Get unique training labels
        trn_labels_unique = np.unique(trn_labels, axis=0)

        ntrnsamples = len(trn_samples)

        if save_gen:
            if trn_samples.shape[-1] != dimsample:
                print(f"Dimension of training data <{trn_samples.shape[-1]}> does not match "\
                    f"dimension of generated data <{dimsample}>.")
                return False
            if trn_labels.shape[-1] != dimlabel:
                print(f"Dimension of training labels <{trn_labels.shape[-1]}> does not match "\
                    f"dimension of generated labels <{dimlabel}>.")
                return False
        else:
            dimsample = trn_samples.shape[-1]
            dimlabel = trn_labels.shape[-1]
        
        dict_trn_labels = {
            str(label): index + labels_count
            for index, label in enumerate(trn_labels_unique)
        }

        trn_numLabels = dimlabel * np.ones(ntrnsamples)
        trn_isGen = np.zeros(ntrnsamples)
        trn_labelIndex = np.empty(ntrnsamples)

        if save_gen:
            arr_numLabels = np.concatenate((trn_numLabels, arr_numLabels))
            arr_isGen = np.concatenate((trn_isGen, arr_isGen))
            arr_labelIndex = np.concatenate((trn_labelIndex, arr_labelIndex))
            arr_data = np.concatenate((trn_samples, gen_samples))
            arr_label = np.concatenate((trn_labels, gen_labels))
        else:
            arr_numLabels = trn_numLabels
            arr_isGen = trn_isGen
            arr_labelIndex = trn_labelIndex
            arr_data = trn_samples
            arr_label = trn_labels
        
        # Iterate through each training event's data
        for i in range(ntrnsamples):
            arr_labelIndex[i] = dict_trn_labels[str(trn_labels[i])]

    df_data = {
        s_numLabels: arr_numLabels,
        s_isGen: arr_isGen,
        s_labelIndex: arr_labelIndex
    }

    s_datadef = "ROOT::RVec<double>{"
    for i in range(dimsample):
        df_data.update({f"data{i}": arr_data[:,i]})
        s_datadef += f"data{i},"
    s_datadef = s_datadef[:-1] + "}" # replace trailing ',' with closing '}'

    s_labeldef = "ROOT::RVec<double>{"
    for i in range(dimlabel):
        df_data.update({f"label{i}": arr_label[:,i]})
        s_labeldef += f"label{i},"
    s_labeldef = s_labeldef[:-1] + "}" # replace trailing ',' with closing '}'

    rdf = ROOT.RDF.FromNumpy(df_data)
    rdf = rdf.Define(s_data, s_datadef)
    rdf = rdf.Define(s_label, s_labeldef)
    rdf.Snapshot(tree_name, path, cols)

    return True
    