"""
Author: Anthony Atkinson
Modified: 2023.07.21

Provided I/O for data
"""

import numpy as np
import os

from ..utils import LOG_WARN, print_msg

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

    np.save(data_dict_path, training_data, allow_pickle=True)