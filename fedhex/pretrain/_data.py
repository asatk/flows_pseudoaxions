"""
Author: Anthony Atkinson
Modified: 2023.07.15

Utility functions for generating, loading, and manipulating training data for
a normalizing flow.
"""


import numpy as np

from ..constants import DEFAULT_SEED, WHITEN_EPSILON
from ..utils import LOG_WARN, LOG_ERROR, LOG_FATAL, print_msg


# TODO perhaps use kwargs instead?? and save those as whiten data?
def whiten(data: np.ndarray, whiten_data: dict|None=None,
           use_logit: bool=False, epsilon: float=WHITEN_EPSILON,
           ret_dict: bool=False) -> tuple[np.ndarray, dict[str, float]]:
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
    """

    # Check whiten_data
    if whiten_data is not None:
        wt_keys = whiten_data.keys()
        for key in ["min", "max", "mean", "std"]:
            if key not in wt_keys:
                print_msg(f"The key '{key}' is not part of the whitening " + \
                    "data provided. No whitening will occur to the data " + \
                    "and `None` will be returned for the whitening data dict.",
                    level=LOG_ERROR)
                return (data, None)

    if whiten_data is not None:
        min_norm = whiten_data["min"]
        max_norm = whiten_data["max"]
    else:
        
        min_norm = np.min(data, axis=0)
        max_norm = np.max(data, axis=0)
        
        # Add tiny constant to prevent nan/inf when using logit/exponential
        if use_logit:
            min_norm -= epsilon
            max_norm += epsilon

    # confine data to the unit interval [0., 1.]
    data_unit = (data - min_norm) / (max_norm - min_norm)

    # apply 'logit' transform to map unit interval to (-inf, +inf)
    if use_logit:
        data_unit = np.log(1 / ((1 / data_unit) - 1))

    if whiten_data is not None:
        mean_norm = whiten_data["mean"]
        std_norm = whiten_data["std"]
    else:
        mean_norm = np.mean(data_unit, axis=0)
        std_norm = np.std(data_unit, axis=0)
        whiten_data = {"min": min_norm, "max": max_norm, "mean": mean_norm,
                       "std": std_norm, "epsilon": epsilon}

    # standardize the data to have 0 mean and unit variance
    data_norm = (data_unit - mean_norm) / std_norm
 
    if ret_dict:
        return data_norm, whiten_data
    return data_norm


def dewhiten(data_norm: np.ndarray, whiten_data: dict, use_logit: bool=False) -> np.ndarray:
    # TODO update docstring
    """
    Invert the standardized data that were output from the network into values
    that are interpretable to the end user. The inverse transformation must use
    the same min/max/mean/std values that were used to first whiten the data
    for an accurate representation of the samples.

    data: np.ndarray
        raw N-D coordinates that have not been pre-processed
    """


    mean_norm = whiten_data["mean"]
    std_norm = whiten_data["std"]
    min_norm = whiten_data["min"]
    max_norm = whiten_data["max"]    
    

    # invert the standardized values from the mean and std
    data_unit = data_norm * std_norm + mean_norm

    # apply inverse 'logit' transform to map from (-inf, +inf) to unit interval
    if use_logit:
        data_unit = 1 / (1 + np.exp(-data_unit))
    
    # shift data from unit interval back to its intended interval
    data = data_unit * (max_norm - min_norm) + min_norm

    return data
