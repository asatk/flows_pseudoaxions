"""
Author: Anthony Atkinson
Created: 08.24.23
Modified: 08.24.23

Loading datasets from numpy
"""

import numpy as np
from numpy import ndarray

from ._data import threshold_data

def load_numpy(path: str, event_thresh: int=100) -> tuple[ndarray, ndarray]:
    loaded_data = np.load(path)
    samples = loaded_data[:,:,0]
    labels = loaded_data[:,:,1]
    
    return threshold_data(samples, labels, event_thresh=event_thresh)
