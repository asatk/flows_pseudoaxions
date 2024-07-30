"""
Author: Anthony Atkinson
Modified: 2023.07.21

Provided I/O for data
"""

import numpy as np


def threshold_data(samples: np.ndarray, labels: np.ndarray, event_thresh: int=100) -> tuple[np.ndarray, np.ndarray]:
    """
    Threshold samples based on the number of events associated with each
    sample's corresponding label.
    """

    # separate samples/labels into groups per label
    labels_unique, inverse_unique = np.unique(labels, return_inverse=True,
        axis=0)
    samples_grp = [
        samples[inverse_unique == i] for i in range(len(labels_unique))
    ]

    # compile samples and labels array
    samples_new = np.zeros(shape=(0, samples.shape[-1]))
    labels_new = np.zeros(shape=(0, labels.shape[-1]))

    for i, sample_i in enumerate(samples_grp):
        label_i = labels_unique[i]
        # only include labels with sufficiently many statistics after cuts
        if len(sample_i) >= event_thresh:
            samples_new = np.r_[samples_new, sample_i]
            labels_new = np.r_[labels_new, np.repeat([label_i], len(sample_i), axis=0)]

    return samples_new, labels_new