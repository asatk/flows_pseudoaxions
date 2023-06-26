import argparse    
import numpy as np

import analysis_utils as autils
import data_utils as dutils
import defs
import train_utils as tutils
import utils as myutils

if __name__ == "__main__":

    # Perform initial checks of paths
    myutils.init()

    # Make new training data and save it
    if defs.newdata:
        samples, labels = dutils.makedata(defs.mode, data_path=defs.root_dir, normalize=True)

    # Load training data
    else:
        datapath = "data/root_2023-06-24_run01.npy"
        labelpath = "data/root_labels_2023-06-24_run01.npy"
        samples = np.load(datapath)
        labels = np.load(labelpath)

    model, distribution, made_list = tutils.getmodel(defs.flow_path)

    if defs.newmodel or defs.epoch_resume != 0:
        tutils.train(model, samples, labels, defs.flow_path)

    if defs.newanalysis:
        autils.analyze()
