from datetime import datetime
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np

import analysis_utils
import defs
import data_utils

def makeplot(samples, label, outpath):

    fig, ax = plt.subplots()

    ax.scatter(samples[:,0], samples[:,1], c="green", s=20, alpha=0.5)
    ax.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Samples for (%f, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    ax.set_xlim((0., 7000.))
    ax.set_ylim((0., 15.))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    ax.grid(visible=True)
    fig.savefig(outpath)
    plt.close()

def makehist(samples, label, outpath):

    fig, ax = plt.subplots()

    nbinsx = 50
    nbinsy = 50

    h, xedges, yedges, img = ax.hist2d(samples[:,0], samples[:,1], bins=(nbinsx, nbinsy), range=((0., 7.e3), (0., 15.)))
    plt.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Samples for (%f, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    ax.set_xlim((0., 7000.))
    ax.set_ylim((0., 15.))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    fig.colorbar(img)
    fig.savefig(outpath)
    plt.close()


if __name__ == "__main__":
    loadnew = False
    runnum = 1

    datadir = "./root/10x10box_10k_jun2023"
    outpath = "./output/temp/plotroot.png"
    outpath_i = "./output/temp/plotroot_dist%03i.png"
    outpath_i_hist = "./output/temp/histroot_dist%03i.png"

    labelpath = "data/root_labels_2023-06-24_run01.npy"
    datapath = "data/root_2023-06-24_run01.npy"
    normdatapath = "data/root_norm_2023-06-24_run01.npy"

    if loadnew:
        samples, labels = data_utils._loadallroot(datadir)
        np.save(datapath, samples)
        np.save(labelpath, labels)
    else:
        samples = np.load(datapath)
        labels = np.load(labelpath)
        normdata = np.load(normdatapath)
        mean = normdata[0]
        std = normdata[1]
        samples = (samples * std) + mean

    labelsunique, inverseunique = np.unique(labels, return_inverse=True, axis=0)

    plt.rcParams.update({
        "font.family": "serif"
    })

    fig, ax = plt.subplots()

    ax.scatter(samples[:,0], samples[:,1], c="green", s=20, alpha=0.5)
    ax.scatter(labelsunique[:,0], labelsunique[:,1], c="red", s=20)
    ax.set_title("All Samples N = %i"%(len(samples)))
    ax.set_xlim((0., 7000.))
    ax.set_ylim((0., 15.))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    ax.grid(visible=True)
    fig.savefig(outpath)
    plt.close()


    for i, label in enumerate(labelsunique):
        samples_i = samples[inverseunique == i]
        makeplot(samples_i, label, outpath_i%i)
        makehist(samples_i, label, outpath_i_hist%i)
