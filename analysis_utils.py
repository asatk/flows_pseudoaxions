import numpy as np
import math
from matplotlib import pyplot as plt
import multiprocessing as mp
from scipy.stats import chisquare
from typing import Any

import defs
import data_utils

plt.rcParams.update({
    "font.family": "serif"
})

def analyze(distribution: Any, made_list: list[Any], gen_labels_path: str=None) -> None:

    normdata_path = "%s/%s_normdata.npy"%(defs.data_dir, defs.data_name)
    normdatacond_path = "%s/%s_normdatacond.npy"%(defs.data_dir, defs.data_name)

    if gen_labels_path is None:
        # whiten the label/convert to normalized net-friendly form
        gen_labels = np.repeat([[2464, 5.125]], defs.ngen, axis=0) # arb. label
        normdatacond = np.load(normdata_path, allow_pickle=True).item()
        min_norm = normdatacond["min"]
        max_norm = normdatacond["max"]
        mean_norm = normdatacond["mean"]
        std_norm = normdatacond["std"]

        data_temp = (gen_labels - min_norm) / (max_norm - min_norm)
        data_temp = np.log(1 / ((1 / data_temp) - 1))
        gen_datacond = (data_temp - mean_norm) / std_norm

    else:
        gen_datacond = np.load(gen_labels_path)
        gen_labels = data_utils.unwhiten(gen_datacond, normdatacond_path)

    gen_labels_unique, gen_inverse_unique = np.unique(gen_labels, return_inverse=True, axis=0)

    # Define the conditional input (labels) for the flow to generate
    current_kwargs = {}
    for i in range(defs.nmade):
        current_kwargs[f"maf_{i}"] = {"conditional_input" : gen_datacond}

    # >>>> THIS needs to be properly fixed. Check out tutorial
    # # plot the output from intermediate flows
    # for i, d in enumerate(made_list):
    #     out = np.array(d.sample((defs.nsamp, ), bijector_kwargs=current_kwargs))
    #     plt.scatter(out[:, 0], out[:, 1], color='darkblue', s=25)

    # Generate the data given the test labels!
    gen_data = np.array(distribution.sample((gen_datacond.shape[0], ), bijector_kwargs=current_kwargs))
    
    gen_samples = data_utils.unwhiten(gen_data, normdata_path)
    gen_labels = data_utils.unwhiten(gen_datacond, normdatacond_path)

    # grouped_data = [samples[inverseunique == i] for i in range(len(labelsunique))]
    # out_paths = [outpath_i%i for i in range(len(labelsunique))]
    # out_paths_hist = [outpath_i_hist%i for i in range(len(labelsunique))]

    # # Plot scatter plots and histograms for each label
    # with mp.Pool(defs.nworkers) as pool:
    #     pool.starmap(plot_one, zip(grouped_data, labelsunique, out_paths))
    #     pool.starmap(hist_one, zip(grouped_data, labelsunique, out_paths_hist))

    outpath_all = "output/06-27_gridout/plotgen.png"
    outpath_i = "output/06-27_gridout/plotgen%03i.png"
    outpath_i_hist = "output/06-27_gridout/histgen%03i.png"

    grouped_data = [gen_samples[gen_inverse_unique == i] for i in range(len(gen_labels_unique))]
    out_paths = [outpath_i%i for i in range(len(gen_labels_unique))]
    out_paths_hist = [outpath_i_hist%i for i in range(len(gen_labels_unique))]

    # Plot scatter plots and histograms for each label
    with mp.Pool(defs.nworkers) as pool:
        pool.starmap(plot_one, zip(grouped_data, gen_labels_unique, out_paths))
        pool.starmap(hist_one, zip(grouped_data, gen_labels_unique, out_paths_hist))

    # Plot scatter plot for all data
    plot_all(gen_samples, gen_labels, outpath_all)
    # plot_all(samples, labelsunique, outpath_all)

    # Numerical analysis
    # ...

    

# Scatter plot of samples for one label
def plot_one(samples, label, outpath):

    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c="green", s=20, alpha=0.5)
    ax.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Generated Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    ax.set_xlim((defs.phi_min, defs.phi_max))
    ax.set_ylim((defs.omega_min, defs.omega_max))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    ax.grid(visible=True)
    fig.savefig(outpath)
    plt.close()

# Histogram of samples for one label
def hist_one(samples, label, outpath):
    nbinsx = 50
    nbinsy = 50
    
    fig, ax = plt.subplots()
    _, _, _, img = ax.hist2d(samples[:,0], samples[:,1], bins=(nbinsx, nbinsy), range=((defs.phi_min, defs.phi_max), (defs.omega_min, defs.omega_max)))
    plt.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Generated Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    ax.set_xlim((defs.phi_min, defs.phi_max))
    ax.set_ylim((defs.omega_min, defs.omega_max))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    fig.colorbar(img)
    fig.savefig(outpath)
    plt.close()

# Scatter plot of samples for all labels
def plot_all(samples, labels_unique, outpath):

    fig, ax = plt.subplots()

    ax.scatter(samples[:,0], samples[:,1], c="blue", s=20, alpha=0.5)
    ax.scatter(labels_unique[:, 0], labels_unique[:,1], c="orange", s=20)
    ax.set_xlim((defs.phi_min, defs.phi_max))
    ax.set_ylim((defs.omega_min, defs.omega_max))
    ax.set_title("Generated Samples N = %i"%(len(samples)))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    ax.grid(visible=True)
    fig.savefig(outpath)
    plt.show()


# training data plot
def plot_train(data, labels, path):
    fig = plt.figure(figsize=(10.0,10.0))
    ax = fig.add_subplot(111)

    ax.scatter(data[:, 0], data[:, 1], s=25, color='darkgreen', alpha=0.4, label='training data')
    ax.scatter(labels[:, 0], labels[:, 1], s=25, color='red')
    fig.legend(loc=1, fontsize=20)
    ax.grid(visible=True)
    ax.set_xlim((defs.xmin, defs.xmax))
    ax.set_ylim((defs.ymin, defs.ymax))
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("Y", fontsize=15)
    ax.set_title("Flow Training Samples", fontsize=25)
    fig.savefig(path)

# generated data plot
def plot_gen(datatrain, datagen, labelstrain, labelsgen, outputpath):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.scatter(datatrain[:, 0], datatrain[:, 1], s=25, color='darkgreen', alpha=0.4, label='training data')
    ax.scatter(datagen[:, 0], datagen[:, 1], s=25, color='darkblue', alpha=0.4, label='generated data')
    ax.scatter(labelstrain[:, 0], labelstrain[:, 1], s=25, color='red')
    ax.scatter(labelsgen[:, 0], labelsgen[:, 1], s=25, color='yellow')
    fig.legend(loc=1, fontsize=20)
    ax.grid(visible=True)
    ax.set_xlim((defs.xmin, defs.xmax))
    ax.set_ylim((defs.ymin, defs.ymax))
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("Y", fontsize=15)
    ax.set_title("Flow Output and Training Samples", fontsize=25)
    fig.savefig(outputpath)



def _gausformula(x, mean, var) -> np.ndarray:
    return np.divide(np.exp(np.multiply(-1./2 / var, np.square(np.subtract(x, mean)))),
                     np.sqrt(2 * np.pi * var))


# next let's compare two dists

def gaussinity(samples, labels, outputpath=None, name=None):

    if outputpath is None:
        outputpath = '%s_%.03f-%.03f.png'
    
    if name is None:
        name = 'DATA'

    uniquelabels, uniqueinverse = np.unique(labels, return_inverse=True)

    for i, label in enumerate(uniquelabels):
        sample = samples[uniqueinverse == i]
        # label = labels[i,:]

        mean = np.mean(sample, axis=0)
        cov = np.cov(sample.T)
        varx = cov[0, 0]
        vary = cov[1, 1]
        # print("Mean\t", mean)
        # print("Cov\t", cov)
        # print("Var_x\t", varx)
        # print("Var_y\t", vary)

        x = sample[:, 0]
        y = sample[:, 1]

        # JP says make width of data = 5 bins but also use ~20 bins?
        nbinsx = math.ceil((x.max() - x.min()) / (math.sqrt(varx) / 5.))
        # binwidthx = (x.max() - x.min()) / nbinsx
        nbinsy = math.ceil((y.max() - y.min()) / (math.sqrt(vary) / 5.))
        # binwidthy = (y.max() - y.min()) / nbinsy

        hist, xedges, yedges = np.histogram2d(x, y, bins=[nbinsx, nbinsy])

        histx = np.sum(hist, axis=1)
        histy = np.sum(hist, axis=0)

        xbincenters = xedges[:-1] + (xedges[1] - xedges[0]) / 2.
        ybincenters = yedges[:-1] + (yedges[1] - yedges[0]) / 2.

        # THIS SHOULD BE THEORY GAUSSIAN!!!

        gausx = _gausformula(xbincenters, mean[0], varx)
        gausx *= sum(histx) / sum(gausx)

        gausy = _gausformula(ybincenters, mean[1], vary) 
        gausy *= sum(histy) / sum(gausy)

        chi2x, px = chisquare(histx, gausx, ddof=1)
        chi2y, py = chisquare(histy, gausy, ddof=1)

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.hist(x, bins=xedges)
        # ax1.plot(xbincenters, histx)
        ax1.plot(xbincenters, gausx, color='red')
        ax1.errorbar(xbincenters, histx, yerr=np.sqrt(histx), ecolor='black', linestyle='None')
        ax1.tick_params('both', labelsize=15)
        ax1.set_xlabel('X', fontsize=20)
        ax1.set_ylabel('Event Counts', fontsize=20)
        ax1.set_title(name + ': X-profile of samples\nfor label %0.3f, %0.3f'%(label[0], label[1]), fontsize=25)
        strx = r'$\mu_x = ' + '%.03f$\n'%(mean[0]) + r'$\sigma_x^2 = ' + '%.05f$\n'%(varx) + \
                r'$\chi_x = ' + '%.2f$\n'%(math.sqrt(chi2x)) + r'$p_x = ' + '%.03f$'%(px)
        ax1.text(0.075, 0.8, strx, fontsize=15, transform=ax1.transAxes,
                 bbox=dict(facecolor='#f5f5dc', alpha=0.5))

        ax2.hist(y, bins=yedges)
        # ax2.plot(ybincenters, histy)
        ax2.plot(ybincenters, gausy, color='red')
        ax2.errorbar(ybincenters, histy, yerr=np.sqrt(histy), ecolor='black', linestyle='None')
        ax2.tick_params('both', labelsize=15)
        ax2.set_xlabel('Y', fontsize=20)
        ax2.set_ylabel('Event Counts', fontsize=20)
        ax2.set_title(name + ': Y-profile of samples\nfor label %0.3f, %0.3f'%(label[0], label[1]), fontsize=25)
        stry = r'$\mu_y = ' + '%.03f$\n'%(mean[1]) + r'$\sigma_y^2 = ' + '%.05f$\n'%(vary) + \
                r'$\chi_y = ' + '%.2f$\n'%(math.sqrt(chi2y)) + r'$p_y = ' + '%.03f$'%(py)
        ax2.text(0.075, 0.8, stry, fontsize=15, transform=ax2.transAxes,
                 bbox=dict(facecolor='#f5f5dc', alpha=0.5))

        fig.tight_layout()
        fig.savefig(outputpath%('hist', label[0], label[1]))