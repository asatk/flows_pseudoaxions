import math
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import multiprocessing as mp
import numpy as np
import os
from scipy.stats import chisquare
from typing import Any

import defs
import data_utils
import flowmodel

plt.rcParams.update({
    "font.family": "serif"
})

def analyze(distribution: Any, made_list: list[Any], gen_labels_path: str=None, losslog: str=None) -> None:

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

        # manually whiten data from loaded params to convert a meaningful label
        # into a regression label that the network will interpret correctly.
        data_temp = (gen_labels - min_norm) / (max_norm - min_norm)
        data_temp = np.log(1 / ((1 / data_temp) - 1))
        gen_datacond = (data_temp - mean_norm) / std_norm

    else:
        gen_datacond = np.load(gen_labels_path)
        gen_labels = data_utils.unwhiten(gen_datacond, normdatacond_path)

    gen_labels_unique, gen_inverse_unique = np.unique(gen_labels, return_inverse=True, axis=0)

    # Define the conditional input (labels) for the flow to generate
    current_kwargs = {}
    for i in range(len(made_list) // 2):
        current_kwargs[f"maf_{i}"] = {"conditional_input" : gen_datacond}

    # Generate the data given the test labels!
    gen_data = np.array(distribution.sample((gen_datacond.shape[0], ), bijector_kwargs=current_kwargs))
    
    gen_samples = data_utils.unwhiten(gen_data, normdata_path)
    gen_labels = data_utils.unwhiten(gen_datacond, normdatacond_path)

    out_path_all = defs.output_dir + "plotgen.png"
    out_path_i = defs.output_dir + "plotgen%03i.png"
    out_path_i_hist = defs.output_dir + "histgen%03i.png"
    out_path_flow_i = defs.output_dir + "plotgen_flow%02i.png"

    # >>> plot generated data
    grouped_samples = [gen_samples[gen_inverse_unique == i] for i in range(len(gen_labels_unique))]
    out_paths = [out_path_i%i for i in range(len(gen_labels_unique))]
    out_paths_hist = [out_path_i_hist%i for i in range(len(gen_labels_unique))]

    # Plot scatter plots and histograms for each label
    with mp.Pool(defs.nworkers) as pool:
        pool.starmap(plot_one, zip(grouped_samples, gen_labels_unique, out_paths))
        pool.starmap(hist_one, zip(grouped_samples, gen_labels_unique, out_paths_hist))
        pool.starmap(sample_stats, zip(grouped_samples, gen_labels_unique))
    # >>> plot generated data

    # Plot scatter plot for all data
    plot_all(gen_samples, gen_labels, out_path_all, show=True)
    
    # Generate samples for each intermediate flow
    flow_distributions = flowmodel.intermediate_flows_chain(made_list)
    for i, dist in enumerate(flow_distributions):
        gen_data_flow_i = dist.sample((gen_datacond.shape[0], ), bijector_kwargs=current_kwargs)
        gen_samples_flow_i = data_utils.unwhiten(gen_data_flow_i, normdata_path)
        title_flow_i = "Generated Output (N = %i) up to Flow %i"%(gen_datacond.shape[0], i)
        plot_all(gen_samples_flow_i, gen_labels_unique, out_path_flow_i%i, title=title_flow_i)

    # Plot losses during training (first epoch exlucded b/c it usually has very high loss)
    losses = np.load(losslog)
    loss_out_path = defs.output_dir + "/loss.png"
    plot_losses(losses[1:], loss_out_path=loss_out_path, show=True)

    # Numerical analysis
    # ...

def analyze_train_data(samples_path, labels_path, normdata_path, normdatacond_path):

    # Load unwhitened training data
    samples = data_utils.unwhiten(np.load(samples_path), normdata_path)
    labels = data_utils.unwhiten(np.load(labels_path), normdatacond_path)

    # Group samples & labels by unique label
    labelsunique, inverseunique = np.unique(labels, return_inverse=True, axis=0)

    out_path_all = defs.output_dir + "plotreal.png"
    out_path_i = defs.output_dir + "plotreal%03i.png"
    out_path_i_hist = defs.output_dir + "histreal%03i.png"
    
    # >>> plot training data
    grouped_data = [samples[inverseunique == i] for i in range(len(labelsunique))]
    out_paths = [out_path_i%i for i in range(len(labelsunique))]
    out_paths_hist = [out_path_i_hist%i for i in range(len(labelsunique))]

    # Plot scatter plots and histograms for each label
    with mp.Pool(defs.nworkers) as pool:
        pool.starmap(plot_one, zip(grouped_data, labelsunique, out_paths))
        pool.starmap(hist_one, zip(grouped_data, labelsunique, out_paths_hist))
    # >>> plot training data

    plot_all(samples, labelsunique, out_path_all)


def sample_stats(data: np.ndarray, data_cond: np.ndarray) -> None:
    '''
    Print some basic statistics of the sample data corresponding to one label

    data: samples
    data_cond: labels
    '''

    sample_mean = np.mean(data, axis=0)
    sample_median = np.median(data, axis=0)
    sample_std = np.std(data, axis=0)
    sample_min = np.min(data, axis=0)
    sample_max = np.max(data, axis=0)

    print_str = "\n----------\nLABEL: %s"%repr(data_cond) + \
                "\nnum events: %i"%data.shape[0] + \
                "\nsample mean: %s"%repr(sample_mean) + \
                "\nsample median: %s"%repr(sample_median) + \
                "\nsample std dev: %s"%repr(sample_std) + \
                "\nsample min: %s"%repr(sample_min) + \
                "\nsample max: %s"%repr(sample_max)
    
    print(print_str)


def plot_losses(losses, loss_out_path: str=None, show=False, **kwargs):
    '''
    Plot loss (negative-log likelihood) of network over time (epochs)
    '''
    
    # Parse keyword arguments
    kwkeys = kwargs.keys()
    title = "Loss vs. Epoch" if "title" not in kwkeys else kwargs["title"]
    xlabel = "Epoch" if "xlabel" not in kwkeys else kwargs["xlabel"]
    ylabel = "Loss (Negative Log Likelihood)" if "ylabel" not in kwkeys else kwargs["ylabel"]

    fig, ax = plt.subplots()
    ax.plot(losses[:,0], losses[:, 1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.get_major_locator().set_params(integer=True)

    if loss_out_path is not None:
        fig.savefig(loss_out_path)

    if show:
        plt.show()
    else:
        plt.close()

def plot_one(samples, label, outpath, **kwargs):
    '''
    Scatter plot of samples corresponding to a single label
    '''

    # Parse keyword arguments
    kwkeys = kwargs.keys()
    title = "Generated Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], len(samples)) if "title" not in kwkeys else kwargs["title"]
    xlabel = "Reconstructed $\Phi$ Mass (GeV)" if "xlabel" not in kwkeys else kwargs["xlabel"]
    ylabel = "Reconstructed $\omega$ Mass (GeV)" if "ylabel" not in kwkeys else kwargs["ylabel"]
    label_c = "red" if "label_c" not in kwkeys else kwargs["label_c"]
    label_s = 20 if "label_s" not in kwkeys else kwargs["label_s"]
    label_alpha = 1.0 if "label_alpha" not in kwkeys else kwargs["label_alpha"]
    sample_c = "green" if "sample_c" not in kwkeys else kwargs["sample_c"]
    sample_s = 20 if "sample_s" not in kwkeys else kwargs["sample_s"]
    sample_alpha = 0.5 if "sample_alpha" not in kwkeys else kwargs["sample_alpha"]
    xlim = (defs.phi_min, defs.phi_max) if "xlim" not in kwkeys else kwargs["xlim"]
    ylim = (defs.omega_min, defs.omega_max) if "ylim" not in kwkeys else kwargs["ylim"]

    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c=sample_c, s=sample_s, alpha=sample_alpha)
    ax.scatter(label[0], label[1], c=label_c, s=label_s, alpha=label_alpha)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True)
    fig.savefig(outpath)
    plt.close()     #plots are closed immediately to save memory


def hist_one(samples, label, outpath, **kwargs):
    '''
    Scatter plot of samples corresponding to a single label
    '''
    
    # Parse keyword arguments
    kwkeys = kwargs.keys()
    title = "Generated Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], len(samples)) if "title" not in kwkeys else kwargs["title"]
    xlabel = "Reconstructed $\Phi$ Mass (GeV)" if "xlabel" not in kwkeys else kwargs["xlabel"]
    ylabel = "Reconstructed $\omega$ Mass (GeV)" if "ylabel" not in kwkeys else kwargs["ylabel"]
    label_c = "red" if "label_c" not in kwkeys else kwargs["label_c"]
    label_s = 20 if "label_s" not in kwkeys else kwargs["label_s"]
    label_alpha = 1.0 if "label_alpha" not in kwkeys else kwargs["label_alpha"]
    xlim = (defs.phi_min, defs.phi_max) if "xlim" not in kwkeys else kwargs["xlim"]
    ylim = (defs.omega_min, defs.omega_max) if "ylim" not in kwkeys else kwargs["ylim"]

    nbinsx = 50
    nbinsy = 50
    
    fig, ax = plt.subplots()
    _, _, _, img = ax.hist2d(samples[:,0], samples[:,1], bins=(nbinsx, nbinsy), range=(xlim, ylim))
    plt.scatter(label[0], label[1], c=label_c, s=label_s, alpha=label_alpha)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(img)
    fig.savefig(outpath)
    plt.close()     #plots are closed immediately to save memory


def plot_all(samples, labels_unique, outpath, show=False, **kwargs):
    '''
    Scatter plot of samples for all labels
    '''

    # Parse keyword arguments
    kwkeys = kwargs.keys()
    title = "Generated Samples N = %i"%(samples.shape[0]) if "title" not in kwkeys else kwargs["title"]
    xlabel = "Reconstructed $\Phi$ Mass (GeV)" if "xlabel" not in kwkeys else kwargs["xlabel"]
    ylabel = "Reconstructed $\omega$ Mass (GeV)" if "ylabel" not in kwkeys else kwargs["ylabel"]
    label_c = "red" if "label_c" not in kwkeys else kwargs["label_c"]
    label_s = 20 if "label_s" not in kwkeys else kwargs["label_s"]
    label_alpha = 1.0 if "label_alpha" not in kwkeys else kwargs["label_alpha"]
    sample_c = "green" if "sample_c" not in kwkeys else kwargs["sample_c"]
    sample_s = 20 if "sample_s" not in kwkeys else kwargs["sample_s"]
    sample_alpha = 0.5 if "sample_alpha" not in kwkeys else kwargs["sample_alpha"]
    xlim = (defs.phi_min, defs.phi_max) if "xlim" not in kwkeys else kwargs["xlim"]
    ylim = (defs.omega_min, defs.omega_max) if "ylim" not in kwkeys else kwargs["ylim"]

    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c=sample_c, s=sample_s, alpha=sample_alpha)
    ax.scatter(labels_unique[:, 0], labels_unique[:,1], c=label_c, s=label_s, alpha=label_alpha)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True)
    fig.savefig(outpath)
    
    if show:
        plt.show()
    else:
        plt.close()

def _gausformula(x, mean, var) -> np.ndarray:
    return np.divide(np.exp(np.multiply(-1./2 / var, np.square(np.subtract(x, mean)))),
                     np.sqrt(2 * np.pi * var))


# next let's compare two dists. chi^2 or KL divergence

def gaussinity(samples, labels, outputpath=None, name=None):

    if outputpath is None:
        outputpath = "%s_%.03f-%.03f.png"
    
    if name is None:
        name = "DATA"

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
        ax1.plot(xbincenters, gausx, color="red")
        ax1.errorbar(xbincenters, histx, yerr=np.sqrt(histx), ecolor="black", linestyle="None")
        ax1.tick_params("both", labelsize=15)
        ax1.set_xlabel("X", fontsize=20)
        ax1.set_ylabel("Event Counts", fontsize=20)
        ax1.set_title(name + ": X-profile of samples\nfor label %0.3f, %0.3f"%(label[0], label[1]), fontsize=25)
        strx = r"$\mu_x = " + "%.03f$\n"%(mean[0]) + r"$\sigma_x^2 = " + "%.05f$\n"%(varx) + \
                r"$\chi_x = " + "%.2f$\n"%(math.sqrt(chi2x)) + r"$p_x = " + "%.03f$"%(px)
        ax1.text(0.075, 0.8, strx, fontsize=15, transform=ax1.transAxes,
                 bbox=dict(facecolor="#f5f5dc", alpha=0.5))

        ax2.hist(y, bins=yedges)
        # ax2.plot(ybincenters, histy)
        ax2.plot(ybincenters, gausy, color="red")
        ax2.errorbar(ybincenters, histy, yerr=np.sqrt(histy), ecolor="black", linestyle="None")
        ax2.tick_params("both", labelsize=15)
        ax2.set_xlabel("Y", fontsize=20)
        ax2.set_ylabel("Event Counts", fontsize=20)
        ax2.set_title(name + ": Y-profile of samples\nfor label %0.3f, %0.3f"%(label[0], label[1]), fontsize=25)
        stry = r"$\mu_y = " + "%.03f$\n"%(mean[1]) + r"$\sigma_y^2 = " + "%.05f$\n"%(vary) + \
                r"$\chi_y = " + "%.2f$\n"%(math.sqrt(chi2y)) + r"$p_y = " + "%.03f$"%(py)
        ax2.text(0.075, 0.8, stry, fontsize=15, transform=ax2.transAxes,
                 bbox=dict(facecolor="#f5f5dc", alpha=0.5))

        fig.tight_layout()
        fig.savefig(outputpath%("hist", label[0], label[1]))