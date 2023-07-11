'''
Author: Anthony Atkinson
Modified: 2023.07.07
'''


import math
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import multiprocessing as mp
import numpy as np
from numpy.typing import ArrayLike
import os
from scipy.stats import chisquare, kstest
from typing import Any, Union

import defs
import data_utils
import flowmodel

plt.rcParams.update({
    "font.family": "serif"
})

ntools = 5

def analyze(distribution: Any, made_list: list[Any], normdata_path: str,
            normcond_path: str, trn_data_path: str, trn_cond_path: str,
            gen_data_path: str=None, gen_cond_path: str=None,
            losslog: str=None, tools: list[int]=None) -> None:

    # Selection of tools used in analysis
    if tools is None:
        tools = list[range(0, ntools + 1)]

    # Plotting paths
    out_trn = defs.output_dir + "plottrn.png"
    out_trn_i = defs.output_dir + "plottrn%03i.png"
    out_trn_hist_i = defs.output_dir + "histtrn%03i.png"
    out_gen = defs.output_dir + "plotgen.png"
    out_gen_i = defs.output_dir + "plotgen%03i.png"
    out_gen_hist_i = defs.output_dir + "histgen%03i.png"
    out_flow_i = defs.output_dir + "plotgen_flow%02i.png"

    # Make sure that the normalizing data are stored in agreement w/ convention
    if not os.path.isfile(normdata_path):
        print("Cannot locate the normalizing constants for data"+\
              " located at `%s`. Do not separate this file from the rest of"+\
              " the data it is generated with."%(normdata_path))
        return

    if not os.path.isfile(normcond_path):
        print("Cannot locate the normalizing constants for conditional data"+\
              " located at `%s`. Do not separate this file from the rest of"+\
              " the data it is generated with."%(normcond_path))
        return
    

    # ++++ Prepare Training Data ++++ #
    # Load dewhitened training data
    samples = data_utils.dewhiten(np.load(trn_data_path), normdata_path)
    labels = data_utils.dewhiten(np.load(trn_cond_path), normcond_path)

    # Group samples & labels by unique label
    labels_unique, inverse_unique = np.unique(labels, return_inverse=True, axis=0)
    # ++++ Prepare Training Data ++++ #
    

    # ++++ Prepare Generated Data ++++ #
    # Obtain both net- and user-friendly versions of the conditional data
    if gen_cond_path is None:
        # This is a hardcoded label just to serve as an example.
        gen_labels = np.repeat([[2464, 5.125]], defs.ngen, axis=0) # arb. label
        gen_cond = data_utils.whiten(gen_labels, normcond_path, load_norm=True)
    else:
        # Load conditional data that is already whitened from file. These data
        # need not be the  conditional data with which the network is trained.
        gen_cond = np.load(gen_cond_path)
        gen_labels = data_utils.dewhiten(gen_cond, normcond_path)

    # Make a list of the unique labels and a list of their occurrence in gen_labels
    gen_labels_unique, gen_inverse_unique = np.unique(gen_labels, return_inverse=True, axis=0)

    # Generate net-friendly version of the data conditioned on provided labels
    if gen_data_path is None:
        # Define the conditional input for the flow to generate samples with at each flow
        current_kwargs = {}
        for i in range(len(made_list) // 2):
            current_kwargs[f"maf_{i}"] = {"conditional_input" : gen_cond}

        # Generate the data conditioned on the net-friendly labels
        gen_data = np.array(distribution.sample((gen_cond.shape[0], ), bijector_kwargs=current_kwargs))
    # Load net-friendly version of the data
    else:
        gen_data = np.load(gen_data_path)
    
    # Transform generated data from net-friendly to user-friendly
    gen_samples = data_utils.dewhiten(gen_data, normdata_path)
    # ++++ Prepare Generated Data ++++ #


    ### TOOL 1 - Plot Training Data and Network Output
    if 1 in tools:
        # ++++ Plot training data ++++ #
        grouped_data = [samples[inverse_unique == i] for i in range(len(labels_unique))]
        out_list = [out_trn_i%i for i in range(len(labels_unique))]
        out_list_hist = [out_trn_hist_i%i for i in range(len(labels_unique))]
        trnplot_kwargs = make_genplot_kwargs(labels_unique, len(samples))

        # Plot scatter plots and histograms for each label
        with mp.Pool(defs.nworkers) as pool:
            pool.starmap(plot_one, zip(grouped_data, labels_unique, out_list, trnplot_kwargs))
            pool.starmap(hist_one, zip(grouped_data, labels_unique, out_list_hist, trnplot_kwargs))
        # ++++ Plot training data ++++ #


        # ++++ Plot generated data ++++ #
        grouped_samples = [gen_samples[gen_inverse_unique == i] for i in range(len(gen_labels_unique))]
        out_list = [out_gen_i%i for i in range(len(gen_labels_unique))]
        out_list_hist = [out_gen_hist_i%i for i in range(len(gen_labels_unique))]
        genplot_kwargs = make_genplot_kwargs(gen_labels_unique, len(gen_samples))

        # Plot scatter plots and histograms for each label
        with mp.Pool(defs.nworkers) as pool:
            pool.starmap(plot_one, zip(grouped_samples, gen_labels_unique, out_list, genplot_kwargs))
            pool.starmap(hist_one, zip(grouped_samples, gen_labels_unique, out_list_hist, genplot_kwargs))
            # pool.starmap(print_stats, zip(grouped_samples, gen_labels_unique))
        # ++++ Plot generated data ++++ #

        # Plot scatter plot for all training and generated data
        plot_all(samples, labels_unique, out_trn, show=True)
        plot_all(gen_samples, gen_labels, out_gen, show=True)
    

    ### TOOL 2 - Plot Intermediate Output (assess each bijector's contribution)
    if 2 in tools:
        # Generate samples for each intermediate flow
        flow_distributions = flowmodel.intermediate_flows_chain(made_list)
        for i, dist in enumerate(flow_distributions):
            gen_data_flow_i = dist.sample((gen_cond.shape[0], ), bijector_kwargs=current_kwargs)
            gen_samples_flow_i = data_utils.dewhiten(gen_data_flow_i, normdata_path)
            title_flow_i = "Generated Output (N = %i) up to Flow %i"%(gen_cond.shape[0], i)
            plot_all(gen_samples_flow_i, gen_labels_unique, out_flow_i%i, title=title_flow_i)


    ### TOOL 3 - Track Training Losses (assess ability to converge)
    if 3 in tools:
        # Plot losses during training
        losses = np.load(losslog)
        loss_out_path = defs.output_dir + "/loss.png"
        plot_losses(losses, loss_out_path=loss_out_path, show=True)


    ### TOOL 4 - Histograms
    if 4 in tools:
        # Projections
        axes = [0]  # project along x-axis/phi-axis
        bins = [50] # bin along axis of interest
        ndim = 1    # dimension of space into which the samples are projected
        samples_h = project(samples, axes=axes, bins=bins, ndim=ndim)
        gen_samples_h = project(gen_samples, axes=axes, bins=bins, ndim=ndim)
        
        # Residuals
        res = residual(gen_samples, samples, axes=axes, bins=bins)
        

    ### TOOL 5 - Numerical Fit Tests
    if 5 in tools:
        # Chi^2 test
        axes = [0]
        bins = [50]
        ndim = 1
        test_chi(gen_samples, samples, axes=axes, bins=bins, ndim=ndim)
        # Kolmogorov-Smirnov test
    

def make_trnplot_kwargs(labels: np.ndarray, nsamples: int) -> list[dict]:
    '''
    labels: conditional data that will be plotted
    '''

    labels_u = np.unqiue(labels)
    nlabels = len(labels_u)
    
    d = {
        "xlabel": "Reconstructed $\Phi$ Mass (GeV)",
        "ylabel": "Reconstructed $\omega$ Mass (GeV)",
        "label_c": "red",
        "label_s": 20,
        "label_alpha": 1.0,
        "sample_c": "green",
        "sample_s": 20,
        "sample_alpha": 0.5,
        "xlim": (defs.phi_min, defs.phi_max),
        "ylim": (defs.omega_min, defs.omega_max)
    }

    d_list = [d] * nlabels
    for label, label_dict in zip(labels_u, d_list):
        title = "Training Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], nsamples)
        label_dict.update({"title": title})
    
    return d_list


def make_genplot_kwargs(labels: np.ndarray, nsamples: int) -> list[dict]:
    '''
    labels: conditional data that will be plotted
    '''

    labels_u = np.unqiue(labels)
    nlabels = len(labels_u)
    
    d = {
        "xlabel": "Reconstructed $\Phi$ Mass (GeV)",
        "ylabel": "Reconstructed $\omega$ Mass (GeV)",
        "label_c": "red",
        "label_s": 20,
        "label_alpha": 1.0,
        "sample_c": "green",
        "sample_s": 20,
        "sample_alpha": 0.5,
        "xlim": (defs.phi_min, defs.phi_max),
        "ylim": (defs.omega_min, defs.omega_max)
    }

    d_list = [d] * nlabels
    for label, label_dict in zip(labels_u, d_list):
        title = "Generated Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], nsamples)
        label_dict.update({"title": title})
    
    return d_list


def print_stats(data: np.ndarray, cond: np.ndarray) -> None:
    '''
    Print some basic statistics of the sample data corresponding to one label

    data: dewhitened data
    cond: dewhitened conditional data
    '''

    sample_mean = np.mean(data, axis=0)
    sample_median = np.median(data, axis=0)
    sample_std = np.std(data, axis=0)
    sample_min = np.min(data, axis=0)
    sample_max = np.max(data, axis=0)

    print_str = "\n----------\nLABEL: %s"%repr(cond) + \
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

    # Separate loss data into positive and negative losses
    losses_nonneg = losses[losses[:, 1] >= 0]
    losses_neg = np.abs(losses[losses[:, 1] < 0])
    turnover_epoch = losses[losses[:, 1] < 0][0, 0]

    # Plot losses
    fig, ax = plt.subplots()
    ax.semilogy(losses_nonneg[:, 0], losses_nonneg[:, 1], c="blue", label="positive loss")
    ax.semilogy(losses_neg[:, 0], losses_neg[:, 1], c="red", label="negative loss")
    ax.vlines(turnover_epoch, ymin=np.min(losses[:, 1]), ymax=np.max(losses[:,1 ]), colors=["gray"], linestyles="dashed")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
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
    samples: 2D coordinate pairs of events in parameter space
    label: 2D coordinate pair of label associated to events
    outpath: location to where the plot is saved
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
    samples: 2D coordinate pairs of events in parameter space
    label: 2D coordinate pair of label associated to events
    outpath: location to where the plot is saved
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
    samples: 2D coordinate pairs of events in parameter space
    labels: 2D coordinate pairs of all labels associated to events
    outpath: location to where the plot is saved
    show: flag indicating whether or not to show plot after generating it
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

def residual(p: np.ndarray, q: np.ndarray, axes: list[int], bins: list[int]):
    p_h, _ = project(p, axes, bins, ndim=1)
    q_h, _ = project(q, axes, bins, ndim=1)
    res = p_h - q_h

    fig, (ax1, ax2) = plt.subplots(2, 1, fig_kw={"figsize": (10, 15)})

    ax1.plot(p_h, label="P")
    ax1.plot(q_h, label="Q")
    ax1.title("Distributions 'P' and 'Q'")
    ax1.legend()

    ax2.plot(res)
    ax2.title("Residuals of 'P' - 'Q'")

    fig.show()

    return res

def residual2d(p: np.ndarray, q: np.ndarray, axes: list[int], bins: list[int]):
    p_h, _ = project(p, axes, bins, ndim=2)
    q_h, _ = project(q, axes, bins, ndim=2)

    res = p_h - q_h

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, fig_kw={"figsize": (10, 15)})

    ax1.imshow(p_h)
    ax1.title("Distribution 'P'")

    ax2.imshow(res)
    ax2.title("Residuals of 'P' - 'Q'")

    ax3.imshow(res)
    ax3.title("Distribution 'Q'")

    fig.show()

    # add colorbar

    return res

def test_chi(obs: np.ndarray, exp: np.ndarray, axes: list[int], bins: list[int], ndim: int=1):
    '''
    Perform chi^2 Goodness-of-fit test on a sample of observed data against
    expected data. 2D chi^2 test is performed by flattening the arrays into 1D
    and then comparing bin-wise.
    '''
    obs_h, _ = project(obs, axes, bins, ndim=ndim)
    exp_h, _ = project(exp, axes, bins, ndim=ndim)

    if ndim == 2:
        obs_h = obs_h.flatten()
        exp_h = exp_h.flatten()

    chi2, pval = chisquare(obs_h, exp_h, ddof=1)

    return chi2, pval

def test_ks(obs: np.ndarray, exp: np.ndarray, axes: list[int], bins: list[int]):
    obs_h, _ = project(obs, axes, bins, ndim=1)
    exp_h, _ = project(exp, axes, bins, ndim=1)
        
    result = kstest(obs_h, exp_h)
    return result.statistic, result.pvalue

def project(samples: np.ndarray, axes: list[int], bins: list[int], ndim: int=1) -> tuple[np.ndarray, np.ndarray]:
    '''
    Project sample points into a histogram of 1 or 2 dimensions.
    '''

    assert len(axes) <= samples.ndim
    assert len(axes) == len(bins)

    points = samples[: axes]

    if ndim == 1:
        hist, xedges = np.histogram(points, bins)
        edges = np.array([xedges])
    elif ndim == 2:
        hist, xedges, yedges = np.histogram2d(points, bins)
        edges = np.concatenate(([xedges], [yedges]), axis=0)
    else:
        print("Will not project points into bins of dimension 3 or greater")
        return (None, None)

    return hist, edges


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