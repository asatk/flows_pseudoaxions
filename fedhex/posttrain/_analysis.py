"""
Author: Anthony Atkinson
Modified: 2023.07.15
"""

import math
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os
# from scipy.stats import chisquare, kstest
from typing import Any

from ..train.tf import _MAF
from ..utils import LOG_ERROR, LOG_FATAL, print_msg
from .plot import hist_one, plot_all, plot_losses, plot_one, \
    make_genplot_kwargs, make_trnplot_kwargs


plt.rcParams.update({
    "font.family": "serif"
})

ntools = 5

def analyze(distribution: Any, made_list: list[Any], training_data_path: str,
            ngen: int, lims: tuple[tuple], generated_data_path: str=None, loss_log: str=None,
            output_dir: str="output", tools: list[int]=None,
            nworkers: int=1) -> None:

    # TODO add docstring
    # TODO update tutorial comments
    """
    Default analysis behavior of network. Will run only if `defs.newanalysis`
    is True. Tools used in analysis can be selected with `tools` argument.
    """


    # Selection of tools used in analysis
    if tools is None:
        tools = list(range(0, ntools + 1))

    # Plotting paths
    out_trn = output_dir + "/plottrn.png"
    out_trn_i = output_dir + "/plottrn%03i.png"
    out_trn_hist_i = output_dir + "/histtrn%03i.png"
    out_gen = output_dir + "/plotgen.png"
    out_gen_i = output_dir + "/plotgen%03i.png"
    out_gen_hist_i = output_dir + "/histgen%03i.png"
    out_flow_i = output_dir + "/plotgen_flow%02i.png"
    out_loss = output_dir + "/loss.png"
    out_res = output_dir + "/res.png"
    out_chi = output_dir + "/chi.png"

    
    # ++++ Prepare Training Data ++++ #
    if not os.path.isfile(training_data_path):
        print_msg("Cannot locate the training data " +
              "located at `%s`."%(training_data_path), level=LOG_FATAL)
        return

    # Load dewhitened training data
    loader = Loader(training_data_path)
    loader.load()
    trn_data = loader.get_data()
    trn_cond = loader.get_cond()
    trn_samples, trn_labels = loader.recover()

    # Group samples & labels by unique label
    labels_unique, inverse_unique = np.unique(trn_labels, return_inverse=True, axis=0)
    # ++++ Prepare Training Data ++++ #
    
    ndim = trn_samples.shape[-1]
    ndim_label = trn_labels.shape[-1]

    # ++++ Prepare Generated Data ++++ #
    # Obtain both net- and user-friendly versions of the conditional data
    if generated_data_path is None:
        # This is a hardcoded label just to serve as an example.
        gen_labels = np.repeat([[2464, 5.125]], ngen, axis=0) # arb. label
        gen_cond = loader.preproc_new(gen_labels, is_cond=True)
        gen_data = None
    else:
        if not os.path.isfile(generated_data_path):
            print_msg("Cannot locate the generated data " +
                  "located at `%s`."%(generated_data_path), level=LOG_FATAL)
            return
        gen_loader = Loader(generated_data_path)
        gen_loader.load()
        gen_data = gen_loader.get_data()
        gen_cond = gen_loader.get_cond()
        # Load conditional data that is already whitened from file. These data
        # need not be the  conditional data with which the network is trained.
        gen_samples, gen_labels = gen_loader.recover()

    # Make a list of the unique labels and a list of their occurrence in gen_labels
    gen_labels_unique, gen_inverse_unique = np.unique(gen_labels, return_inverse=True, axis=0)

    # Generate net-friendly version of the data conditioned on provided labels
    # Only generated new data if some is not already provided at loading
    if gen_data is None:
        print(made_list)
        print(len(made_list))
        # Define the conditional input for the flow to generate samples with at each flow
        current_kwargs = {}
        for i in range(len(made_list) // 2):
            current_kwargs[f"maf_{i}"] = {"conditional_input" : gen_cond}

        # Generate the data conditioned on the net-friendly labels
        gen_data = np.array(distribution.sample((gen_cond.shape[0], ), bijector_kwargs=current_kwargs))
    # Load net-friendly version of the data
    
    # Transform generated data from net-friendly to user-friendly
    gen_samples = loader.recover_new(gen_data, is_cond=False)
    # ++++ Prepare Generated Data ++++ #

    # Group data
    trn_samples_grp = [
        trn_samples[inverse_unique == i] for i in range(len(labels_unique))
    ]
    gen_samples_grp = [
        gen_samples[gen_inverse_unique == i] for i in range(len(gen_labels_unique))
    ]


    # define the samples compared as only those which share a label in
    # both the trianing and generated data
    _, indexes_int = intersect_labels(labels_unique, gen_labels_unique, return_index=True)
    indexes_compare = list(zip(*indexes_int))

    if len(indexes_int) > 0:

        # TODO check for empty set or some errors in intersect labels
        trn_indexes_compare = indexes_compare[0]
        gen_indexes_compare = indexes_compare[1]

        trn_compare = np.empty(shape=(0, ndim))
        gen_compare = np.empty(shape=(0, ndim))

        for index in trn_indexes_compare:
            trn_compare = np.concatenate((trn_compare, trn_samples_grp[index]), axis=0)

        for index in gen_indexes_compare:
            gen_compare = np.concatenate((gen_compare, gen_samples_grp[index]), axis=0)
    elif 4 in tools or 5 in tools:
        print_msg("No labels are shared between the training and generated "+\
                  "data, so tools 4 and 5 cannot be used.", level=LOG_ERROR)
        return
    

    ### TOOL 1 - Plot Training Data and Network Output
    if 1 in tools:
        # ++++ Plot training data ++++ #
        trnplot_kwargs = make_trnplot_kwargs(labels_unique, trn_samples_grp, len(trn_samples), out_trn_i, lims)
        trnhist_kwargs = make_trnplot_kwargs(labels_unique, trn_samples_grp, len(trn_samples), out_trn_hist_i, lims)

        # Plot scatter plots and histograms for each label
        with mp.Pool(nworkers) as pool:
            pool.starmap(plot_one, trnplot_kwargs)
            pool.starmap(hist_one, trnhist_kwargs)
        # ++++ Plot training data ++++ #


        # ++++ Plot generated data ++++ #
        genplot_kwargs = make_genplot_kwargs(gen_labels_unique, gen_samples_grp, len(gen_samples), out_gen_i, lims)
        genhist_kwargs = make_genplot_kwargs(gen_labels_unique, gen_samples_grp, len(gen_samples), out_gen_hist_i, lims)

        # Plot scatter plots and histograms for each label
        with mp.Pool(nworkers) as pool:
            pool.starmap(plot_one, genplot_kwargs)
            pool.starmap(hist_one, genhist_kwargs)
            # pool.starmap(print_stats, zip(grouped_samples, gen_labels_unique))
        # ++++ Plot generated data ++++ #

        # Plot scatter plot for all training and generated data
        plot_all(trn_samples, labels_unique, out_trn, show=True, xlim=lims[0], ylim=lims[1])
        plot_all(gen_samples, gen_labels, out_gen, show=True, xlim=lims[0], ylim=lims[1])
    

    ### TOOL 2 - Plot Intermediate Output (assess each bijector's contribution)
    if 2 in tools:
        # Generate samples for each intermediate flow
        flow_distributions = _MAF.intermediate_flows_chain(made_list)
        for i, dist in enumerate(flow_distributions):
            gen_data_flow_i = dist.sample((gen_cond.shape[0], ), bijector_kwargs=current_kwargs)
            gen_samples_flow_i = loader.recover_new(gen_data_flow_i, is_cond=False)
            plot_args = {"title": "Generated Output (N = %i) up to Flow %i"%(gen_cond.shape[0], i)}
            plot_all(gen_samples_flow_i, gen_labels_unique, out_flow_i%i, kwargs=plot_args)


    ### TOOL 3 - Track Training Losses (assess ability to converge)
    if 3 in tools:
        # Plot losses during training
        losses = np.load(loss_log)
        plot_losses(losses, out_path=out_loss, show=True)


    ### TOOL 4 - Histograms
    if 4 in tools:
        
        axes = [0]  # project along x-axis/phi-axis
        bins = [8] # bin along axis of interest
        ndim = 1    # dimension of space into which the samples are projected
        h_range = (2000., 3000.)

        # Projections
        trn_samples_h, e1 = project(trn_compare, axes=axes, bins=bins, ndim=ndim, h_range=h_range)
        gen_samples_h, e2 = project(gen_compare, axes=axes, bins=bins, ndim=ndim, h_range=h_range)

        # Residuals
        norm = False
        pname = "Generated"
        qname = "Training"
        ptitle = f"Generated Samples N = {len(gen_compare)}"
        qtitle = f"Training Samples N = {len(trn_compare)}"
        xlabel = "Reconstructed $\Phi$ Mass"
        res = residual(gen_compare, trn_compare, axes=axes, bins=bins,
                       h_range=h_range, norm=norm, pname=pname, qname=qname,
                       ptitle=ptitle, qtitle=qtitle, out_path=out_res,
                       xlabel=xlabel)
        

    ### TOOL 5 - Numerical Fit Tests
    if 5 in tools:

        # make chi-2 more valid/stable
        
        axes = [0]
        bins = [8]
        ndim = 1
        h_range = (2000, 3000)

        pname = "Generated"
        qname = "Training"
        ptitle = f"Generated Samples N = {len(gen_compare)}"
        qtitle = f"Training Samples N = {len(trn_compare)}"
        xlabel = "Reconstructed $\Phi$ Mass"

        args = {"pname": pname, "qname": qname, "ptitle": ptitle,
                "qtitle": qtitle, "xlabel": xlabel}
        
        plot_err = False

        # Chi^2 test
        chi2, pval = test_chi(gen_compare, trn_compare, axes=axes, bins=bins,
                              ndim=ndim, h_range=h_range, out_path=out_chi,
                              plot_err=plot_err, **args)
        print_msg("Chi^2 analysis comparing the generated samples to the" + 
                  " training samples yields: " +
                  f"chi={np.sqrt(chi2):3f}; p={pval:3f}")
        # Kolmogorov-Smirnov test
    

# This isn't the best way to do what I want but it does what I want so it stays
# What it does: finds the labels shared by two sets (if any)
def intersect_labels(labels_1: np.ndarray, labels_2: np.ndarray,
                     return_index: bool=False) -> np.ndarray:
    """
    Returns the intersection of the two sets of labels. If the intersection is
    empty, an empty list is returned. Otherwise, a list containing the shared
    labels is returned. If `return_index` is True, a list of indices noting the
    location of the shared label for each set is returned. If the intersection
    is empty, an empty list is returned.
    """
    labels_int = []
    indexes_int = []    # the indexes in the first array are returned only

    labels_1_copy = labels_1.copy()
    labels_2_copy = labels_2.copy()

    i = 0

    # check if the label is shared by both sets
    while i < len(labels_1_copy):
        j = 0
        label_1 = labels_1_copy[i]
        len_l2 = len(labels_2_copy)
        while j < len_l2:
            label_2 = labels_2_copy[j]
            if np.array_equiv(label_1, label_2):
                labels_int.append(label_1)
                indexes_int.append([i, j])

                # speeds up search for labels by removing found labels
                # ensure that labels arrays are unique
                np.delete(labels_2_copy, j, axis=0)
            
            j += 1
        i += 1

    if return_index:
        return labels_int, indexes_int
    return labels_int


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


def residual(p: np.ndarray, q: np.ndarray, axes: list[int],
             bins: list[int], h_range: tuple|None=None,
             norm: bool=False, ret_fig_ax: bool=False,
             out_path: str=None, plot_err: bool=False, **kwargs) -> np.ndarray:
    
    kwkeys = kwargs.keys()
    pname = kwargs["pname"] if "pname" in kwkeys else "P"
    qname = kwargs["qname"] if "qname" in kwkeys else "Q"
    ptitle = kwargs["ptitle"] if "ptitle" in kwkeys else f"Distribution '{pname}'"
    qtitle = kwargs["qtitle"] if "qtitle" in kwkeys else f"Distribution '{qname}'"
    btitle = kwargs["btitle"] if "btitle" in kwkeys else f"Distribution '{pname}' and '{qname}'"
    rtitle = kwargs["rtitle"] if "rtitle" in kwkeys else f"Residuals of '{pname}' - '{qname}'"
    xlabel = kwargs["xlabel"] if "xlabel" in kwkeys else "X"
    ylabel = "density" if norm else "events"

    p_h, edges = project(p, axes, bins, ndim=1, h_range=h_range, norm=norm)
    h_range = (np.min(edges), np.max(edges)) if h_range is None else h_range
    q_h, _ = project(q, axes, bins, ndim=1, h_range=h_range, norm=norm)
    res = p_h - q_h

    sharex = "all"
    sharey = "row" if norm else "none"
    fig, ax = plt.subplots(2, 2, figsize=(10., 10.), sharex=sharex,
                           sharey=sharey)

    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes
    ax4: plt.Axes

    ((ax1, ax2), (ax3, ax4)) = ax
    ax1.set_xlim(h_range) # sets x limits for all plots

    p_err = np.sqrt(p_h)
    q_err = np.sqrt(p_h)
    res_err = np.sqrt(p_h + q_h)
    centers = np.ediff1d(edges)
    nudge = 0.001 * (h_range[1] - h_range[0])

    # TODO for normalized data these sqrts might be messed up...

    ax1.stairs(p_h, edges=edges, color='C0', alpha=0.5, fill=False, linewidth=2.0)
    if plot_err:
        ax1.errorbar(centers, p_h, yerr=p_err, ecolor='gray', elinewidth=1.0)
    ax1.set_title(ptitle)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2.stairs(q_h, edges=edges, color='C1', alpha=0.5, fill=False, linewidth=2.0)
    if plot_err:
        ax2.errorbar(centers, q_h, yerr=q_err, ecolor='gray', elinewidth=1.0)
    ax2.set_title(qtitle)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)

    

    ax3.stairs(p_h, edges=edges, color='C0', alpha=0.5, fill=False, linewidth=2.0, label=pname)
    ax3.stairs(q_h, edges=edges, color='C1', alpha=0.5, fill=False, linewidth=2.0, label=qname)
    if plot_err:
        ax3.errorbar(centers + nudge, p_h, yerr=p_err, ecolor='gray', elinewidth=1.0)
        ax3.errorbar(centers - nudge, q_h, yerr=q_err, ecolor='gray', elinewidth=1.0)
    ax3.set_title(btitle)
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    ax3.legend()

    res_pos = res.astype(float)
    res_pos[res_pos <= 0] = np.nan

    res_neg = res.astype(float)
    res_neg[res_neg >= 0] = np.nan

    ax4.hlines(y=0, xmin=h_range[0], xmax=h_range[1], colors='gray', alpha=0.8, linestyles='dashed')
    ax4.stairs(res_pos, edges=edges, color='cornflowerblue', linewidth=2.0)
    ax4.stairs(res_neg, edges=edges, color='red', linewidth=2.0)
    if plot_err:
        ax4.errorbar(centers, res, yerr=res_err, ecolor="black", elinewidth=1.0)
    ax4.set_title(rtitle)
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel(ylabel)

    fig.tight_layout(pad=1.2)

    if out_path is not None:

        fig.savefig(out_path)

    if ret_fig_ax:
        return res, fig, ax
    else:
        plt.show()

    return res


def residual2d(p: np.ndarray, q: np.ndarray, axes: list[int],
               bins: list[int], h_range: tuple|None=None, norm: bool=False,
               **kwargs) -> np.ndarray:
    
    kwkeys = kwargs.keys()
    pname = kwargs["pname"] if "pname" in kwkeys else "P"
    qname = kwargs["qname"] if "qname" in kwkeys else "Q"
    
    p_h, _ = project(p, axes, bins, ndim=2, h_range=h_range, norm=norm)
    q_h, _ = project(q, axes, bins, ndim=2, h_range=h_range, norm=norm)

    res = p_h - q_h

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, fig_kw={"figsize": (10, 15)})

    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes

    ax1.imshow(p_h)
    ax1.set_title(f"Distribution '{pname}'")

    ax2.imshow(res)
    ax2.set_title(f"Residuals of '{pname}' - '{qname}'")

    ax3.imshow(res)
    ax3.set_title(f"Distribution '{qname}'")

    plt.show()

    # add colorbar

    return res


def test_chi(obs: np.ndarray, exp: np.ndarray, axes: list[int],
             bins: list[int], ndim: int=1, h_range: tuple=None,
             out_path: str=None, **kwargs) -> tuple[float, float]:
    '''
    Perform chi^2 Goodness-of-fit test on a sample of observed data against
    expected data. 2D chi^2 test is performed by flattening the arrays into 1D
    and then comparing bin-wise.
    '''

    # if len(obs) != len(exp):
    #     norm = True
    # else:
    #     norm = False

    obs_h, _ = project(obs, axes, bins, ndim=ndim, h_range=h_range, norm=True)
    exp_h, _ = project(exp, axes, bins, ndim=ndim, h_range=h_range, norm=True)

    if ndim == 2:
        obs_h = obs_h.flatten()
        exp_h = exp_h.flatten()

    good_categories = (obs_h > 0) | (exp_h > 0)
    obs_h_chi = obs_h[good_categories]
    exp_h_chi = exp_h[good_categories]

    # chi2, pval = chisquare(obs_h_chi, exp_h_chi, ddof=1)
    chi2, pval = (0., 0.)

    if chi2 == np.inf:
        print_msg("chi2 analysis failed", level=LOG_ERROR)

    if ndim == 1:
        _, fig, ax = residual(obs, exp, axes=axes, bins=bins, ndim=1, h_range=h_range, norm=True, ret_fig_ax=True, **kwargs)
        ax3: plt.Axes
        ax3 = ax[1][0]
        ax3.text(0.7, 0.5,
                 f"$\chi={np.sqrt(chi2):3f}$\np$={pval:3f}$\nbins$={bins[0]}$",
                 transform=ax3.transAxes,
                 bbox=dict(facecolor="#f5f5dc", alpha=0.5))

        fig.tight_layout(pad=1.2)

        if out_path is not None:
            fig.savefig(out_path)

        plt.show()

    return chi2, pval


def test_ks(obs: np.ndarray, exp: np.ndarray, axes: list[int],
            bins: list[int]) -> tuple[float, float]:
    obs_h, _ = project(obs, axes, bins, ndim=1)
    exp_h, _ = project(exp, axes, bins, ndim=1)
        
    # result = kstest(obs_h, exp_h)
    result = None
    return result.statistic, result.pvalue


def project(samples: np.ndarray, axes: list[int], bins: list[int],
            ndim: int=1, h_range: tuple|None=None, norm: bool=False) -> tuple[np.ndarray, np.ndarray]:
    '''
    Project sample points into a histogram of 1 or 2 dimensions.

    sapmles: np.ndarry
        N x M array of coordinate samples where N is the number of samples
        and M is the original number of dimensions of the data.
    axes: list[int]:
        Of the M dimensions of `samples`, choose the axes over which to
        project these data.
    bins: list[int]
        List of the number of bins used along each axis in the histogram.
    ndim: int
        number of dimensions of projection
    h_range: tuple or None
        range of the histogram calculated. If None, the min and max of `samples`
        is used as it is the default behavior of `np.histogram`.
    '''

    assert len(axes) <= samples.ndim
    assert len(axes) == len(bins)

    points = samples[:, axes]

    if ndim == 1:
        hist, xedges = np.histogram(points, bins=bins[0], range=h_range,
                                    density=norm)
        edges = np.array(xedges)
    elif ndim == 2:
        hist, xedges, yedges = np.histogram2d(points, bins, range=h_range,
                                              density=norm)
        edges = np.concatenate(([xedges], [yedges]), axis=0)
    else:
        print("Will not project points into bins of dimension 3 or greater")
        return (None, None)

    return hist, edges
