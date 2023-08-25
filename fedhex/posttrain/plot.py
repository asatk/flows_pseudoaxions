"""
Author: Anthony Atkinson
Modified: 2023.07.15
"""

import math
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import multiprocessing as mp
import numpy as np
from numpy.typing import ArrayLike
import os
from scipy.stats import chisquare, kstest
from typing import Any

from ..train.tf import _MADEflow
from ..utils import LOG_ERROR, LOG_FATAL

plt.rcParams.update({
    "font.family": "serif"
})

def make_trnplot_kwargs(labels_u: np.ndarray, grouped_data: np.ndarray, nsamples: int, out_path: str, lims: tuple[tuple]) -> list[dict]:
    '''
    labels_u: conditional data that will be plotted
    '''
    

    d = {
        "xlabel": "Reconstructed $\Phi$ Mass (GeV)",
        "ylabel": "Reconstructed $\omega$ Mass (GeV)",
        "label_c": "red",
        "label_s": 20,
        "label_alpha": 1.0,
        "sample_c": "green",
        "sample_s": 20,
        "sample_alpha": 0.5,
        "xlim": lims[0],
        "ylim": lims[1]
    }

    title_str = f"Generated Samples for (%g, %.3f)\nN = {nsamples}"

    args_list = []
    for i, label_u in enumerate(labels_u):
        args_list.append([grouped_data[i], label_u, out_path%i,
                          {"title": title_str%(label_u[0], label_u[1]), **d}])
    
    return args_list


def make_genplot_kwargs(labels_u: np.ndarray, grouped_data: np.ndarray, nsamples: int, out_path: str, lims: tuple[tuple]) -> list[dict]:
    '''
    labels_u: unique conditional data that will be plotted
    '''
    
    d = {
        "xlabel": "Reconstructed $\Phi$ Mass (GeV)",
        "ylabel": "Reconstructed $\omega$ Mass (GeV)",
        "label_c": "orange",
        "label_s": 20,
        "label_alpha": 1.0,
        "sample_c": "blue",
        "sample_s": 20,
        "sample_alpha": 0.5,
        "xlim": lims[0],
        "ylim": lims[1]
    }

    title_str = f"Generated Samples for (%g, %.3f)\nN = {nsamples}"

    args_list = []
    for i, label_u in enumerate(labels_u):
        args_list.append([grouped_data[i], label_u, out_path%i,
                          {"title": title_str%(label_u[0], label_u[1]), **d}])
    
    return args_list


def plot_losses(losses, out_path: str=None, show=False,
                plot_args: dict={}) -> None:
    '''
    Plot loss (negative-log likelihood) of network over time (epochs)
    '''
    
    # Parse keyword arguments
    kwkeys = plot_args.keys()
    title = "Loss vs. Epoch" if "title" not in kwkeys else plot_args["title"]
    xlabel = "Epoch" if "xlabel" not in kwkeys else plot_args["xlabel"]
    ylabel = "Loss (Negative Log Likelihood)" if "ylabel" not in kwkeys else plot_args["ylabel"]

    # TODO check pos/neg len > 0
    # Separate loss data into positive and negative losses
    losses_nonneg = losses[losses[:, 1] >= 0]
    losses_neg = np.abs(losses[losses[:, 1] < 0])
    turnover_epoch = losses[losses[:, 1] < 0][0, 0]

    # Plot losses
    fig, ax = plt.subplots()
    
    if len(losses_nonneg) > 0:
        ax.semilogy(losses_nonneg[:, 0], losses_nonneg[:, 1], c="blue", label="positive loss")
    if len(losses_neg) > 0:
        ax.semilogy(losses_neg[:, 0], losses_neg[:, 1], c="red", label="negative loss")
    if len(turnover_epoch) > 0:
        ax.vlines(turnover_epoch, ymin=np.min(losses[:, 1]), ymax=np.max(losses[:,1 ]), colors=["gray"], linestyles="dashed")
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.xaxis.get_major_locator().set_params(integer=True)

    if out_path is not None:
        fig.savefig(out_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_one(samples: np.ndarray, label: np.ndarray, out_path: str=None,
             plot_args: dict={}) -> None:
    '''
    Scatter plot of samples corresponding to a single label
    samples: 2D coordinate pairs of events in parameter space
    label: 2D coordinate pair of label associated to events
    outpath: location to where the plot is saved
    '''

    xdata = samples[:, 0]
    ydata = samples[:, 1]
    xmin = np.min(xdata)
    xmax = np.max(xdata)
    ymin = np.min(ydata)
    ymax = np.max(ydata)
    xlbl = label[0]
    ylbl = label[1]

    default_title = f"Generated Samples for ({xlbl:g}, " + \
                    f"{ylbl:.3f})\nN = {len(samples)}"
    default_xlim = (xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
    default_ylim = (ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))

    # Parse keyword arguments
    title = plot_args.get("title", default_title)
    xlabel = plot_args.get("xlabel", "Reconstructed $\Phi$ Mass (GeV)")
    ylabel = plot_args.get("ylabel", "Reconstructed $\omega$ Mass (GeV)")
    label_c = plot_args.get("label_c", "red")
    label_s = plot_args.get("label_s", 20)
    label_alpha = plot_args.get("label_alpha", 1.0)
    sample_c = plot_args.get("sample_c", "green")
    sample_s = plot_args.get("sample_s", 20)
    sample_alpha = plot_args.get("sample_alpha", 0.5)
    xlim = plot_args.get("xlim", default_xlim)
    ylim = plot_args.get("ylim", default_ylim)


    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata, c=sample_c, s=sample_s, alpha=sample_alpha)
    ax.scatter(xlbl, ylbl, c=label_c, s=label_s, alpha=label_alpha)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True)

    if out_path is not None:
        fig.savefig(out_path)
    else:
        fig.show()
    plt.close()     #plots are closed immediately to save memory


def hist_one(samples: np.ndarray, label: np.ndarray, out_path: str=None,
             plot_args: dict={}) -> None:
    '''
    Scatter plot of samples corresponding to a single label
    samples: 2D coordinate pairs of events in parameter space
    label: 2D coordinate pair of label associated to events
    outpath: location to where the plot is saved
    '''

    xdata = samples[:, 0]
    ydata = samples[:, 1]
    xmin = np.min(xdata)
    xmax = np.max(xdata)
    ymin = np.min(ydata)
    ymax = np.max(ydata)
    xlbl = label[0]
    ylbl = label[1]

    default_title = f"Generated Samples for ({xlbl:g}, " + \
                    f"{ylbl:.3f})\nN = {len(samples)}"
    default_xlim = (xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin))
    default_ylim = (ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))

    # Parse keyword arguments
    title = plot_args.get("title", default_title)
    xlabel = plot_args.get("xlabel", "Reconstructed $\Phi$ Mass (GeV)")
    ylabel = plot_args.get("ylabel", "Reconstructed $\omega$ Mass (GeV)")
    label_c = plot_args.get("label_c", "red")
    label_s = plot_args.get("label_s", 20)
    label_alpha = plot_args.get("label_alpha", 1.0)
    xlim = plot_args.get("xlim", default_xlim)
    ylim = plot_args.get("ylim", default_ylim)
    nbinsx = plot_args.get("nbinsx", 50)
    nbinsy = plot_args.get("nbinsy", 50)

    
    fig, ax = plt.subplots()
    _, _, _, img = ax.hist2d(xdata, ydata, bins=(nbinsx, nbinsy), range=(xlim, ylim))
    plt.scatter(xlbl, ylbl, c=label_c, s=label_s, alpha=label_alpha)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(img)
    if out_path is not None:
        fig.savefig(out_path)
    else:
        fig.show()
    plt.close()     #plots are closed immediately to save memory


def plot_all(samples: np.ndarray, labels_unique: np.ndarray, out_path: str,
             show: bool=False, **kwargs) -> None:
    '''
    Scatter plot of samples for all labels
    samples: 2D coordinate pairs of events in parameter space
    labels: 2D coordinate pairs of all labels associated to events
    outpath: location to where the plot is saved
    show: flag indicating whether or not to show plot after generating it
    '''

    xdata = samples[:, 0]
    ydata = samples[:, 1]
    xmin = np.min(xdata)
    xmax = np.max(xdata)
    ymin = np.min(ydata)
    ymax = np.max(ydata)
    xlbls = labels_unique[:, 0]
    ylbls = labels_unique[:, 1]

    default_title = f"Generated Samples N = {len(samples)}"
    default_xlim = (xmin - 0.05 * (xmax - xmin), xmin + 0.05 * (xmax - xmin))
    default_ylim = (ymin - 0.05 * (ymax - ymin), ymin + 0.05 * (ymax - ymin))

    # Parse keyword arguments
    title = kwargs.get("title", default_title)
    xlabel = kwargs.get("xlabel", "Reconstructed $\Phi$ Mass (GeV)")
    ylabel = kwargs.get("ylabel", "Reconstructed $\omega$ Mass (GeV)")
    label_c = kwargs.get("label_c", "red")
    label_s = kwargs.get("label_s", 20)
    label_alpha = kwargs.get("label_alpha", 1.0)
    sample_c = kwargs.get("sample_c", "green")
    sample_s = kwargs.get("sample_s", 20)
    sample_alpha = kwargs.get("sample_alpha", 0.5)
    xlim = kwargs.get("xlim", default_xlim)
    ylim = kwargs.get("ylim", default_ylim)


    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata, c=sample_c, s=sample_s, alpha=sample_alpha)
    ax.scatter(xlbls, ylbls, c=label_c, s=label_s, alpha=label_alpha)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True)
    fig.savefig(out_path)
    
    if show:
        plt.show()
    else:
        plt.close()
