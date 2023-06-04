import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.stats import chisquare
from scipy.special import erf

import defs

# training data plot
def plot_train(data, labels, path):
    fig = plt.figure(figsize=(10,10))
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

# plot the intermediate results
def plot_gen_intermediate():
    # for i, d in enumerate(feat_extraction_dists):
    #     out = np.array(d.sample((n_samples, ), bijector_kwargs=current_kwargs))
    #     plt.scatter(out[:, 0], out[:, 1], color='darkblue', s=25)
    pass

# def _chi2(p, q):
#     return np.sum(np.divide(np.square(np.subtract(p, q)), q))

def _gausformula(x, mean, var) -> np.ndarray:
    return np.divide(np.exp(np.multiply(-1./2 / var, np.square(np.subtract(x, mean)))),
                     np.sqrt(2 * np.pi * var))


# next let's compare two dists

def gaussinity(samples, labels, outputpath=None, name=None):

    if outputpath is None:
        outputpath = '%s_%.03f-%.03f.png'
    
    if name is None:
        name = 'DATA'

    uniquelabels = np.unique(labels)

    for label in uniquelabels:
        sample = samples[np.where(labels == label)[0]]
        label = labels[i,:]

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