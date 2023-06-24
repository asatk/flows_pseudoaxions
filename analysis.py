from datetime import datetime
import numpy as np
from os import mkdir

import defs
import utils as myutils
import analysis_utils as autils

flowrunname = datetime.today().date()
datarunname = '2023-05-26'

def test_samples(distribution, mode=defs.GRID):
    ###------GENERATE SAMPLE DATA------###

    # choose label for which samples are generated:
    if mode == defs.LINE:
        testlabels = np.array([[1./(defs.ngaus + 1)]])
    elif mode == defs.GRID:
        # testlabels = np.array([[3./(ngausx + 1), 3./(ngausy + 1)]])
        # testlabels = np.array([[1./(ngausx + 1), 1./(ngausy + 1)],
    #                        [2./(ngausx + 1), 2./(ngausy + 1)],
    #                        [3./(ngausx + 1), 3./(ngausy + 1)],
    #                        [4./(ngausx + 1), 4./(ngausy + 1)],
    #                        [5./(ngausx + 1), 5./(ngausy + 1)]])

        xax = np.arange(1, defs.ngausx + 1, dtype=float)
        yax = np.arange(1, defs.ngausy + 1, dtype=float)
        x, y = np.meshgrid(xax, yax)
        testlabels = np.array([x.ravel() / (defs.ngausx + 1.), y.ravel() / (defs.ngausy + 1.)]).T

    current_kwargs = {}
    gen_labels = np.repeat(testlabels, defs.ngen, axis=0)
    for i in range(defs.nmade):
        current_kwargs[f"maf_{i}"] = {'conditional_input' : gen_labels}

    # generate new samples
    gen_data = np.array(distribution.sample((defs.ngen * testlabels.shape[0], ), bijector_kwargs=current_kwargs))

    return gen_data, gen_labels

# def plot(labels, testlabels, mode=defs.GRID):
#     ###---PLOT TRAINING AND GENERATED SAMPLES------###

#     # prep data for plotting
#     uniquelabels = np.unique(labels, axis=0)
#     if mode == defs.LINE:
#         plotlabels = np.array([uniquelabels, np.ones(defs.ngaus) * defs.yval])
#         linevals = np.ones(shape=uniquelabels.shape) * defs.yval
#         plottestlabels = np.stack((testlabels, linevals), axis=1)
#     elif mode == defs.GRID:
#         plotlabels = labels
#         plottestlabels = testlabels

#     outputdirname = 'run_%s'%(flowrunname)
#     outputdir = myutils.namefile('output/', outputdirname, isdir=True)
#     mkdir(outputdir)
#     outputname = 'gaus2D_%02i-%02i_GEN'%(defs.ngausx, defs.ngausy)
#     outputpath = myutils.namefile(outputdir, outputname, ext=".png")
#     trainfigname = 'gaus2D_%02i-%02i_TRN'%(defs.ngausx, defs.ngausy)
#     trainfigpath = myutils.namefile(outputdir, trainfigname, ext=".png")
#     autils.plot_train(samples, plotlabels, trainfigpath)
#     autils.plot_gen(samples, samples_gen, plotlabels, plottestlabels, outputpath)