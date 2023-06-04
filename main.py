from datetime import datetime
import numpy as np
from os import mkdir
import pickle
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions

import analysis_utils
import defs as defs
import flowmodel
import utils as myutils

###------MODE CONSTANTS------###
LINE = 0
GRID = 1
ROOT = 2

mode = GRID

###------GET HYPERPARAMS------###

###------CREATE MODEL------###

# flags
newtraindata = False
# saved_model = False
saved_model = True
retrain = False
runnum = 0

# network params
epochs = defs.nepochs
n_samples = defs.nsamp
n_gen_samples = n_samples
# n_gen_samples = 10 * n_samples
gaus_width = defs.sigma_gaus
num_made = defs.nmade
num_inputs = defs.ndim
num_cond_inputs = defs.ndim_label

# gaussian sample data parameters
if mode == defs.LINE:
    n_gaus_train = 2
    yval = defs.val
    batch_size = n_gaus_train * n_samples // (10)
elif mode == defs.GRID:
    ngausx = 3
    ngausy = 3
    batch_size = ngausx * ngausy * n_samples // (10)
elif mode == defs.ROOT:
    ndistx = 20
    ndisty = 20
    batch_size = 0

# plot output info
flowrunname = datetime.today().date()
datarunname = '2023-05-26'


###------CREATE TRAINING SAMPLES------###
if newtraindata:

    datadir = 'data/'
    labelfile = 'gaussian_labels_%s_%02i-%02i'%(datarunname, ngausx, ngausy)
    labelpath = myutils.namefile(datadir, labelfile, ext=".npy")
    datafile = 'gaussians_%s_%02i-%02i'%(datarunname, ngausx, ngausy)
    datapath = myutils.namefile(datadir, datafile, ext=".npy")

    if mode == defs.LINE:
        # assign the 1D labels for each gaussian
        labels = myutils.normalize_labels_line_1d(
            myutils.train_labels_line_1d(n_gaus_train))
        # calculate expected center of each gaussian
        ctrs = myutils.dist_center_line_1d(labels, yval)

    elif mode == defs.GRID:
        labels = myutils.normalize_labels_grid_2d(
            myutils.train_labels_grid_2d(ngausx, ngausy))
        ctrs = myutils.dist_center_grid_2d(labels)

    elif mode == defs.ROOT:
        # these are provided in the data - not nec. regularly spaced
        labels = None
        ctrs = None
        # labels = myutils.normalize_labels_pseudo_2d(
        #     myutils.train_labels_pseudo_2d(ndistx, ndisty))
        # ctrs = myutils.dist_center_pseudo

    if mode != defs.ROOT:
        # generate cov mtx for each gaussian
        # gaus_cov_indv = myutils.cov_skew(gaus_width, gaus_width/2., gaus_width/2., gaus_width)
        gaus_cov_indv = myutils.cov_xy(gaus_width)
        gaus_covs = myutils.cov_change_none(ctrs, gaus_cov_indv)
        # samples each gaussian for n_samples points, each with an associated label
        gaussians, gaussian_labels = \
            myutils.sample_real_gaussian(n_samples, labels, ctrs, gaus_covs)

        data = np.concatenate((gaussians, gaussian_labels), axis=1)
        # np.random.shuffle(data)

        np.save(labelpath, gaussian_labels)
        np.save(datapath, gaussians)

    else:
        # process via uproot
        data = None
    
else:
    datapath = 'data/gaussians_%s_%02i-%02i_%i.npy'%(datarunname, ngausx, ngausy, runnum)
    labelpath = 'data/gaussian_labels_%s_%02i-%02i_%i.npy'%(datarunname, ngausx, ngausy, runnum)

    gaussian_labels = np.load(labelpath)
    gaussians = np.load(datapath)

    data = np.concatenate((gaussians, gaussian_labels), axis=1)

if saved_model:
    modeldir = 'model/flow_%s_%02i-%02i'%(flowrunname, ngausx, ngausy)
else:
    modeldir = 'flow_%s_%02i-%02i'%(flowrunname, ngausx, ngausy)
    modeldir = myutils.namefile('model/', modeldir, isdir=True)

if retrain or saved_model:
    model = keras.models.load_model(modeldir, custom_objects={"lossfn": flowmodel.lossfn, "Made": flowmodel.Made})
    made_blocks = []
    for i in range(num_made):
        made_blocks.append(model.get_layer(name=f"made_{i}"))
    # print(made_blocks)
    distribution, made_list = flowmodel.build_distribution(made_blocks, num_inputs)
    
else:
    #Define ten-block made network
    model, distribution, made_list = flowmodel.compile_MAF_model(num_made, num_inputs=num_inputs, num_cond_inputs=num_cond_inputs, return_layer_list=True)

if retrain or not saved_model:

    ###------TRAIN MODEL------###

    # for i,made in enumerate(made_list):
    #     print(made)
    #     made.save(madedir%(i))
    
    steps = gaussians.shape[0] // batch_size
    #as this is once again a unsupervised task, the target vector y is again zeros
    model.fit(x=[data[:, 0:num_inputs], data[:, num_inputs:]],
            y=np.zeros((gaussians.shape[0], 0), dtype=np.float32),
            batch_size=batch_size,
            epochs=epochs,
            # steps_per_epoch=steps,
            verbose=0,
            initial_epoch=defs.epoch_resume if retrain else 0,
            callbacks=[myutils.SelectiveProgbarLogger(verbose=1, epoch_interval=10, epoch_end=epochs)],
            shuffle=True)

    model.save(modeldir)

# define intermediate outputs for later
# feat_extraction_dists = flowmodel.intermediate_flows_chain(made_list)

###------GENERATE SAMPLE DATA------###

# choose label for which samples are generated:
if mode == LINE:
    testlabels = np.array([[1./(n_gaus_train + 1)]])
elif mode == GRID:
    # testlabels = np.array([[3./(ngausx + 1), 3./(ngausy + 1)]])
    # testlabels = np.array([[1./(ngausx + 1), 1./(ngausy + 1)],
#                        [2./(ngausx + 1), 2./(ngausy + 1)],
#                        [3./(ngausx + 1), 3./(ngausy + 1)],
#                        [4./(ngausx + 1), 4./(ngausy + 1)],
#                        [5./(ngausx + 1), 5./(ngausy + 1)]])

    xax = np.arange(1, ngausx + 1, dtype=float)
    yax = np.arange(1, ngausy + 1, dtype=float)
    x, y = np.meshgrid(xax, yax)
    testlabels = np.array([x.ravel() / (ngausx + 1.), y.ravel() / (ngausy + 1.)]).T


current_kwargs = {}
cond = np.repeat(testlabels, n_gen_samples, axis=0)
# np.random.shuffle(cond)
for i in range(num_made):
    current_kwargs[f"maf_{i}"] = {'conditional_input' : cond}

#generate 1000 new samples
samples = np.array(distribution.sample((n_gen_samples * testlabels.shape[0], ), bijector_kwargs=current_kwargs))

###------SAMPLE STATISTICS------###

# statistics on results
'''
mean_train = np.mean(gaussians[0:n_samples], axis=0)
cov_train = np.cov(gaussians[0:n_samples].T)
print("Training Data (1st Gaus): \n - Mean: ", mean_train, "\n - Cov: ", cov_train, "\n - Labels: ", labels)
print("\nGenerating For Label %0.4f"%(testlabel))
mean_exp = np.array([testlabel, 0.5])
# print("\nGenerating For Label (%0.4f, %0.4f)"%(label[0], label[1]))
# mean_exp = np.array(label)
cov_exp = myutils.cov_change_none(np.array([[testlabel,0.5]]), gaus_cov_indv)
print("\nExpected Stats: \n - Mean: ", mean_exp, "\n - Cov: ", cov_exp)
mean_gen = np.mean(samples, axis=0)
cov_gen = np.cov(samples.T)
print("\nGenerated Data: \n - Mean", mean_gen, "\n - Cov", cov_gen)
'''

###---PLOT TRAINING AND GENERATED SAMPLES------###

# prep data for plotting
labels = np.unique(gaussian_labels, axis=0)
if mode == LINE:
    plotlabels = np.array([labels, np.ones(n_gaus_train) * yval])
    linevals = np.ones(shape=labels.shape) * defs.yval
    plottestlabels = np.stack((testlabels, linevals), axis=1)
elif mode == GRID:
    plotlabels = labels
    plottestlabels = testlabels

outputdirname = 'run_%s'%(flowrunname)
outputdir = myutils.namefile('output/', outputdirname, isdir=True)
mkdir(outputdir)
outputname = 'gaus2D_%02i-%02i_GEN'%(ngausx, ngausy)
outputpath = myutils.namefile(outputdir, outputname, ext=".png")
trainfigname = 'gaus2D_%02i-%02i_TRN'%(ngausx, ngausy)
trainfigpath = myutils.namefile(outputdir, trainfigname, ext=".png")
analysis_utils.plot_train(gaussians, plotlabels, trainfigpath)
analysis_utils.plot_gen(gaussians, samples, plotlabels, plottestlabels, outputpath)

exit()

# plotlabels_temp = plotlabels.reshape(-1, 2)
# gaussians_temp = gaussians.reshape(len(plotlabels_temp), n_samples, 2)
# plottestlabels_temp = plottestlabels.reshape(-1, 2)
# samples_temp = samples.reshape((len(plottestlabels_temp), n_gen_samples, 2))

# genoutputpath = outputdir + '/%s_%.03f-%.03f_GEN.png'
# trnoutputpath = outputdir + '/%s_%.03f-%.03f_TRN.png'

# analysis.gaussinity(samples, plottestlabels, genoutputpath, 'GEN')
# analysis.gaussinity(gaussians, plotlabels, trnoutputpath, 'TRAIN')
# analysis.comparegentotrain(gaussians_temp, plotlabels_temp, plottestlabels_temp, samples_temp)