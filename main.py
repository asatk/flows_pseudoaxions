from datetime import datetime
import numpy as np
from os import mkdir
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions

import analysis_utils
import defs as defs
import flowmodel
import train_utils
import utils as myutils

### --- HYPERPARAMETERS --- ###

# flags
mode = defs.mode
newtraindata = True
saved_model = False
retrain = False

# network params
num_made = defs.nmade
num_inputs = defs.ndim
num_cond_inputs = defs.ndim_label

# training params
epochs = defs.nepochs
n_samples = defs.nsamp
n_gen_samples = defs.ngen
batch_size = defs.batch_size

# sample data params
if mode == defs.LINE:
    n_gaus_train = 2
    yval = defs.val
    gaus_width = defs.sigma_gaus
elif mode == defs.GRID:
    ngausx = defs.ngausx
    ngausy = defs.ngausy
    gaus_width = defs.sigma_gaus
elif mode == defs.ROOT:
    ndistx = defs.ndistx
    ndisty = defs.ndisty

# path info
runnum = 0
flowrunname = datetime.today().date()
datarunname = datetime.today().date()
rootdatapath = "./root/10x10box_10k_jun2023"
# datarunname = '2023-05-26'

### --- HYPERPARAMETERS --- ###



### --- TRAINING SAMPLES --- ###

# Make new training data and save it
if newtraindata:
    samples, labels = train_utils.makedata(mode, rootdatapath=rootdatapath)

# Load training data
else:
    datapath = 'data/gaussians_%s_%02ix%02i_run%02i.npy'%(datarunname, defs.ndistx, defs.ndisty, runnum)
    labelpath = 'data/gaussian_labels_%s_%02ix%02i_run%02i.npy'%(datarunname, defs.ndistx, defs.ndisty, runnum)
    samples = np.load(datapath)
    labels = np.load(labelpath)

### --- TRAINING SAMPLES --- ###



### --- MODEL --- ###

# Locate saved model
if saved_model:
    modeldir = 'model/flow_%s_%02ix%02i_run%02i'%(flowrunname, defs.ndistx, defs.ndisty, runnum)

# Name a new model
else:
    modeldir = 'flow_%s_%02ix%02i'%(flowrunname, defs.ndistx, defs.ndisty)
    modeldir = myutils.namefile('model/', modeldir, isdir=True)

# Load a model
if retrain or saved_model:
    model = keras.models.load_model(modeldir, custom_objects={"lossfn": flowmodel.lossfn, "Made": flowmodel.Made})
    made_blocks = []
    for i in range(num_made):
        made_blocks.append(model.get_layer(name=f"made_{i}"))
    
    distribution, made_list = flowmodel.build_distribution(made_blocks, num_inputs)

# Build a model from scratch
else:
    model, distribution = flowmodel.compile_MAF_model(num_made, num_inputs=num_inputs, num_cond_inputs=num_cond_inputs)

### --- MODEL --- ###



### --- TRAINING --- ###

# Train a model
if retrain or not saved_model:

    tstart = datetime.now()
    print("training begins: ", tstart)

    ###------TRAIN MODEL------###
    ckptpath = modeldir + "/cp-{epoch:04d}.ckpt"
    # as this is once again a unsupervised task, the target vector y is zeros
    model.fit(x=[samples, labels],
            y=np.zeros((samples.shape[0], 0), dtype=np.float32),
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            initial_epoch=defs.epoch_resume if retrain else 0,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath=ckptpath,
                    verbose=1,
                    save_weights_only=False,
                    save_freq=int(defs.epoch_save * len(samples) / batch_size)),
                myutils.SelectiveProgbarLogger(
                    verbose=1,
                    epoch_interval=10,
                    epoch_end=epochs)])

    model.save(modeldir)

    tend = datetime.now()
    print("training ends: ", tend, "\ntime elapsed: ", tend - tstart)

### --- TRAINING --- ###



### --- ANALYSIS --- ###

# define intermediate outputs for later
# feat_extraction_dists = flowmodel.intermediate_flows_chain(made_list)

# plotlabels_temp = plotlabels.reshape(-1, 2)
# gaussians_temp = gaussians.reshape(len(plotlabels_temp), n_samples, 2)
# plottestlabels_temp = plottestlabels.reshape(-1, 2)
# samples_temp = samples.reshape((len(plottestlabels_temp), n_gen_samples, 2))

# genoutputpath = outputdir + '/%s_%.03f-%.03f_GEN.png'
# trnoutputpath = outputdir + '/%s_%.03f-%.03f_TRN.png'

# analysis.gaussinity(samples, plottestlabels, genoutputpath, 'GEN')
# analysis.gaussinity(gaussians, plotlabels, trnoutputpath, 'TRAIN')
# analysis.comparegentotrain(gaussians_temp, plotlabels_temp, plottestlabels_temp, samples_temp)

### --- ANALYSIS --- ###