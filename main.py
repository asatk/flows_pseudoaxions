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

###------MODE CONSTANTS------###
mode = defs.mode

###------GET HYPERPARAMS------###

###------CREATE MODEL------###

# flags
newtraindata = True
# saved_model = False
saved_model = False
retrain = False
runnum = 0

# network params
epochs = defs.nepochs
n_samples = defs.nsamp
n_gen_samples = n_samples
# n_gen_samples = 10 * n_samples
num_made = defs.nmade
num_inputs = defs.ndim
num_cond_inputs = defs.ndim_label
batch_size = defs.batch_size

# gaussian sample data parameters
if mode == defs.LINE:
    n_gaus_train = 2
    yval = defs.val
    gaus_width = defs.sigma_gaus
elif mode == defs.GRID:
    ngausx = defs.ngausx
    ngausy = defs.ngausy
    gaus_width = defs.sigma_gaus
elif mode == defs.ROOT:
    ndistx = 20
    ndisty = 20

# plot output info
flowrunname = datetime.today().date()
datarunname = datetime.today().date()
# datarunname = '2023-05-26'

###------TRAINING SAMPLES------###
if newtraindata:
    samples, labels = train_utils.makedata(mode)
else:
    datapath = 'data/gaussians_%s_%02ix%02i_run%02i.npy'%(datarunname, ngausx, ngausy, runnum)
    labelpath = 'data/gaussian_labels_%s_%02ix%02i_run%02i.npy'%(datarunname, ngausx, ngausy, runnum)
    samples = np.load(datapath)
    labels = np.load(labelpath)

###------MODEL------###
if saved_model:
    modeldir = 'model/flow_%s_%02ix%02i_run%02i'%(flowrunname, ngausx, ngausy, runnum)
else:
    modeldir = 'flow_%s_%02ix%02i'%(flowrunname, ngausx, ngausy)
    modeldir = myutils.namefile('model/', modeldir, isdir=True)

if retrain or saved_model:
    model = keras.models.load_model(modeldir, custom_objects={"lossfn": flowmodel.lossfn, "Made": flowmodel.Made})
    made_blocks = []
    for i in range(num_made):
        made_blocks.append(model.get_layer(name=f"made_{i}"))
    
    distribution, made_list = flowmodel.build_distribution(made_blocks, num_inputs)
    
else:
    model, distribution = flowmodel.compile_MAF_model(num_made, num_inputs=num_inputs, num_cond_inputs=num_cond_inputs)

if retrain or not saved_model:

    print("training begins: ", datetime.now())

    ###------TRAIN MODEL------###
    ckptpath = modeldir + "/cp-{epoch:04d}.ckpt"
    # as this is once again a unsupervised task, the target vector y is again zeros
    model.fit(x=[samples, labels],
            y=np.zeros((samples.shape[0], 0), dtype=np.float32),
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
                    epoch_end=epochs)],
            shuffle=True)

    model.save(modeldir)

    print("training ends: ", datetime.now())

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