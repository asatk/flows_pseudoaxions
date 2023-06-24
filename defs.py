"""
This module holds definitions of (hyper)parameters used to define the
geometry and data of the problem being interpolated.
"""

### --- Network/Run parameters --- ###
seed = 0xace1ace1ace1ace1 #seed for RNG
nepochs = 200            #num epochs for training flow
epoch_resume = 0        #num iterations to resume training on a trained flow model
epoch_save = 50        #num iterations to save GAN state
nmade =  10         # number of MADE blocks in masked autoregressive flow
ndim = 2             #dimensions of the data, i.e., 2 for 2-D problem - also referred to as latent dimension of GAN? understand what that means
ndim_label = 2          #dimensions of the labels
base_lr = 1.0e-3               #learning rate: keep btwn [1e-3, 1e-6]
end_lr =  1.0e-4

### --- Geometry/Problem Parameters --- ###
# Use-case Parameters
phi_min = 0.            #min phi (x) val in GeV
phi_max = 6000.         #max phi (x) val in GeV
phi_bins = 60          #bins in phi (x)
omega_min = 0.          #min omega (y) val in GeV
omega_max = 10.         #max omega (y) val in GeV
omega_bins = 100        #bins in omega (y)
ndistx = 10
ndisty = 10

# Gaussian Line Parameters
ngaus = 5               #num gaussians from which training data is created
ngausx = 10
ngausy = 10
ngausplot = 5           #num gaussians to plot/use in analysis
nsamp = 1000           #num samples in training gaussians
ngen = 1000             # nump samples for analysis/generation
sigma_gaus = 0.025     #std dev of training gaussians
val = 0.5               #specific value used for gaussian problems (circ/line)
xmin = 0.               #min xval of grid
xmax = 1.               #max xval of grid
xbins = 200             #bins in x
ymin = 0.               #min yval of grid
ymax = 1.               #max yval of grid
ybins = 200             #bins in y
xcov_change_linear_max_factor = 4.   #max change in growth in x
ycov_change_linear_max_factor = 2.   #max change in growth in y
cov_change_skew_rate = 2.0            #radian angle rotation

batch_size = nsamp           # size of the batch of data trained on per epoch

### --- MODE CONSTANTS --- ###
LINE = 0
GRID = 1
ROOT = 2
mode = ROOT