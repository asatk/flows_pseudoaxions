"""
This module holds definitions of (hyper)parameters used to define the
geometry and data of the problem being interpolated.
"""

### --- RUN INFO --- ###
# Paths
data_dir = "data"
model_dir = "model"
output_dir = "output"
root_dir = "root/10x10box_10k_jun2023"
flow_name = "06-30_BOX02"
data_name = "06-30_BOX01"

flow_path = model_dir + "/" + flow_name
samples_path = data_dir + "/" + data_name + "_samples.npy"
labels_path = data_dir + "/" + data_name +"_labels.npy"
normdata_path = data_dir + "/" + data_name + "_normdata.npy"
normdata_path = data_dir + "/" + data_name + "_normdatacond.npy"

# Run flags
newdata = False
normalize = True
newmodel = False
newanalysis = True

### --- MODE CONSTANTS --- ###
LINE = 0    #2-D gaussians along a line
GRID = 1    #2-D gaussians in a grid
ROOT = 2    #2-D distributions in a grid, usuall
mode = ROOT

### --- Network Hyperparameters --- ###
seed = 0xace1ace1ace1ace1   #seed for RNG
nepochs = 250               #iterations to train flow
epoch_resume = 0            #iterations to resume training a trained model
epoch_save = 50             #iterations to save flow state in checkpoints
nmade = 10                  #MADE blocks in masked autoregressive flow
ndim = 2                    #dimensions of the sample data
ndim_label = 2              #dimensions of the conditional data (labels)
base_lr = 1.0e-3            #learning rate range: keep btwn [1e-3, 1e-6]
end_lr = 1.0e-4             #ditto
batch_size = 1024           #num samples in each epoch's minibatches
ngen = 500                  #num samples for analysis/generation
hidden_layers = 1           #num layers in the MADE block (flow complexity)
hidden_units = 128          #num parameters in each MADE block layer

### --- Geometry/Problem Parameters --- ###
# Use-case Parameters
phi_min = 0.                #min phi (x) val in GeV
phi_max = 7000.             #max phi (x) val in GeV
phi_bins = 70               #bins in phi (x)
omega_min = 0.              #min omega (y) val in GeV
omega_max = 15.             #max omega (y) val in GeV
omega_bins = 150            #bins in omega (y)
event_threshold = 0.011

# Gaussian Parameters
ngausx = 10                 #gaussians in x dimension
ngausy = 10                 #gaussians in y dimension
nsamp = 1000                #num samples in training gaussians
sigma_gaus = 0.025          #std dev/width of training gaussians
val = 0.5                   #special value used for circ/line gaussian problems
xmin = 0.                   #min xval of grid
xmax = 1.                   #max xval of grid
xbins = 200                 #bins in x
ymin = 0.                   #min yval of grid
ymax = 1.                   #max yval of grid
ybins = 200                 #bins in y
xcov_change_linear_max_factor = 4.   #max change in x-growth
ycov_change_linear_max_factor = 2.   #max change in y-growth
cov_change_skew_rate = 2.0           #radian angle rotation

nworkers = 8                #worker processes used in multiprocessing