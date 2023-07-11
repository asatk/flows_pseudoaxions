from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from tensorflow import keras

import data_utils as dutils
import defs
import flowmodel

### Plotting functions (usually stuffed away in some module)

# Scatter plot of samples for one label
def plot_one(samples, label, outpath):

    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c="green", s=20, alpha=0.5)
    ax.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Generated Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    ax.set_xlim((defs.phi_min, defs.phi_max))
    ax.set_ylim((defs.omega_min, defs.omega_max))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    ax.grid(visible=True)
    fig.savefig(outpath)
    plt.close()

# Histogram of samples for one label
def hist_one(samples, label, outpath):
    nbinsx = 50
    nbinsy = 50
    
    fig, ax = plt.subplots()
    _, _, _, img = ax.hist2d(samples[:,0], samples[:,1], bins=(nbinsx, nbinsy), range=((defs.phi_min, defs.phi_max), (defs.omega_min, defs.omega_max)))
    plt.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Generated Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    ax.set_xlim((defs.phi_min, defs.phi_max))
    ax.set_ylim((defs.omega_min, defs.omega_max))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    fig.colorbar(img)
    fig.savefig(outpath)
    plt.close()

# Scatter plot of samples for all labels
def plot_all(samples, labels_unique, outpath):

    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c="blue", s=20, alpha=0.5)
    ax.scatter(labels_unique[:, 0], labels_unique[:,1], c="orange", s=20)
    ax.set_xlim((defs.phi_min, defs.phi_max))
    ax.set_ylim((defs.omega_min, defs.omega_max))
    ax.set_title("Generated Samples N = %i"%(len(samples)))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    ax.grid(visible=True)
    fig.savefig(outpath)
    plt.show()

# Relevant paths
data_prefix = "data/07-04_TEST01"
model_dir = "model/07-04_TEST01/"
out_dir = "output/07-04_box1_test2/"

samples_path = data_prefix + "_samples.npy"
labels_path = data_prefix + "_labels.npy"
normdata_path = data_prefix + "_normdata.npy"
normdatacond_path = data_prefix + "_normdatacond.npy"

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
else:
    print("Continuing this analysis may overwrite some files in output directory: %s."%out_dir)
    c = " "
    while c not in ["y", "N"]:
        c = input("Proceed? [y/N]")
        if c == "N":
            exit()
        elif c != "y":
            print("Response: '%s' is not a valid response. Enter 'y' or 'N'."%c)

outpath_all = out_dir + "plotgen.png"
outpath_i = out_dir + "plotgen%03i.png"
outpath_i_hist = out_dir + "histgen%03i.png"

# Load an entire model (not checkpoint) from specified directory
model = keras.models.load_model(model_dir, custom_objects={"lossfn": flowmodel.lossfn, "Made": flowmodel.Made})
made_blocks = []
for i in range(defs.nmade):
    made_blocks.append(model.get_layer(name=f"made_{i}"))

# Create just the bijection `distribution` necessary for generating new samples
distribution, made_list = flowmodel.build_distribution(made_blocks, defs.ndim, hidden_layers=defs.hidden_layers, hidden_units=defs.hidden_units)

# Load unwhitened training data
data = np.load(samples_path)
datacond = np.load(labels_path)
samples = dutils.dewhiten(data, normdata_path)
labels = dutils.dewhiten(datacond, normdata_path)
labels_unique, inverse_unique = np.unique(labels, return_inverse=True, axis=0)

### Create test labels for which new data are generated:

# (1) can define new labels at points you wish to evalute the flow...
# By picking unwhitened labels (not pre-processed for use by the flow), these
# labels will have to be whitened in order to be logical to the net.
gen_labels = np.repeat([[2464, 5.125]], defs.ngen, axis=0) # arb. label
# whiten the label/convert to normalized net-friendly form
normdatacond = np.load(normdata_path, allow_pickle=True).item()
min_norm = normdatacond["min"]
max_norm = normdatacond["max"]
mean_norm = normdatacond["mean"]
std_norm = normdatacond["std"]

# manually whiten data from loaded params to convert a meaningful label
# into a regression label that the network will interpret correctly.
data_temp = (gen_labels - min_norm) / (max_norm - min_norm)
data_temp = np.log(1 / ((1 / data_temp) - 1))
gen_datacond = (data_temp - mean_norm) / std_norm

# (2) ...or use training labels to compare generated data to training data.
# These instructions group samples & labels by unique label
# These labels are already whitened since they are the ones used by the network
# and saved in the training data generation process. In order for us to
# interpret these labels logically, we must unwhiten them.
datacond_unique = np.unique(datacond, axis=0)
gen_datacond = np.repeat(datacond_unique, defs.ngen, axis=0)

# Get only the unique labels and a mapping to where each label occurs in the list of all labels
gen_datacond_unique, gen_inverse_unique = np.unique(gen_datacond, return_inverse=True, axis=0)
gen_labels_unique = dutils.dewhiten(gen_datacond_unique, normdatacond_path)
gen_labels = gen_labels_unique[gen_inverse_unique]  # gen all labels if not already defined

# Define the conditional input (labels) for the flow to generate
current_kwargs = {}
for i in range(defs.nmade):
    current_kwargs[f"maf_{i}"] = {"conditional_input" : gen_datacond}

# Generate the data given the test labels! De-normalize them once generated.
gen_data = np.array(distribution.sample((gen_datacond.shape[0], ), bijector_kwargs=current_kwargs))
gen_samples = dutils.dewhiten(gen_data, normdata_path)

print(np.min(gen_samples, axis=0))
print(np.max(gen_samples, axis=0))
print(np.mean(gen_samples, axis=0))
print(np.std(gen_samples, axis=0))

# Matplotlib settings
plt.rcParams.update({
    "font.family": "serif"
})

# grouped_samples = [samples[inverse_unique == i] for i in range(len(labels_unique))]
# out_paths = [out_path_i%i for i in range(len(labels_unique))]
# out_paths_hist = [out_path_i_hist%i for i in range(len(labels_unique))]

# # Plot scatter plots and histograms for each label
# with mp.Pool(defs.nworkers) as pool:
#     pool.starmap(plot_one, zip(grouped_samples, labels_unique, out_paths))
#     pool.starmap(hist_one, zip(grouped_samples, labels_unique, out_paths_hist))

grouped_gen_samples = [gen_samples[gen_inverse_unique == i] for i in range(len(gen_labels_unique))]
out_paths = [outpath_i%i for i in range(len(gen_labels_unique))]
out_paths_hist = [outpath_i_hist%i for i in range(len(gen_labels_unique))]

# Plot scatter plots and histograms for each label
with mp.Pool(defs.nworkers) as pool:
    pool.starmap(plot_one, zip(grouped_gen_samples, gen_labels_unique, out_paths))
    pool.starmap(hist_one, zip(grouped_gen_samples, gen_labels_unique, out_paths_hist))

# Plot scatter plot for all data
plot_all(gen_samples, gen_labels_unique, outpath_all)
# plot_all(samples, labelsunique, outpath_all)

# Numerical analysis
# ...
