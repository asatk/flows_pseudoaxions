from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
from tensorflow import keras

import data_utils as dutils
import defs
import flowmodel

### Plotting functions (usually stuffed away in some module)

# Scatter plot of samples for one label
def plot_one(samples, label, outpath):

    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c="green", s=20, alpha=0.5)
    # ax.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Generated Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    # ax.set_xlim((defs.phi_min, defs.phi_max))
    # ax.set_ylim((defs.omega_min, defs.omega_max))
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
    # plt.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Generated Samples for (%g, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    # ax.set_xlim((defs.phi_min, defs.phi_max))
    # ax.set_ylim((defs.omega_min, defs.omega_max))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    fig.colorbar(img)
    fig.savefig(outpath)
    plt.close()

# Scatter plot of samples for all labels
def plot_all(samples, labels_unique, outpath):

    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c="blue", s=20, alpha=0.5)
    # ax.scatter(labels_unique[:, 0], labels_unique[:,1], c="orange", s=20)
    # ax.set_xlim((defs.phi_min, defs.phi_max))
    # ax.set_ylim((defs.omega_min, defs.omega_max))
    ax.set_title("Generated Samples N = %i"%(len(samples)))
    ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
    ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
    ax.grid(visible=True)
    fig.savefig(outpath)
    plt.show()

# Relevant paths
modeldir = "model/06-27_BOX04"

samples_path = "data/06-27_BOX03_samples.npy"
labels_path = "data/06-27_BOX03_labels.npy"
# normdata_path = "data/06-27_BOX01_normdata.npy"

outpath_all = "output/06-27_boxout/plotgen.png"
outpath_i = "output/06-27_boxout/plotgen%03i.png"
outpath_i_hist = "output/06-27_boxout/histgen%03i.png"

# Load an entire model (not checkpoint) from specified directory
model = keras.models.load_model(modeldir, custom_objects={"lossfn": flowmodel.lossfn, "Made": flowmodel.Made})
made_blocks = []
for i in range(defs.nmade):
    made_blocks.append(model.get_layer(name=f"made_{i}"))

# Create just the bijection `distribution` necessary for generating new samples
distribution, made_list = flowmodel.build_distribution(made_blocks, defs.ndim, hidden_layers=defs.hidden_layers, hidden_units=defs.hidden_units)

# Load training data
samples = np.load(samples_path)
labels = np.load(labels_path)

### Create test labels for which new data are generated:

# (1) can define new labels at points you wish to evalute the flow...
gen_labels = np.repeat([[2464., 5.125]], defs.ngen, axis=0)

# (2) ...or use training labels to compare generated data to training data.
labelsunique, inverseunique = np.unique(labels, return_inverse=True, axis=0)
gen_labels = np.repeat(labelsunique, defs.ngen, axis=0)
genlabelsunique, geninverseunique = np.unique(gen_labels, return_inverse=True, axis=0)

# Define the conditional input (labels) for the flow to generate
current_kwargs = {}
for i in range(defs.nmade):
    current_kwargs[f"maf_{i}"] = {"conditional_input" : gen_labels}

# Generate the data given the test labels!
gen_data = np.array(distribution.sample((gen_labels.shape[0], ), bijector_kwargs=current_kwargs))

print(np.min(gen_data, axis=0))
print(np.max(gen_data, axis=0))
print(np.mean(gen_data, axis=0))
print(np.std(gen_data, axis=0))

# Post-processing of data only consists of de-normalzing the output
# samples = dutils.unwhiten(samples, normdata_path)
# gen_data = dutils.unwhiten(gen_data, normdata_path)

# Matplotlib settings
plt.rcParams.update({
    "font.family": "serif"
})

# grouped_data = [samples[inverseunique == i] for i in range(len(labelsunique))]
# out_paths = [outpath_i%i for i in range(len(labelsunique))]
# out_paths_hist = [outpath_i_hist%i for i in range(len(labelsunique))]

# # Plot scatter plots and histograms for each label
# with mp.Pool(defs.nworkers) as pool:
#     pool.starmap(plot_one, zip(grouped_data, labelsunique, out_paths))
#     pool.starmap(hist_one, zip(grouped_data, labelsunique, out_paths_hist))

grouped_data = [gen_data[geninverseunique == i] for i in range(len(genlabelsunique))]
out_paths = [outpath_i%i for i in range(len(genlabelsunique))]
out_paths_hist = [outpath_i_hist%i for i in range(len(genlabelsunique))]

# Plot scatter plots and histograms for each label
with mp.Pool(defs.nworkers) as pool:
    pool.starmap(plot_one, zip(grouped_data, genlabelsunique, out_paths))
    pool.starmap(hist_one, zip(grouped_data, genlabelsunique, out_paths_hist))

# Plot scatter plot for all data
plot_all(gen_data, genlabelsunique, outpath_all)
# plot_all(samples, labelsunique, outpath_all)

# Numerical analysis
# ...
