from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras

import defs
import flowmodel

### Plotting functions (usually stuffed away in some module)

# Scatter plot of samples for one label
def plot_one(samples, label, outpath):
    fig, ax = plt.subplots()
    ax.scatter(samples[:,0], samples[:,1], c="green", s=20, alpha=0.5)
    ax.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Samples for (%f, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    ax.set_xlim((0., 7000.))
    ax.set_ylim((0., 15.))
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
    h, xedges, yedges, img = ax.hist2d(samples[:,0], samples[:,1], bins=(nbinsx, nbinsy), range=((defs.phi_min, defs.phi_max), (defs.omega_min, defs.omega_max)))
    plt.scatter(label[0], label[1], c="red", s=20)
    ax.set_title("Samples for (%f, %.3f)\nN = %i"%(label[0], label[1], len(samples)))
    ax.set_xlim((0., 7000.))
    ax.set_ylim((0., 15.))
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
modeldir = "./model/flow_2023-06-25_10x10_run01"

samplepath = "./data/root_2023-06-24_run01.npy"
labelpath = "./data/root_labels_2023-06-24_run01.npy"
normdatapath = "./data/root_norm_2023-06-24_run01.npy"

outpath_all = "./output/temp/06-26_plotgen.png"
outpath_i = "./output/temp/06-26_plotgen%03i.png"
outpath_i_hist = "./output/temp/06-26_histgen%03i.png"

# Load an entire model (not checkpoint) from specified directory
model = keras.models.load_model(modeldir, custom_objects={"lossfn": flowmodel.lossfn, "Made": flowmodel.Made})
made_blocks = []
for i in range(defs.nmade):
    made_blocks.append(model.get_layer(name=f"made_{i}"))

# Create just the bijection `distribution` necessary for generating new samples
distribution, made_list = flowmodel.build_distribution(made_blocks, defs.ndim)

# Load training data
samples = np.load(samplepath)
labels = np.load(labelpath)
normdata = np.load(normdatapath)

### Create test labels for which new data are generated:

# (1) can define new labels at points you wish to evalute the flow...
gen_labels = np.repeat([[2464., 5.125]], defs.ngen, axis=0)

# (2) ...or use training labels to compare generated data to training data.
labelsunique, _ = np.unique(labels, return_inverse=True, axis=0)
gen_labels = np.repeat(labelsunique, defs.ngen, axis=0)
genlabelsunique, inverseunique = np.unique(gen_labels, return_inverse=True, axis=0)

# Define the conditional input (labels) for the flow to generate
current_kwargs = {}
for i in range(defs.nmade):
    current_kwargs[f"maf_{i}"] = {"conditional_input" : gen_labels}

# Generate the data given the test labels!
gen_data = np.array(distribution.sample((gen_labels.shape[0], ), bijector_kwargs=current_kwargs))

# Post-processing of data only consists of de-normalzing the output
gen_data = gen_data * normdata[1] + normdata[0]

# Matplotlib settings
plt.rcParams.update({
    "font.family": "serif"
})

# Plot scatter plots and histograms for each label
for i, label in enumerate(genlabelsunique):
    gen_samples_i = gen_data[inverseunique == i]
    plot_one(gen_samples_i, label, outpath_i%i)
    hist_one(gen_samples_i, label, outpath_i_hist%i)

# Plot scatter plot for all data
plot_all(gen_data, genlabelsunique, outpath_all)

# Numerical analysis
# ...