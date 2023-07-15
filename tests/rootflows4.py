from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

from tensorflow import keras

import defs
import flows.flowmodel as flowmodel

import rootflows2


# load model
modeldir = "./model/flow_2023-06-25_10x10_run01"

model = keras.models.load_model(modeldir, custom_objects={"lossfn": flowmodel.lossfn, "Made": flowmodel.Made})
made_blocks = []
for i in range(defs.nmade):
    made_blocks.append(model.get_layer(name=f"made_{i}"))

distribution, made_list = flowmodel.build_distribution(made_blocks, defs.ndim, hidden_layers=defs.hidden_layers, hidden_units=defs.hidden_units)

# xax = np.linspace(defs.phi_min, defs.phi_max, defs.ndistx, endpoint=True, dtype=float)
# yax = np.linspace(defs.omega_min, defs.omega_max, defs.ndisty, endpoint=True, dtype=float)
# x, y = np.meshgrid(xax, yax)
# testlabels = np.array([x.ravel(), y.ravel()]).T

# labelpath = "./data/root_labels_%s_%02ix%02i_run%02i.npy"%(flowrunname, defs.ndistx, defs.ndisty, runnum)
# trainlabels = np.load(labelpath)
# testlabels = np.unique(trainlabels, axis=0)

samplepath = "./data/root_2023-06-24_run01.npy"
labelpath = "./data/root_labels_2023-06-24_run01.npy"
normdatapath = "./data/root_norm_2023-06-24_run01.npy"

samples = np.load(samplepath)
labels = np.load(labelpath)

# testlabels = np.array([[2464., 5.125]])
# gen_labels = np.repeat(testlabels, defs.ngen, axis=0)
# gen_labels = labels

labelsunique, _ = np.unique(labels, return_inverse=True, axis=0)
gen_labels = np.repeat(labelsunique, defs.ngen, axis=0)
genlabelsunique, inverseunique = np.unique(gen_labels, return_inverse=True, axis=0)

current_kwargs = {}

for i in range(defs.nmade):
    current_kwargs[f"maf_{i}"] = {"conditional_input" : gen_labels}

gen_data = np.array(distribution.sample((gen_labels.shape[0], ), bijector_kwargs=current_kwargs))

# change out pathnames aaaaaaaa
# exit()

plt.rcParams.update({
    # "text.usetex": True,
    "font.family": "serif"
})

outpath = "./output/temp/plotgen.png"
outpath_i = "./output/temp/plotgen_dist%03i.png"
outpath_i_hist = "./output/temp/histgen_dist%03i.png"

# for i, label in enumerate(genlabelsunique):
#     gen_samples_i = gen_data[inverseunique == i]
#     rootflows2.makeplot(gen_samples_i, label, outpath_i%i)
#     rootflows2.makehist(gen_samples_i, label, outpath_i_hist%i)

fig, ax = plt.subplots()

ax.scatter(gen_data[:,0], gen_data[:,1], c="blue", s=20, alpha=0.5)
ax.scatter(genlabelsunique[:, 0], genlabelsunique[:,1], c="orange", s=20)
ax.set_xlim((0., 7000.))
ax.set_ylim((0., 15.))
ax.set_title("Generated Samples N = %i"%(len(gen_data)))
ax.set_xlabel("Reconstructed $\Phi$ Mass (GeV)")
ax.set_ylabel("Reconstructed $\omega$ Mass (GeV)")
ax.grid(visible=True)
fig.savefig(outpath)
plt.show()
