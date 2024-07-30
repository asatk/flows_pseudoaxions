from fedhex import flows
from fedhex.datasets import RandomGaussians, GridGaussians

from matplotlib import pyplot as plt
import numpy as np
from tensorflow_probability import distributions as tfd

from tensorflow_probability.python.distributions import Normal
import tensorflow as tf

n_dim = 2
n_observations = int(1e3)
# ds = RandomGaussians(scale=0.05, n_dim=2)
n_locs = 10
# ds = GridGaussians(scale=1 / (n_locs + 1), n_locs=n_locs, n_dim=2, dtype=np.float32)
ds = GridGaussians(scale=1 / (n_locs + 1), n_locs=n_locs, n_dim=2)
sample, labels = ds.generate(n_observations)
# ds = tf.data.Dataset.from_tensor_slices([sample, labels])
# ds_mean = ds.reduce(0, lambda state, value: state + value).numpy() / len(ds)
# ds_std = tf.sqrt(ds.reduce(0, lambda state, value: state + tf.square(value - ds_mean)) / len(ds))

made = flows.MADE(params=2,
                  event_shape=2,
                  conditional=True,
                  conditional_event_shape=2,
                  hidden_units=[16, 16, 16])
maf = flows.MAFComponent(shift_and_log_scale_fn=made,
                         conditional=True)
perm = flows.Permute(permutation=np.arange(n_dim)[::-1])

num_flows = 20

builder = num_flows * (perm + maf)
builder = builder[:-1]

base = tfd.Sample(tfd.Normal(loc=0.0, scale=1.0), sample_shape=[2])

flow = builder.build(base=base)

# flow.adapt([sample, labels])
flow.compile(optimizer="adam", loss=flows.NLL())
history = flow.fit(x=[sample, labels],
                   y=np.zeros_like(sample),
                   batch_size=1024,
                   epochs=200)

loss = history.history["loss"]
plt.plot(loss)
plt.yscale("semilogy")
plt.show()

n_samples = 1000
new_labels = np.repeat([[0.5, 0.5]], n_samples, axis=0)
new_samples = flow.sample(n_samples, c=new_labels)

ax = plt.gca()
ax: plt.Axes
ax.cla()

plt.hist2d(new_samples[:,0], new_samples[:,1], bins=(100,100), range=((-2, 2), (-2, 2)))
plt.scatter(new_labels[:,0], new_labels[:,1], c="red", label="label = (0.5, 0.5)")
ax.add_patch(plt.Circle((0.5, 0.5), radius=0.05, color="red", fill=False))
ax.add_patch(plt.Circle((0.5, 0.5), radius=0.10, color="orange", fill=False))
ax.add_patch(plt.Circle((0.5, 0.5), radius=0.15, color="yellow", fill=False))
plt.legend()
plt.show()