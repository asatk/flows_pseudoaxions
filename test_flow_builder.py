from matplotlib import pyplot as plt
import numpy as np

import fedhex as fx
from fedhex import flows
from tensorflow_probability import distributions as tfd

# Make data
len_event = 3
len_cond = 2
num_samples = 10000

rng = np.random.default_rng(seed=0x1ace)
data = rng.normal(0, 1, size=(num_samples, len_event))
cond = rng.uniform(0, 10, size=(num_samples, len_cond))


# Define autoregressive bijector
made = flows.MADE(params=2,
               event_shape=len_event,
               conditional=True,
               conditional_event_shape=len_cond,
               hidden_units=[16, 16],
               name="made")

# Define base distribution
base = tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[len_event])

# Create flow components
maf = flows.MAFComponent(shift_and_log_scale_fn=made, conditional=True)
perm = flows.Permute(permutation=np.arange(0, len_event, 1)[::-1])
bn = flows.BatchNorm()

# Design flow
num_flows = 20
temp = num_flows * (perm + bn + maf)
builder = temp[2:]  # remove first two layers (permute + bn)

# Build flow
flow = builder.build(base=base)

# Compile flow
flow.compile(optimizer="adam", loss=fx.flows.NLL())

# Train flow
flow.fit(x=[data, cond],
         y=np.zeros(num_samples),
         batch_size=128,
         initial_epoch=0,
         epochs=200,
         shuffle=True)

samples = flow.sample(num_samples=num_samples,
                      c=cond,
                      seed=0x2024)

plt.scatter(cond[:,0], cond[:,1])
plt.hist2d(samples[:,0], samples[:,1], bins=(100,100))
plt.show()