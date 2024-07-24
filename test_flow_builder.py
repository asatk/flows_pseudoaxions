from fedhex.train._flows import *
from tensorflow_probability import distributions as tfd
from fedhex.train import MADE, NLL

# Make data
len_event = 3
len_cond = 2
num_samples = 1000

rng = np.random.default_rng(seed=0x1ace)
data = rng.normal(0, 1, size=(num_samples, len_event))
cond = rng.uniform(0, 10, size=(num_samples, len_cond))


# Define autoregressive bijector
made = MADE(params=2,
            event_shape=len_event,
            conditional=True,
            conditional_event_shape=len_cond,
            hidden_units=[16, 16],
            name="made")

# Define base distribution
base = tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[len_event])

# Create flow components
maf = MAFComponent(shift_and_log_scale_fn=made, conditional=True)
perm = Permute(permutation=np.arange(0, len_event, 1)[::-1])
bn = BatchNorm()

# Set up full flow
num_flows = 10
temp = num_flows * (perm + bn + maf)
builder = temp[2:]  # remove first two layers (permute + bn)

# Build flow
flow = builder.build(base=base)

# Compile flow
flow.compile(optimizer="adam", loss=NLL())

# Train flow
flow.fit(x=[data, cond],
         y=np.zeros(num_samples),
         batch_size=128,
         initial_epoch=0,
         epochs=10,
         shuffle=True)
