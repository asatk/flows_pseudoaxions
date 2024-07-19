from fedhex.train._flows import *
from tensorflow_probability import distributions as tfd

# DATA
len_event = 3
len_cond = 2
num_samples = 1000

rng = np.random.default_rng(seed=0x1ace)
data = rng.normal(0, 1, size=(num_samples, len_event))
cond = rng.uniform(0, 10, size=(num_samples, len_cond))

# FLOW
num_flows = 10

made_factory = MADEFactory(params=2,
                           event_shape=len_event,
                           conditional=True,
                           conditional_event_shape=len_cond,
                           hidden_units=[16, 16])

base = tfd.Sample(tfd.Normal(loc=0., scale=1.), sample_shape=[2])

# COMPONENTS
maf = MAFComponent(shift_and_log_scale_fn_factory=made_factory)
whiten = Whiten(data=data, at_head=True)
perm = Permute(permutation=np.arange(0, len_event,)[::-1])
bn = BatchNorm()

builder = whiten + num_flows * (perm + bn + maf)
flow = builder.build(base=base)


flow.compile(optimizer="adam",
             loss=NLL,
             metrics=None)

flow.fit([data, cond],
         y=np.zeros(num_samples),
         initial_epoch=0,
         epochs=1)

print(flow.history)

samples = flow.eval(c=cond)