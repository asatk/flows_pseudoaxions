from tensorflow_probability.python import distributions as tfd

import fedhex as fx
from fedhex.pretrain import generation as fxgen


nx = ny = 10
sigma = 1 / (nx + 1) / 2

dm = fx.GridGaussGenerator(
    cov_strat=fxgen.RepeatStrategy(
        s=fxgen.DiagCov(ndim=2, sigma=sigma)),
    ngausx=nx,
    ngausy=ny
)

nsamp = 1000
data, cond = dm.generate(nsamp=nsamp)
data_norm, cond_norm = dm.preproc()

num_flows = 10
hidden_units = [16, 16]

maf = fx.MAFManager(num_flows=num_flows,
                    len_event=2,
                    len_cond_event=2,
                    hidden_units=hidden_units)

prior = tfd.Sample(
    tfd.Normal(loc=[0.], scale=[sigma]),
    sample_shape=[2])

maf.compile(prior=prior,
            optimizer=None,
            loss=None)

batch_size = 1024
initial_epoch = 0
epochs = 10
flow_path = "./model/test-prior"

maf.train(dm,
          data=data,
          cond=cond,
          batch_size=batch_size,
          initial_epoch=initial_epoch,
          epochs=epochs,
          verbose=1,
          flow_path=flow_path)