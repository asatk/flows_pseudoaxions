import numpy as np

import fedhex as fx
from fedhex.pretrain import generation as fxgen
from fedhex.train import SelectiveProgbarLogger, BatchLossHistory
from fedhex.train.tf import NLL

ndim = 2
nx = ny = 10
sigmax = 1 / (nx + 1) / 2
sigmay = 1 / (ny + 1) / 2
sigma = np.array([sigmax, sigmay])

gen = fx.GridGaussGenerator(cov_strat=fxgen.RepeatStrategy(fxgen.DiagCov(ndim=ndim, sigma=sigma)),
                      ngausx=nx,
                      ngausy=ny,
                      seed=0x2024)

nsamp = 100
samp, lab = gen.generate(nsamp=nsamp)
lablocs = np.unique(lab, axis=0)

batch_size = 1 << 8
epochs = 10

log_freq = 10
flow_path = "model/test-train"


def f(x, logprob):
    return -logprob

loss = None     # Default loss fn
# loss = NLL()  # import/create keras.losses.Loss class
# loss = f      # create function
mm = fx.MADEManager.import_model(flow_path, loss=loss)

epochs = int(2 * epochs)
callbacks = []
callbacks.append(SelectiveProgbarLogger(1, epoch_interval=log_freq, epoch_end=epochs))
callbacks.append(BatchLossHistory(flow_path + "/loss.npy"))
mm.train_model(dm=gen,
               batch_size=batch_size,
               initial_epoch=int(epochs/2),
               epochs=epochs,
               callbacks=callbacks,
               path=flow_path)