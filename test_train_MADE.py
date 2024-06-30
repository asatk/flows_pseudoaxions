import numpy as np
from tensorflow import keras

import fedhex as fx
from fedhex.pretrain import generation as fxgen
from fedhex.train import SelectiveProgbarLogger, BatchLossHistory
from fedhex.train import NLL

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


nmade = 1
ninputs = ncinputs = ndim
hidden_units = [16, 16]
batch_size = 1 << 12
epochs = 10


log_freq = 10
flow_path = "model/test-train"
callbacks = []
callbacks.append(SelectiveProgbarLogger(1, epoch_interval=log_freq, epoch_end=epochs))
callbacks.append(BatchLossHistory(flow_path + "/loss.npy"))


model = fx.MAFManager(num_flows=nmade,
               len_event=ninputs,
               len_cond_event=ncinputs,
               hidden_units=hidden_units,
               activation="relu")

opt = keras.optimizers.Adam(learning_rate=1e-3)

def _loss(x, y):
    return -y

loss = None     # Default loss fn
# loss = NLL()  # import/create keras.losses.Loss class
# loss = _loss  # create function

model.compile(optimizer=opt, loss=loss)

model.train(dm=gen,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=callbacks,
                  flow_path=flow_path)