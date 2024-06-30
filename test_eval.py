from fedhex import MAFManager, UnitCubeGaussGenerator
import numpy as np
from matplotlib import pyplot as plt

generator = UnitCubeGaussGenerator(ndim=2, ngaus=10)
samples, labels = generator.generate(nsamp=1000)

rng = np.random.default_rng(seed=0x2024)
val_data_samp, val_data_cond = generator.generate(nsamp=1000)
# val_data_inds = rng.choice(10 * 10 * 1000, size=10 * 1000, replace=False)
# val_data = ([val_data_samp[val_data_inds],
#              val_data_cond[val_data_inds]],
#              np.zeros(10 * 1000))
val_data = ([val_data_samp,
             val_data_cond],
             np.zeros(10 * 10 * 1000))

plt.hist2d(samples[:,0], samples[:,1], bins=(100,100))
plt.scatter(labels[:,0], labels[:,1], c="orange")
plt.show()

model = MAFManager(num_flows=10,
                   len_event=2,
                   len_cond_event=2,
                   hidden_units=[32, 32],
                   activation="relu")

model.compile(prior=None,
              optimizer=None,
              loss=None)

history = model.train(dm=generator,
                      batch_size=(1 << 10),
                      epochs=10,
                      validation_data=val_data)

new_samples = model.eval(cond=labels,
                         dm=generator)

loss = np.array(history.history["loss"])
val_loss = np.array(history.history["val_loss"])


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1: plt.Axes
ax1.hist2d(new_samples[:,0], new_samples[:,1], bins=(100,100))
ax1.scatter(labels[:,0], labels[:,1], c="orange")

ax2: plt.Axes
ax2.set_title("Loss over 10 epochs")
ax2.plot(np.array([loss, val_loss]).T,
         label=["train", "valid"])
ax2.legend()

plt.show()