import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow.test as tft
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import train_utils_flows as myutils
import defs_flows as defs
import flow

# gaussian sample data parameters
n_gaus_train = 2
n_samples = 1000
gaus_width = 0.025
yval = defs.val

# Are the data conditioned or not
conditioned = False
num_cond_inputs = 1
# choose label via hardcoding label:
label = 1./(n_gaus_train + 1)

if not conditioned:
    num_cond_inputs = None

#Define two-block made network
num_made = 10
epochs = 100

model, distribution, made_list = flow.compile_MAF_model(num_made, num_inputs=2, num_cond_inputs=num_cond_inputs, return_layer_list=True)

# define intermediate outputs for later
feat_extraction_dists = flow.intermediate_flows_chain(made_list)

# assign the 1D labels for each gaussian
labels = myutils.normalize_labels_line_1d(
    myutils.train_labels_line_1d(n_gaus_train))
# calculate expected center of each gaussian
gaus_pts = myutils.gaus_point_line_1d(labels, yval)
# generate cov mtx for each gaussian
# gaus_cov_indv = myutils.cov_skew(gaus_width, gaus_width/2., gaus_width/2., gaus_width)
gaus_cov_indv = myutils.cov_xy(gaus_width)
gaus_covs = myutils.cov_change_const(gaus_pts, gaus_cov_indv)
# samples each gaussian for n_samples points, each with an associated label
gaussians, gaussian_labels = \
    myutils.sample_real_gaussian(n_samples, labels, gaus_pts, gaus_covs)

# scaler = StandardScaler()
# scaler.fit(gaussians)

# gaussians_scaled = scaler.transform(gaussians)
gaussians_scaled = np.array(gaussians)

if conditioned:
    data = [gaussians_scaled, gaussian_labels]
else:
    data = gaussians_scaled

batch_size = n_gaus_train * n_samples // (10)
steps = gaussians_scaled.shape[0] // batch_size
#as this is once again a unsupervised task, the target vector y is again zeros
model.fit(x=data,
          y=np.zeros((gaussians_scaled.shape[0], 0), dtype=np.float32),
          batch_size=batch_size,
          epochs=epochs,
          steps_per_epoch=steps,
          verbose=1,
          shuffle=True)

current_kwargs = {}
if conditioned:
    # if you want random labels use this line
    # cond = np.random.randint(n_gaus_train, size=(n_samples,1))
    
    cond = label * np.ones((n_samples, 1))
    for i in range(num_made):
        current_kwargs[f"maf_{i}"] = {'conditional_input' : cond}


#generate 1000 new samples
samples_scaled = distribution.sample((n_samples, ), bijector_kwargs=current_kwargs)

# samples = scaler.inverse_transform(samples_scaled)
samples = np.array(samples_scaled)

#transform them like we did when plotting the original dataset
# samples = StandardScaler().fit_transform(samples)

# statistics on results
mean_train = np.mean(gaussians[0:n_samples], axis=0)
cov_train = np.cov(gaussians[0:n_samples].T)
print("Training Data (1st Gaus): \n - Mean: ", mean_train, "\n - Cov: ", cov_train, "\n - Labels: ", labels)
print("\nGenerating For Label %0.4f"%(label))
mean_exp = np.array([label, 0.5])
cov_exp = myutils.cov_change_const(np.array([[label,0.5]]), gaus_cov_indv)
print("\nExpected Stats: \n - Mean: ", mean_exp, "\n - Cov: ", cov_exp)
mean_gen = np.mean(samples, axis=0)
cov_gen = np.cov(samples.T)
print("\nGenerated Data: \n - Mean", mean_gen, "\n - Cov", cov_gen)

#plot the results
plt.scatter(gaussians[:, 0], gaussians[:, 1], s=25, color='darkgreen', alpha=0.4, label='training data')
plt.scatter(samples[:, 0], samples[:, 1], s=25, color='darkblue', alpha=0.4, label='generated data')
plt.scatter(labels, np.ones(n_gaus_train) * 0.5, s=25, color='red')
plt.scatter(label, 0.5, s=25, color='yellow')
# plt.scatter(gaus_pts[:, 0], gaus_pts[:, 1], color='red')
plt.legend(loc=1)
plt.grid(visible=True)
plt.xlim((defs.xmin, defs.xmax))
plt.ylim((defs.ymin, defs.ymax))
plt.title("Flow Output and Training Samples")
plt.savefig("./flow_output_gaus.png")
plt.show()
plt.close()


# #plot the intermediate results
# for i, d in enumerate(feat_extraction_dists):
#     out_scaled = d.sample((n_samples, ), bijector_kwargs=current_kwargs)
# #   out = StandardScaler().fit_transform(out)
#     # out = scaler.inverse_transform(out_scaled)
#     out = out_scaled
#     plt.scatter(out[:, 0], out[:, 1], color='darkblue', s=25)
#     plt.grid(visible=True)
#     plt.xlim((defs.xmin, defs.xmax))
#     plt.ylim((defs.ymin, defs.ymax))
#     plt.title("Intermediate Transformation f_%i"%(i))
#     plt.savefig("./flow_%i_output_gaus.png"%(i))
#     plt.close()
# #   plt.show()