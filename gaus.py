import numpy as np
from matplotlib import pyplot as plt

import defs_flows as defs
import train_utils_flows as myutils

# make gaussian sample data
n_gaus_train = 2
n_samples = 1000
gaus_width = 0.025
yval = 0.5

labels = myutils.normalize_labels_line_1d(
    myutils.train_labels_line_1d(n_gaus_train))
gaus_covs = myutils.cov_change_const(labels,
    myutils.cov_xy(gaus_width))
gaus_pts = myutils.gaus_point_line_1d(labels, yval)

gaussians, gaussian_labels = myutils.sample_real_gaussian(n_samples, labels, gaus_pts, gaus_covs)

print(labels)
# print(gaussian_labels)
# print(gaus_covs)
print(gaus_pts)
print(gaussians.shape)


# [hex(int(255/n * i + 255/(2*n))) for i in range(n)]
# colors = ['#99%s99'%(hex(int(255/n_gaus_train * i + 255/(2*n_gaus_train))))[-2:] for i in range(n_gaus_train)]

# fig, ax = plt.subplots(111)

for i in range(n_gaus_train):
    color = '#%02x8080'%(int(255/n_gaus_train * i + 255/(2*n_gaus_train)))
    plt.scatter(gaussians[i * n_samples: (i + 1) * n_samples - 1, 0], gaussians[i * n_samples: (i + 1) * n_samples - 1, 1], c=color, edgecolor='none', alpha=0.5, s=25, label="group %i"%(i))
    plt.scatter(gaussian_labels, np.ones(gaussian_labels.shape[0]) * yval, c='red', edgecolor='none', alpha=0.5, s=25)

plt.legend(loc=1)
plt.grid(visible=True)
plt.xlim((defs.xmin, defs.xmax))
plt.ylim((defs.ymin, defs.ymax))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('%i Gaussians'%(n_gaus_train))
plt.show()