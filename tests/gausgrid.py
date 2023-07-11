import numpy as np
from matplotlib import pyplot as plt

import defs as defs
import utils as myutils

# make gaussian sample data
ngausx = ngausy = 2
n_samples = 1000
gaus_width = 0.025

labels = myutils.normalize_labels_grid_2d(
    myutils.train_labels_grid_2d(ngausx, ngausy))
gaus_covs = myutils.cov_change_none(labels,
    myutils.cov_xy(gaus_width))
gaus_pts = myutils.gaus_point_grid_2d(labels)

gaussians, gaussian_labels = myutils.sample_real_gaussian(n_samples, labels, gaus_pts, gaus_covs)

# print(labels)
# print(gaussian_labels)
# print(gaus_covs)
print(gaus_pts)
print(gaussians.shape)


# [hex(int(255/n * i + 255/(2*n))) for i in range(n)]
# colors = ['#99%s99'%(hex(int(255/n_gaus_train * i + 255/(2*n_gaus_train))))[-2:] for i in range(n_gaus_train)]

# fig, ax = plt.subplots(111)

for i in range(ngausx):
    for j in range (ngausy):
        color = '#%02x80%02x'%(int(255/ngausx * i + 255/(2*ngausx)), int(255/ngausy * j + 255/(2*ngausy)))
        plt.scatter(
                gaussians[(i * ngausy + j) * n_samples: (i * ngausx + j + 1) * n_samples - 1, 0],
                gaussians[(i * ngausy + j) * n_samples: (i * ngausx + j + 1) * n_samples - 1, 1],
                c=color, edgecolor='none', alpha=0.5, s=25, label="group (%i, %i)"%(i, j))

plt.scatter(gaus_pts[:,0], gaus_pts[:,1], c='red', edgecolor='none', alpha=0.5, s=25)
# plt.legend(loc=1)
plt.grid(visible=True)
plt.xlim((defs.xmin, defs.xmax))
plt.ylim((defs.ymin, defs.ymax))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('(%i, %i) Gaussians'%(ngausx, ngausy))
plt.show()