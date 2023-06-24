from matplotlib import pyplot as plt
import numpy as np
import os
import uproot as up

np.set_printoptions(threshold=1e6, edgeitems=10)

up.default_library = "np"
cutstr = "(CBL_Region == 1) & (pt > 220)"
# cutstr = "(CBL_Region == 1) & (pt > 220) & (CBL_RecoPhi_twoprongindex != -1)"
# tempcutstr = "CBL_RecoPhi_twoprongindex != -1"
ptstr = "Photon_pt[CBL_RecoPhi_photonindex]"
phistr = "CBL_RecoPhi_mass"
omegastr = "TwoProng_massPi0"
labelphistr = "GenPhi_mass"
labelomegastr = "GenOmega_mass"
indexstr = "CBL_RecoPhi_twoprongindex"

filepath = "./root/10x10box_10k_jun2023/Phi_1280_omega_3p175/2023-06-12-11-53-55/v1p0-78-8cf4/ATTOAODv1p2c2_0.root"

if os.stat(filepath).st_size == 0:
    print("--- ^ empty file ^ ---")
    exit()

datafile = up.open(filepath)
events = datafile["Events;1"]

# arrs = events.arrays([phistr, 'omega'], cut=cutstr, aliases={'omega': '%s[%s]'%(omegastr, indexstr), 'pt': ptstr})
arrs = events.arrays([phistr, 'omega'], cut=cutstr, aliases={'omega': '%s[%s]'%(omegastr, indexstr), 'pt': ptstr})

# print(arrs[cutvarstr])

phi = arrs[phistr]
omega = arrs['omega']

print(len(phi), len(omega))
print(phi)
print(omega)


datafile.close()
