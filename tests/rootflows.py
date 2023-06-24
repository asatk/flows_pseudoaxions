from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
import uproot as up
import os

up.default_library = "np"

cwd = os.getcwd()
if 'root' not in os.listdir(cwd):
    print('\'root\' directory not in the cwd.\nExiting...')
    exit()

rootd = cwd + '/root/attoaod2_Box30by30-10k-Mar2023/'
files = os.listdir(rootd)

p = re.compile("\w+(\d)p(\d+)_(\d+)\.root$")
matches = [p.match(fname) for fname in files]

# for fname, m in zip(files, matches):
#     print(fname, m.group(1), m.group(2), m.group(3))

# the old stuff
'''
        # drawing variables and cuts
        phi_var = "Obj_PhotonTwoProng.mass"
        omega_var = "TwoProng_MassPi0[0]"
        draw_str = omega_var+":"+phi_var
        cut_str = "nTwoProngs>0 && nIDPhotons>0 && Obj_PhotonTwoProng.dR>0.1 && Obj_PhotonTwoProng.part2_pt>150"

        # Tree name in Ntuplizer files
        tree_name = "twoprongNtuplizer/fTree" 
'''

testfile = up.open(rootd + files[205])
print(files[500])

print(testfile.classnames())

# print(testfile["Events;1"].keys())

events = testfile["Events;1"]
# print(testfile["Runs;1"].keys())

# for k in events.keys():
#     print(k)

print(events["GenPhi_mass"].array())
print(events["GenOmega_mass"].array())

cutstr = "(Region == 1) & (HighPtIdPhoton_pt > 220)"
phistr = "RecoPhi_mass"
omegastr = "TwoProng_massPi0"
labelphistr = "GenPhi_mass"
labelomegastr = "GenOmega_mass"
indexstr = "RecoPhi_twoprongindex"
# omegastr = "TwoProng_massPi0[RecoPhi_twoprongindex]"
# data = events.arrays([phistr,omegastr], cutstr)
# phi = data[phistr]
# omega = data[omegastr]

# print(phi)
# print(omega)

phi = events[phistr].array()
omega = events[omegastr].array()
labelphi = events[labelphistr].array()
labelomega = events[labelomegastr].array()
index = events[indexstr].array()

nevents = len(phi)

data = np.empty(shape=(0, 4))

print(phistr, '\n', phi)
print(omegastr, '\n', omega)
print(indexstr, '\n', index)

for i in range(nevents):
    if index[i] == -1:
        continue
    # print(phi[i], omega[i, index[i]])
    data_i = np.array([phi[i], omega[i, index[i]], labelphi[i, 0], labelomega[i, 0]])
    data = np.concatenate((data, [data_i]), axis=0)

print(data)