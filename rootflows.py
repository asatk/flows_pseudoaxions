from matplotlib import pyplot as plt
import numpy as np
import re
import uproot as up
import os

# up.default_library(library="np")

cwd = os.getcwd()
if 'root' not in os.listdir(cwd):
    print('\'root\' directory not in the cwd.\nExiting...')
    exit()

rootd = cwd + '/root/attoaod_Box30by30-10k-Mar2023/'
files = os.listdir(rootd)

p = re.compile("\w+(\d)p(\d+)_(\d+)\.root$")
matches = [p.match(fname) for fname in files]

# for fname, m in zip(files, matches):
#     print(fname, m.group(1), m.group(2), m.group(3))

test = files[0]
data = up.open(rootd + test + ":Events;1")
print(data.keys())
phi = data["TwoProng_massPi0"].array(library="np")
omega = data["TwoProng_massEta"].array(library="np")

print(phi)
print(omega)
