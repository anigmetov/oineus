#!/usr/bin/env python3
# vim: foldmethod=marker foldlevel=0
#m> # Example of Oineus Notebook

#m> Import necessary modules
import numpy as np
from matplotlib import pyplot as plt

import oineus as oin

np.random.seed(1)

num_points = 60
noise_std_dev = 0.02

max_dim = 2

angles = np.random.uniform(low=0, high=2*np.pi, size=num_points)
x = np.cos(angles)
y = np.sin(angles)

x += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)
y += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)

pts = np.vstack((x, y)).T
pts = np.array(pts, dtype=np.float32)
points = pts

#m> Get filtration
fil, longest_edges = oin.get_vr_filtration_and_critical_edges(pts, max_dim=2, max_radius=4.0, n_threads=8)
print("computed fil")
#o> computed fil

#m> Perform reduction
params = oin.ReductionParams()
params.clearing_opt = False
params.compute_u = False
params.compute_v = False
params.n_threads = 8

decmp = oin.Decomposition(fil, False)
decmp.reduce(params)

#m> Get 0-pers diagram
zdgms = decmp.zero_persistence_diagram(fil)

#m> Get 0-pers diagram
zdgm = zdgms[1][:, 0]
plt.hist(zdgm, bins=30, edgecolor="black")
plt.xlabel("birth-death value")
plt.ylabel("Frequency")
plt.title("No outliers")
plt.show()

#m> # Add outlier in the centre

x_ctr = np.append(x, 0.0)
y_ctr = np.append(y, 0.0)

pts_ctr = np.vstack((x_ctr, y_ctr)).T
pts_ctr = np.array(pts_ctr, dtype=np.float32)

#m> Get filtration
fil, longest_edges = oin.get_vr_filtration_and_critical_edges(pts_ctr, max_dim=2, max_radius=4.0, n_threads=8)

#m> Perform reduction
params = oin.ReductionParams()
params.clearing_opt = False
params.compute_u = False
params.compute_v = False
params.n_threads = 8

decmp = oin.Decomposition(fil, False)
decmp.reduce(params)

#m> Get 0-pers diagram
zdgms = decmp.zero_persistence_diagram(fil)

#m> Get 0-pers diagram
zdgm = zdgms[1][:, 0]
plt.hist(zdgm, bins=30, edgecolor="black")
plt.xlabel("birth-death value")
plt.ylabel("Frequency")
plt.title("Outlier in the centre")
plt.show()


#m> # Add outlier outside the circle
x_off_ctr = np.append(x, 2.0)
y_off_ctr = np.append(y, 0.0)

pts_off_ctr = np.vstack((x_off_ctr, y_off_ctr)).T
pts_off_ctr = np.array(pts_off_ctr, dtype=np.float32)

#m> Get filtration
fil, longest_edges = oin.get_vr_filtration_and_critical_edges(pts_off_ctr, max_dim=2, max_radius=4.0, n_threads=8)

#m> Perform reduction
params = oin.ReductionParams()
params.clearing_opt = False
params.compute_u = False
params.compute_v = False
params.n_threads = 8

decmp = oin.Decomposition(fil, False)
decmp.reduce(params)

#m> Get 0-pers diagram
zdgms = decmp.zero_persistence_diagram(fil)

#m> Get 0-pers diagram
zdgm = zdgms[1][:, 0]
plt.hist(zdgm, bins=30, edgecolor="black")
plt.xlabel("birth-death value")
plt.ylabel("Frequency")
plt.title("Outlier outside")
plt.show()


