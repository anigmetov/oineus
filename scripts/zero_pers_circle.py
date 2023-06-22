#!/usr/bin/env python3
# vim: foldmethod=marker foldlevel=0
#m> # Example of Oineus Notebook

#m> Import necessary modules
import numpy as np
from matplotlib import pyplot as plt

import oineus as oin

def get_zdgm(pts):
    fil, longest_edges = oin.get_vr_filtration_and_critical_edges(pts, max_dim=2, max_radius=4.0, n_threads=8)
    params = oin.ReductionParams()
    params.clearing_opt = False
    params.compute_u = False
    params.compute_v = False
    params.n_threads = 8
    decmp = oin.Decomposition(fil, False)
    decmp.reduce(params)
    zdgms = decmp.zero_persistence_diagram(fil)
    zdgm = zdgms[1][:, 0]
    return zdgm


def plot(pts, zdgm, outliers=None, pts_title="", dgm_title=""):
    fig, (ax_pts, ax_zdgm) = plt.subplots(1, 2)
    ax_pts.scatter(pts[:, 0], pts[:, 1], color="blue")
    ax_pts.set_title(pts_title)
    ax_zdgm.hist(zdgm, bins=30, edgecolor="black")
    ax_zdgm.set_xlabel("birth-death value")
    ax_zdgm.set_ylabel("Frequency")
    ax_zdgm.set_title(dgm_title)
    if outliers:
        ax_pts.scatter(pts[outliers, 0], pts[outliers, 1], color="red")
    plt.tight_layout()
    plt.show()


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

zdgm = get_zdgm(pts)
plot(pts, zdgm, pts_title="Point cloud", dgm_title="No outliers")

#m> # Add outlier in the centre

x_ctr = np.append(x, 0.0)
y_ctr = np.append(y, 0.0)

pts_ctr = np.vstack((x_ctr, y_ctr)).T
pts_ctr = np.array(pts_ctr, dtype=np.float32)
zdgm = get_zdgm(pts_ctr)
plot(pts_ctr, zdgm, pts_title="Point cloud", dgm_title="Outlier in the centre", outliers=np.array([x_ctr.shape[0]-1]))

#m> # Add outlier inside, not in the centre

x_noctr = np.append(x, 0.0)
y_noctr = np.append(y, 0.5)

pts_noctr = np.vstack((x_noctr, y_noctr)).T
pts_noctr = np.array(pts_noctr, dtype=np.float32)

zdgm = get_zdgm(pts_noctr)
plot(pts_noctr, zdgm, pts_title="Point cloud", dgm_title="Outlier inside, not in the centre", outliers=np.array([x_ctr.shape[0]-1]))



#m> # Add outlier outside the circle
x_off_ctr = np.append(x, 2.0)
y_off_ctr = np.append(y, 0.0)

pts_off_ctr = np.vstack((x_off_ctr, y_off_ctr)).T
pts_off_ctr = np.array(pts_off_ctr, dtype=np.float32)

zdgm = get_zdgm(pts_off_ctr)
plot(pts_off_ctr, zdgm, pts_title="Point cloud", dgm_title="Outlier outside", outliers=np.array([x_off_ctr.shape[0]-1]))

