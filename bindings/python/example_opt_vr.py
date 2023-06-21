#!/usr/bin/env python3

import numpy as np
import torch
from icecream import ic

import oineus as oin

# sample points from the unit circle
np.random.seed(1)

num_points = 50
noise_std_dev = 0.1

angles = np.random.uniform(low=0, high=2*np.pi, size=num_points)
x = np.cos(angles)
y = np.sin(angles)

x += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)
y += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)

pts = np.vstack((x, y)).T

# start with topological part

fil, longest_edges = oin.get_vr_filtration_and_critical_edges(pts, max_dim=2, max_radius=2.0, n_threads=1)
ic(longest_edges)

top_opt = oin.TopologyOptimizer(fil)

dim = 1
n = 2

dgm = top_opt.compute_diagram(include_inf_points=False)
eps = top_opt.get_nth_persistence(dim, n)
ic(eps)
indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)
critical_sets = top_opt.singletons(indices, values)
crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)

crit_indices = np.array(crit_indices, dtype=np.int32)
ic(crit_indices)
crit_edges = longest_edges[crit_indices, :]
crit_edges_x, crit_edges_y = crit_edges[:, 0], crit_edges[:, 1]
ic(crit_edges_x, crit_edges_y)

# torch part
# convert everything we need to torch.Tensor
pts = torch.Tensor(pts)
pts.requires_grad_(True)

crit_values = torch.Tensor(crit_values)
# verify the shapes: here we compute the lengths of critical edges
ic(torch.sum((pts[crit_edges_x, :] - pts[crit_edges_y, :])**2, axis=1).shape, crit_values.shape)
top_loss = torch.mean(torch.sum((pts[crit_edges_x, :] - pts[crit_edges_y, :])**2, axis=1) ** 0.5 - crit_values)

# let Torch figure the gradient on the coordinates
top_loss.backward()
