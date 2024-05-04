#!/usr/bin/env python3

import numpy as np
import torch

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

pts = torch.Tensor(np.vstack((x, y)).T)
pts.requires_grad_(True)

# start with topological part

fil = oin.diff.vietoris_rips_pts(pts, max_dim=2, max_radius=2.0, n_threads=1)
top_opt = oin.TopologyOptimizer(fil.under_fil)

dim = 1
n = 2

dgm = top_opt.compute_diagram(include_inf_points=False)
eps = top_opt.get_nth_persistence(dim, n)
indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)
critical_sets = top_opt.singletons(indices, values)
crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)
crit_indices = np.array(crit_indices, dtype=np.int32)
crit_values = torch.Tensor(crit_values)

top_loss = torch.mean(fil.values[crit_indices] - crit_values)

# let Torch figure the gradient on the coordinates
top_loss.backward()
