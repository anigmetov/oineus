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

x1 = x + np.random.normal(loc=0, scale=noise_std_dev, size=num_points)
y1 = y + np.random.normal(loc=0, scale=noise_std_dev, size=num_points)

pts_1 = torch.Tensor(np.vstack((x1, y1)).T)
pts_1.requires_grad_(True)

x2 = x + np.random.normal(loc=0, scale=noise_std_dev, size=num_points)
y2 = y + np.random.normal(loc=0, scale=noise_std_dev, size=num_points)
pts_2 = torch.Tensor(np.vstack((x2, y2)).T)
pts_2.requires_grad_(True)


# start with topological part

fil_1 = oin.diff.vietoris_rips_pts(pts_1, max_dim=2, max_radius=20.0, n_threads=1)
fil_2 = oin.diff.vietoris_rips_pts(pts_2, max_dim=2, max_radius=20.0, n_threads=1)

fil = oin.diff.min_filtration(fil_1, fil_2)

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

lr = 0.05

opt_1 = torch.optim.SGD([pts_1], lr=lr)
opt_2 = torch.optim.SGD([pts_2], lr=lr)


pts_1_old = pts_1.clone().detach()
pts_2_old = pts_2.clone().detach()

opt_1.step()
opt_2.step()

# just to make sure that we moved both point clouds
print(torch.linalg.norm(pts_1 - pts_1_old))
print(torch.linalg.norm(pts_2 - pts_2_old))

