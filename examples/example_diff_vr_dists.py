#!/usr/bin/env python3

import numpy as np
import torch

import oineus as oin

# sample points from the unit circle
np.random.seed(1)

num_points = 40
noise_std_dev = 0.1

angles = np.random.uniform(low=0, high=2*np.pi, size=num_points)
x = np.cos(angles)
y = np.sin(angles)

x += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)
y += np.random.normal(loc=0, scale=noise_std_dev, size=num_points)

pts = torch.Tensor(np.vstack((x, y)).T)
pts.requires_grad_(True)

# computer pairwise distances differentiably

pts1 = pts.unsqueeze(1)
pts2 = pts.unsqueeze(0)

epsilon = 1e-8

sq_dists = torch.sum((pts1 - pts2) ** 2, dim=2)
dists = torch.sqrt(sq_dists + epsilon)

# start with topological part

fil = oin.diff.vietoris_rips_pwdists(dists, max_dim=2, max_radius=2.0, n_threads=1)
top_opt = oin.diff.TopologyOptimizer(fil)

print(f"{len(fil) = }")

dim = 1
n = 2

dgm = top_opt.compute_diagram(include_inf_points=False)
eps = top_opt.get_nth_persistence(dim, n)
indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)
critical_sets = top_opt.singletons(indices, values)
crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)
crit_indices = np.array(crit_indices, dtype=np.int32)
crit_values = torch.Tensor(crit_values)

help(top_opt.simplify)

print(f"{indices=}, {values=}, {crit_indices=}, {crit_values=}")

top_loss = torch.mean((fil.values[crit_indices] - crit_values) ** 2)

print(f"{top_loss=}")

# let Torch figure the gradient on the coordinates
top_loss.backward()

print(f"{pts.grad=}")
