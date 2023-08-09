#!/usr/bin/env python3

import numpy as np
from icecream import ic

import oineus as oin
import torch


def evaluate_func(x, y, z):
    return np.sin(x + 1.8 * y + 3.1 * z ** 2)

def evaluate_func_template(x, y, z):
    return np.sin(x + 2 * y + 3 * z ** 2)


# Create a 16x16x16 grid
x = np.linspace(-np.pi, np.pi, 16)
y = np.linspace(-np.pi, np.pi, 16)
z = np.linspace(-np.pi, np.pi, 16)

# Generate a 3D meshgrid
X, Y, Z = np.meshgrid(x, y, z)

# Evaluate the function on the grid
f = evaluate_func(X, Y, Z)

# linearize 3D array to 1-D torch tensor
torch_f = torch.Tensor(f.reshape(-1))
torch_f.requires_grad_(True)

f_template = evaluate_func_template(X, Y, Z)

negate = False
wrap =False


fil, max_value_vertices = oin.get_freudenthal_filtration_and_critical_vertices(f, negate=negate, wrap=wrap, max_dim=2, n_threads=8)

top_opt = oin.TopologyOptimizer(fil)

dim = 1

rp = oin.ReductionParams()
rp.compute_u = False
rp.compute_v = False
rp.clearing_opt = True

template_dgms = oin.compute_diagrams_ls(f_template, negate=negate, wrap=wrap, max_dim=dim, include_inf_points=False, params=rp)
template_dgm = template_dgms.in_dimension(dim, as_numpy=False)
ic(template_dgm)


wass_q = 1.0

# if we only want to know where to move the points
indices, values = top_opt.match(template_dgm, dim, wass_q, False)
# if we want to get Wasserstein distance as well
(indices, values), distance = top_opt.match(template_dgm, dim, wass_q, True)
critical_sets = top_opt.singletons(indices, values)
crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)

crit_values = torch.Tensor(crit_values)

crit_indices = np.array(crit_indices, dtype=np.int32)
crit_vertices = max_value_vertices[crit_indices]
crit_vertices = torch.LongTensor(crit_vertices)

ic(crit_indices.shape, crit_vertices.shape, torch_f.shape)

opt = torch.optim.SGD([torch_f], lr=0.2)

opt.zero_grad()
top_loss = torch.sum((torch_f[crit_vertices] - crit_values) ** 2)
top_loss.backward()
opt.step()
