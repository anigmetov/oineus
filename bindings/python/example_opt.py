#!/usr/bin/env python3

import numpy as np
from icecream import ic

import oineus as oin


def evaluate_func(x, y, z):
    return np.sin(x + 2 * y + 3 * z ** 2)


# Create a 16x16x16 grid
x = np.linspace(-np.pi, np.pi, 16)
y = np.linspace(-np.pi, np.pi, 16)
z = np.linspace(-np.pi, np.pi, 16)

# Generate a 3D meshgrid
X, Y, Z = np.meshgrid(x, y, z)

# Evaluate the function on the grid
f = evaluate_func(X, Y, Z)

fil, max_value_vertices = oin.get_freudenthal_filtration_and_critical_vertices(f, negate=False, wrap=False, max_dim=2, n_threads=8)

top_opt = oin.TopologyOptimizer(fil)

dim = 1
n = 4

dgm = top_opt.compute_diagram(include_inf_points=False)
eps = top_opt.get_nth_persistence(dim, 4)
indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)
critical_sets = top_opt.singletons(indices, values)
crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)

crit_indices = np.array(crit_indices, dtype=np.int32)
crit_vertices = max_value_vertices[crit_indices]