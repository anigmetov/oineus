#!/usr/bin/env python3
"""Optimize a 2D cubical filtration to reduce the most persistent H0 feature."""

import numpy as np
import torch

import oineus as oin
import oineus.diff


torch.manual_seed(0)
np.random.seed(0)

# Small 2D image with a few local minima / saddles -> non-trivial H0 diagram.
data = torch.tensor(
    np.random.uniform(-1.0, 1.0, size=(6, 6)),
    dtype=torch.float64,
    requires_grad=True,
)

fil = oin.diff.cube_filtration(data, max_dim=2, n_threads=1)
top_opt = oin.diff.TopologyOptimizer(fil)

dim = 0
n = 1  # target the 2nd-most-persistent class

initial_loss = None
for step in range(10):
    fil = oin.diff.cube_filtration(data, max_dim=2, n_threads=1)
    top_opt = oin.diff.TopologyOptimizer(fil)

    eps = top_opt.get_nth_persistence(dim, n)
    indices, values = top_opt.simplify(eps, oin.DenoiseStrategy.BirthBirth, dim)
    critical_sets = top_opt.singletons(indices, values)
    crit_indices, crit_values = top_opt.combine_loss(critical_sets, oin.ConflictStrategy.Max)
    crit_indices = np.asarray(crit_indices, dtype=np.int64)
    crit_values = torch.tensor(crit_values, dtype=torch.float64)

    loss = torch.mean((fil.values[crit_indices] - crit_values) ** 2)
    if initial_loss is None:
        initial_loss = float(loss)

    data.grad = None
    loss.backward()
    with torch.no_grad():
        data -= 0.1 * data.grad

print(f"initial loss: {initial_loss:.6g}")
print(f"final loss:   {float(loss):.6g}")

assert float(loss) <= initial_loss + 1e-9, "loss should be non-increasing on this toy example"
