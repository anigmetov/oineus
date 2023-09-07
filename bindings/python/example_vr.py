#!python3

import numpy as np
import oineus as oin

# random points in space
np.random.seed(1)
n_points = 20
dim = 3
points = np.random.uniform(size=(n_points, dim))

# triangulate domain via Freudenthal and create lower star filtration
# negate: set to True to get upper-star filtration
# wrap: set to True to work on torus (periodic boundary conditions)
fil = oin.get_vr_filtration(points=points, max_dim=3, max_radius=2, n_threads=1)

cells = fil.cells()

# Vertices in cells are ids, not sorted_ids
print(f"Filtration with {len(cells)} cells created,\nvertex 0: {cells[0]},\nlast simplex: {cells[-1]}")

# no cohomology
dualize = False
# create VRU decomposition object, does not perform reduction yet
dcmp = oin.Decomposition(fil, dualize)

# reduction parameters
# relevant members:
# rp.clearing_opt --- whether you want to use clearing, True by default
# rp.compute_v: True by default
# rp.n_threads: number of threads to use, default is 1
# rp. compute_u: False by default (cannot do it in multi-threaded mode, so switch off just to be on the safe side)
rp = oin.ReductionParams()

# perform reduction
dcmp.reduce(rp)

# now we can acess V, R and U
# indices are sorted_ids of simplices == indices in fil.cells()
V = dcmp.v_data
print(f"Example of a V column: {V[-1]}, this chain contains cells:")
for sigma_idx in V[-1]:
    print(cells[sigma_idx])

# get diagram, including points at infinity
include_inf_points=True
dgm = dcmp.diagram(fil, include_inf_points)
# get only points of zero persistence
zero_dgm = dcmp.zero_pers_diagram(fil)

# diagram in dimension d is numpy array dgm[d], shape = (number of diagram points, 2)

for dim in range(dim):
    print(f"Diagram in dimension {dim}:")
    print(dgm[dim])

