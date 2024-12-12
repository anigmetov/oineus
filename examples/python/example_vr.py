#!python3

import numpy as np
import oineus as oin
import time
import sys

from icecream import ic


# random points in space
np.random.seed(1)
n_points = 50
dim = 3
points = np.random.uniform(size=(n_points, dim))

# create Vietoris--Rips filtration
n_threads = 8
fil = oin.vr_filtration(points, n_threads=n_threads)

# we can specify max_dim and max_diameter manually,
# if we need to.
# max_dim default value is the dimension of the points,
# max_diameter default value is same minimax as in Ripser (after that the complex becomes contractible)

fil = oin.vr_filtration(points, max_dim=1, max_diameter=0.5)

# if we want to get parallel array of critical edges,
# just specify with_critical_edges = True
# function will return a tuple: (filtration, edges)

fil, edges = oin.vr_filtration(points, with_critical_edges=True, n_threads=n_threads)

# edges is NumPy array of shape (N, 2)
# it contains indices of simplices in the filtration order (sorted_ids)

# we can subscript filtration
print(f"Filtration with {len(fil)} cells created,\nvertex 0: {fil[0]}, edge {edges[0]},\nlast simplex: {fil[-1]}, edge {edges[-1]}")

# we can also get a copy of sorted cells:
cells = fil.cells()

# we can access boundary matrix
bm = fil.boundary_matrix()

# create VRU decomposition object, does not perform reduction yet
# we want to use cohomology:
dualize = True
dcmp = oin.Decomposition(fil, dualize)


# reduction parameters
# relevant members:
# rp.clearing_opt --- whether you want to use clearing, True by default
# rp.compute_v: False by default
# rp.n_threads: number of threads to use, default is 1
# rp. compute_u: False by default (cannot do it in multi-threaded mode, so switch off just to be on the safe side)
rp = oin.ReductionParams()
rp.compute_v = True
rp.n_threads = 4

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

