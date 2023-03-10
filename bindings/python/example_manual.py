#!python3

import numpy as np
import oineus as oin

# here we create a simple example (triangle) with vertices
# v0 (critical value 0.2)
# v1 (critical value 0.1)
# v3 (critical value 0.3)
# edges: e1 = [v0, v1]
#

# The filtration we create is not a lower-star filtration,
# but we use lower-star simplices, it doesn't really matter.
# The last parameter to ctor, critical_value_location (int)
# is not used in this example, so we use 0 everywhere.

v0 = oin.LSSimplex_double([0], 0.2, 0)
v1 = oin.LSSimplex_double([1], 0.1, 0)
v2 = oin.LSSimplex_double([2], 0.3, 0)

# indices of vertices should be sorted
e1 = oin.LSSimplex_double([0, 1], 0.9, 0)
e2 = oin.LSSimplex_double([0, 2], 0.5, 0)
e3 = oin.LSSimplex_double([1, 2], 0.8, 0)

t = oin.LSSimplex_double([0, 1, 2], 1.0, 0)

simplices = [v0, v1, v2, e1, e2, e3, t]

negate = False
n_threads = 1

# constructor will sort simplices and assign sorted_ids
fil = oin.LSFiltration_double(simplices, negate, n_threads)

fil_simplices = fil.simplices()

for sigma in fil.simplices():
    print(sigma)

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

# get diagram, including points at infinity
include_inf_points=True
dgm = dcmp.diagram(fil, include_inf_points)

# diagram in dimension d is numpy array dgm[d], shape = (number of diagram points, 2)

for dim in [0, 1]:
    print(f"Diagram in dimension {dim}:")
    print(dgm[dim])

