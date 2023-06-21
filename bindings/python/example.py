#!python3

import numpy as np
import oineus as oin

# scalar function on 8x8x8 grid
np.random.seed(1)
f = np.random.uniform(size=(8, 8, 8))

# triangulate domain via Freudenthal and create lower star filtration
# negate: set to True to get upper-star filtration
# wrap: set to True to work on torus (periodic boundary conditions)
fil = oin.get_freudenthal_filtration(data=f, negate=False, wrap=False, max_dim=3, n_threads=1)

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

