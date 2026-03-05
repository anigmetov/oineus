import oineus as oin
import numpy as np

x = np.zeros((30, 30, 30))

cube_fil = oin.cube_filtration(x, negate=True)

cubes = cube_fil.cubes()
assert len(cubes) == cube_fil.size()
assert cube_fil.max_dim() == 3

dcmp_coh = oin.Decomposition(cube_fil, dualize=True)
rp = oin.ReductionParams()
rp.compute_u = rp.compute_v = True
dcmp_coh.reduce(rp)
include_inf_points = True

dgm = dcmp_coh.diagram(cube_fil, include_inf_points)
# get only points of zero persistence
zero_dgm = dcmp_coh.zero_pers_diagram(cube_fil)

# smoke-check diagram access in all homological dimensions
for dim in range(cube_fil.max_dim()):
    _ = dgm.in_dimension(dim)
    _ = zero_dgm.in_dimension(dim)
