import oineus as oin
import numpy as np

x = np.zeros((30, 30, 30))

cube_fil = oin.cube_filtration(x, negate=True)

for cube in cube_fil.cubes:
    print(cube)

dcmp_coh = oin.Decomposition(cube_fil, dualize=True)
rp = oin.ReductionParams()
rp.compute_u = rp.compute_v = True
dcmp_coh.reduce(rp)
include_inf_points = True

dgm = dcmp_coh.diagram(cube_fil, include_inf_points)
# get only points of zero persistence
zero_dgm = dcmp_coh.zero_pers_diagram(cube_fil)
