#!python3

import numpy as np
import oineus as oin

n_pts = 10

pts = np.random.uniform(size=n_pts * 2).reshape(n_pts, 2).astype(np.float64)

fil_1 = oin.vr_filtration(pts)
fil_2 = oin.vr_filtration(pts)

simplices = fil_1.simplices()

fil_min_simplices = []

for sigma in simplices:
    min_sigma = oin.Simplex(sigma.vertices, min(sigma.value, fil_2.value_by_uid(sigma.uid)))
    fil_min_simplices.append(min_sigma)

fil_min = oin.Filtration(fil_min_simplices)

cone_v = fil_min.n_vertices()

# must make copy here! otherwise iteration in for loop below will never end,
# since we keep adding to the same list over which we iterate
coned_simplices = simplices[:]

# append cone vertex, force it to be the first vertex in the list
coned_simplices.append(oin.Simplex([cone_v], -0.000000001))

for sigma in simplices:
    coned_sigma = sigma.join(new_vertex=cone_v, value=sigma.value)
    coned_simplices.append(coned_sigma)

fil_coned = oin.Filtration(coned_simplices)

print(fil_coned)
