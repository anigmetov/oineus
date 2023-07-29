#!python3

from icecream import ic
import numpy as np
import oineus as oin

n_pts = 5

pts = np.random.uniform(size=n_pts * 2).reshape(n_pts, 2).astype(np.float64)

fil_1 = oin.get_vr_filtration(pts, 1, 2.0, 1)
fil_2 = oin.get_vr_filtration(pts, 1, 2.0, 1)


simplices = fil_1.simplices()

fil_min_simplices = []

for sigma in simplices:
    min_sigma = oin.Simplex_double(sigma.vertices, min(sigma.value, fil_2.simplex_value_by_vertices(sigma.vertices)))
    fil_min_simplices.append(min_sigma)

fil_min = oin.Filtration_double(fil_min_simplices)

cone_v = fil_min.n_vertices()

# must make copy here! otherwise iteration in for loop below will never end,
# since we keep adding to the same list over which we iterate
coned_simplices = simplices[:]
simplex_id = fil_min.size()

for sigma in simplices:
    coned_sigma = sigma.join(new_vertex=cone_v, value=sigma.value, new_id=simplex_id)
    simplex_id += 1
    coned_simplices.append(coned_sigma)


fil_coned = oin.Filtration_double(coned_simplices, sort_only_by_dimension=True, set_ids=False)

print(fil_coned)
