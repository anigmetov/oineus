#!python3

from icecream import ic
import numpy as np
import oineus as oin
import torch

n_pts = 50

np.random.seed(1)

pts_1 = np.random.uniform(low=0.0, high=5.0, size=n_pts * 2).reshape(n_pts, 2).astype(np.float64)
pts_2 = pts_1 + np.random.uniform(size=n_pts *2, low=-0.01, high=0.01).reshape(pts_1.shape)

# input: pts
# output: VR filtration and torch Tensor with the lengths
# of critical edges in filtration order, that is:
# fil.simplices()[sorted_idx].value == lengths[sorted_idx]
def get_vr_filtration_and_vals(pts, max_dim, max_radius):
    fil, edges = oin.get_vr_filtration_and_critical_edges(pts, max_dim=max_dim, max_radius=max_radius, n_threads=1)
    fil.reset_ids_to_sorted_ids()
    pts = torch.Tensor(pts)
    edge_start, edge_end = edges[:, 0], edges[:, 1]
    lengths = torch.sum((pts[edge_start, :] - pts[edge_end, :])**2, axis=1) ** 0.5

    return fil, lengths

max_dim = 2
max_radius = 1000.0

fil_1, lengths_1 = get_vr_filtration_and_vals(pts_1, max_dim, max_radius)
fil_2, lengths_2 = get_vr_filtration_and_vals(pts_2, max_dim, max_radius)

# create fil_min

simplices_1 = fil_1.simplices()
simplices_2 = fil_2.simplices()

fil_min_simplices = []

for sigma in simplices_1:
    min_sigma = oin.Simplex_double(sigma.sorted_id, sigma.vertices,
                                   min(sigma.value, fil_2.simplex_value_by_vertices(sigma.vertices)))
    fil_min_simplices.append(min_sigma)

fil_min = oin.Filtration_double(fil_min_simplices, set_ids=False)

# now simplices became reshuffled
# we must permute lengths_1 and lengths_2 accordingly
# id of simplex in fil_min is it's sorted_id in the original filtration

min_sorted_id_to_sorted_id_1 = fil_min.get_inv_sorting_permutation()

lengths_1_in_min_order = lengths_1[torch.LongTensor(min_sorted_id_to_sorted_id_1)]

min_simplices = fil_min.simplices()


min_sorted_id_to_sorted_id_2 = [ fil_2.get_sorted_id_by_vertices(min_sigma.vertices) for min_sigma in min_simplices ]

lengths_2_in_min_order = lengths_2[torch.LongTensor(min_sorted_id_to_sorted_id_2)]

for i, sigma in enumerate(min_simplices):
    s1 = simplices_1[fil_1.get_sorted_id_by_vertices(sigma.vertices)]
    s2 = simplices_2[fil_2.get_sorted_id_by_vertices(sigma.vertices)]
    assert np.abs(s1.value - lengths_1_in_min_order[i]) < 0.0001
    assert np.abs(s2.value - lengths_2_in_min_order[i]) < 0.0001


min_critical_values = torch.min(torch.stack((lengths_1_in_min_order, lengths_2_in_min_order)), axis=0)[0]

# verify that we got the critical values correctly
for sigma in min_simplices:
    assert torch.abs(sigma.value - min_critical_values[sigma.sorted_id]) < 0.00001


min_and_coned_simplices = min_simplices[:]
cone_v = fil_min.n_vertices()

# append cone vertex
# TODO: how to ensure that cone_vertex appears first?
# for now we can ignore it
min_and_coned_simplices.append(oin.Simplex_double(cone_v, [cone_v], 0.0))

simplex_id = len(min_and_coned_simplices)

# take cones over simplices from fil_1
for sigma in simplices_1:
    coned_sigma = sigma.join(new_vertex=cone_v, value=sigma.value, new_id=simplex_id)
    simplex_id += 1
    min_and_coned_simplices.append(coned_sigma)

fil_min_and_coned = oin.Filtration_double(min_and_coned_simplices, set_ids=False)

# in filtration order
min_and_coned_simplices = fil_min_and_coned.simplices()

for sigma in min_and_coned_simplices:
    if cone_v in sigma.vertices:
        if sigma.dim() == 0:
            continue
        # subtract additional 1 for cone vertex
        orig_sorted_sigma_id = sigma.id - len(fil_1) - 1
        assert(np.abs(lengths_1[orig_sorted_sigma_id] - sigma.value) < 0.00001)
    else:
        orig_sigma_id = sigma.id
        orig_sorted_sigma_id = fil_min.get_sorted_id_by_id(orig_sigma_id)
        assert(np.abs(min_critical_values[orig_sorted_sigma_id] - sigma.value) < 0.00001)

print(fil_min_and_coned)
