#!python3

from icecream import ic
import numpy as np
import oineus as oin
import oineus.diff
import torch

n_pts = 50

np.random.seed(1)

pts_1 = np.random.uniform(low=0.0, high=5.0, size=n_pts * 2).reshape(n_pts, 2).astype(np.float64)
pts_2 = pts_1 + np.random.uniform(size=n_pts *2, low=-0.01, high=0.01).reshape(pts_1.shape)

pts_1 = torch.DoubleTensor(pts_1)
pts_2 = torch.DoubleTensor(pts_2)

pts_1.requires_grad_()
pts_2.requires_grad_()


fil_1 = oin.diff.vr_filtration(pts_1, eps=1e-9)
fil_2 = oin.diff.vr_filtration(pts_1, eps=1e-0)

fil_min = oin.diff.min_filtration(fil_1, fil_2)

# we will append coned simplices later
min_and_coned_simplices = fil_min.cells()

cone_v = fil_min.n_vertices()

# append cone vertex
# TODO: how to ensure that cone_vertex appears first?
# for now we can ignore it
min_and_coned_simplices.append(oin.Simplex(cone_v, [cone_v], -0.0))

# take cones over simplices from fil_1
for sigma in fil_1:
    coned_sigma = sigma.join(new_vertex=cone_v, value=sigma.value)
    min_and_coned_simplices.append(coned_sigma)

fil_min_and_coned = oin.Filtration(min_and_coned_simplices)

# in filtration order
min_and_coned_simplices = fil_min_and_coned.simplices()

lengths_1 = fil_1.values.detach().cpu().numpy()
lengths_2 = fil_1.values.detach().cpu().numpy()
lengths_min = np.min(np.stack((lengths_1, lengths_2)), axis=0)

for sigma in min_and_coned_simplices:
    if sigma.dim() == 0:
        continue
    # get cone base, we only need it's uid, so value does not matter
    base_vertices = [v for v in sigma if v != cone_v]
    sigma_base = oin.Simplex(base_vertices)
    # here is how we can get the index of sigma_base in the original min_filtration:
    # (note that uid is uniquely determined by simplex vertices)
    orig_sorted_sigma_id = fil_min.sorted_id_by_uid(sigma_base.uid)
    if sigma_base.dim() == sigma.dim():
        assert(np.abs(lengths_1[orig_sorted_sigma_id] - sigma.value) < 0.001)
    else:
        assert(np.abs(lengths_min[orig_sorted_sigma_id] - sigma.value) < 0.001)

