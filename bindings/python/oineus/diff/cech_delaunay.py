import numpy as np
import torch
import diode

from .. import _oineus
from .diff_filtration import DiffFiltration

def triangle_meb(p0, p1, p2, eps=1e-12):
    """
    Compute minimum enclosing ball center and radius squared for triangles.

    Args:
        p0, p1, p2: Tensor of shape (n, d) for n triangles in d dimensions
        eps: Small value for numerical stability

    Returns:
        centers: Tensor of shape (n, d) - MEB centers
        radii_sq: Tensor of shape (n,) - MEB radii squared
    """
    a = p1 - p0
    b = p2 - p0
    c = p2 - p1

    a_sq = torch.sum(a ** 2, dim=1)
    b_sq = torch.sum(b ** 2, dim=1)
    c_sq = torch.sum(c ** 2, dim=1)

    d = p0.shape[1]
    if d == 2:
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        area_2_sq = cross ** 2
    else:
        cross = torch.cross(a, b, dim=1)
        area_2_sq = torch.sum(cross ** 2, dim=1)

    circum_radii_sq = (a_sq * b_sq * c_sq + eps) / (4 * area_2_sq + eps)

    if d == 3:
        cross_ab = torch.cross(a, b, dim=1)
        cross_ab_sq = torch.sum(cross_ab ** 2, dim=1, keepdim=True)
        a_dot_a = a_sq.unsqueeze(1)
        b_dot_b = b_sq.unsqueeze(1)
        b_cross_axb = torch.cross(b, cross_ab, dim=1)
        axb_cross_a = torch.cross(cross_ab, a, dim=1)
        circum_centers = p0 + (a_dot_a * b_cross_axb + b_dot_b * axb_cross_a) / (2 * cross_ab_sq + eps)
    else:
        a_dot_a = a_sq.unsqueeze(1)
        b_dot_b = b_sq.unsqueeze(1)
        D = 2 * (a[:, 0:1] * b[:, 1:2] - a[:, 1:2] * b[:, 0:1])
        ux = (b[:, 1:2] * a_dot_a - a[:, 1:2] * b_dot_b) / (D + eps)
        uy = (a[:, 0:1] * b_dot_b - b[:, 0:1] * a_dot_a) / (D + eps)
        circum_centers = p0 + torch.cat([ux, uy], dim=1)

    abc_sq = torch.stack((a_sq, b_sq, c_sq), dim=0)
    s_abc_sq, sort_idx = torch.sort(abc_sq, dim=0)
    obtuse_mask = s_abc_sq[2, :] > s_abc_sq[0, :] + s_abc_sq[1, :]

    longest_edge_idx = sort_idx[2, :]

    midpoint_a = (p0 + p1) / 2
    midpoint_b = (p0 + p2) / 2
    midpoint_c = (p1 + p2) / 2

    centers = circum_centers.clone()
    radii_sq = circum_radii_sq.clone()

    if obtuse_mask.any():
        obtuse_longest = longest_edge_idx[obtuse_mask]
        mask_a = obtuse_longest == 0
        mask_b = obtuse_longest == 1
        mask_c = obtuse_longest == 2
        obtuse_indices = torch.where(obtuse_mask)[0]

        if mask_a.any():
            centers[obtuse_indices[mask_a]] = midpoint_a[obtuse_mask][mask_a]
        if mask_b.any():
            centers[obtuse_indices[mask_b]] = midpoint_b[obtuse_mask][mask_b]
        if mask_c.any():
            centers[obtuse_indices[mask_c]] = midpoint_c[obtuse_mask][mask_c]

        radii_sq[obtuse_mask] = s_abc_sq[2, obtuse_mask] / 4

    return centers, radii_sq


def tetrahedron_meb(p0, p1, p2, p3, eps=1e-12):
    """
    Compute minimum enclosing ball center and radius squared for tetrahedra.

    The MEB of a tetrahedron is one of:
    1. The circumsphere (all 4 vertices on boundary)
    2. A face's MEB (if opposite vertex is inside that MEB)
    3. An edge's MEB (if other two vertices are inside that MEB)

    Args:
        p0, p1, p2, p3: Tensor of shape (n, 3) for n tetrahedra
        eps: Small value for numerical stability

    Returns:
        centers: Tensor of shape (n, 3) - MEB centers
        radii_sq: Tensor of shape (n,) - MEB radii squared
    """
    n = p0.shape[0]
    device = p0.device
    dtype = p0.dtype

    # Compute circumsphere of tetrahedron
    a = p1 - p0
    b = p2 - p0
    c = p3 - p0

    a_sq = torch.sum(a ** 2, dim=1, keepdim=True)
    b_sq = torch.sum(b ** 2, dim=1, keepdim=True)
    c_sq = torch.sum(c ** 2, dim=1, keepdim=True)

    cross_bc = torch.cross(b, c, dim=1)
    cross_ca = torch.cross(c, a, dim=1)
    cross_ab = torch.cross(a, b, dim=1)

    volume_6 = torch.sum(a * cross_bc, dim=1)

    numerator_vec = a_sq * cross_bc + b_sq * cross_ca + c_sq * cross_ab
    circum_disp = numerator_vec / (2 * volume_6.unsqueeze(1) + eps)
    circum_center = p0 + circum_disp
    circum_radii_sq = torch.sum(circum_disp ** 2, dim=1)

    # Compute MEB for each of the 4 faces
    face_centers_0, face_radii_sq_0 = triangle_meb(p1, p2, p3, eps)  # opposite to p0
    face_centers_1, face_radii_sq_1 = triangle_meb(p0, p2, p3, eps)  # opposite to p1
    face_centers_2, face_radii_sq_2 = triangle_meb(p0, p1, p3, eps)  # opposite to p2
    face_centers_3, face_radii_sq_3 = triangle_meb(p0, p1, p2, eps)  # opposite to p3

    # Check if opposite vertex is contained in face's MEB
    dist_sq_0 = torch.sum((p0 - face_centers_0) ** 2, dim=1)
    dist_sq_1 = torch.sum((p1 - face_centers_1) ** 2, dim=1)
    dist_sq_2 = torch.sum((p2 - face_centers_2) ** 2, dim=1)
    dist_sq_3 = torch.sum((p3 - face_centers_3) ** 2, dim=1)

    contains_0 = dist_sq_0 <= face_radii_sq_0 + eps
    contains_1 = dist_sq_1 <= face_radii_sq_1 + eps
    contains_2 = dist_sq_2 <= face_radii_sq_2 + eps
    contains_3 = dist_sq_3 <= face_radii_sq_3 + eps

    # Build candidate radii and centers
    inf_val = torch.tensor(float('inf'), dtype=dtype, device=device)

    # Stack all candidates: circumsphere, 4 faces, 6 edges = 11 candidates
    all_radii_sq = torch.stack([
        circum_radii_sq,
        torch.where(contains_0, face_radii_sq_0, inf_val),
        torch.where(contains_1, face_radii_sq_1, inf_val),
        torch.where(contains_2, face_radii_sq_2, inf_val),
        torch.where(contains_3, face_radii_sq_3, inf_val),
    ], dim=0)

    all_centers = torch.stack([
        circum_center,
        face_centers_0,
        face_centers_1,
        face_centers_2,
        face_centers_3,
    ], dim=0)

    # Find minimum radius for each tetrahedron
    min_radii_sq, min_idx = torch.min(all_radii_sq, dim=0)

    # Gather corresponding centers
    centers = all_centers[min_idx, torch.arange(n)]

    return centers, min_radii_sq


def cech_delaunay_filtration(alpha_fil, points, eps=0.0):
    """
    :param alpha_fil: Alpha filtration from diode or oineus
    :param points: Tensor of point coordinates
    :param eps: Small value for numerical stability
    :return: differentiable Cech-Delaunay filtration
    """
    if type(alpha_fil) is not _oineus.Filtration:
        alpha_fil = _oineus.Filtration([_oineus.Simplex(vs, val) for vs, val in alpha_fil])

    values_in_dim = [torch.zeros(alpha_fil.size_in_dimension(0), requires_grad=True, device=points.device)]

    for dim in range(1, alpha_fil.max_dim() + 1):
        if dim == 1:
            edges = torch.LongTensor(alpha_fil.get_edges().astype(np.uint64))
            sqdists = torch.sum((points[edges[:, 0]] - points[edges[:, 1]]) ** 2, axis=1)
            radii_sq = 0.25 * sqdists
            assert edges.shape[0] == radii_sq.shape[0]

        elif dim == 2:
            triangles = torch.LongTensor(alpha_fil.get_triangles().astype(np.uint64))
            p0 = points[triangles[:, 0]]
            p1 = points[triangles[:, 1]]
            p2 = points[triangles[:, 2]]

            # ignore centers
            _, radii_sq = triangle_meb(p0, p1, p2, eps)

            assert triangles.shape[0] == radii_sq.shape[0]

        elif dim == 3:
            tetra = torch.LongTensor(alpha_fil.get_tetrahedra().astype(np.uint64))
            p0 = points[tetra[:, 0]]
            p1 = points[tetra[:, 1]]
            p2 = points[tetra[:, 2]]
            p3 = points[tetra[:, 3]]

            _, radii_sq = tetrahedron_meb(p0, p1, p2, p3, eps)

            assert tetra.shape[0] == radii_sq.shape[0]
        else:
            raise RuntimeError("Dimension not supported")

        values_in_dim.append(radii_sq)

    # this will sort the simplices in the filtration correctly, cd_vals_np is not monotonic
    cd_vals = torch.cat(values_in_dim)
    cd_vals_list = [ float(x) for x in cd_vals.clone().detach().cpu() ]
    # set non-differentiable internal Oineus values:
    alpha_fil.set_values(cd_vals_list)
    # sort values in each dimension independently, in a differentiable way:
    sorted_vals = torch.cat([ torch.sort(vals)[0] for vals in values_in_dim])
    return DiffFiltration(alpha_fil, sorted_vals)
