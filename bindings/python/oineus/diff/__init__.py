import functools as ft
from typing import Optional, Dict, Any, Union, Tuple
from icecream import ic

import numpy as np
import eagerpy as epy

from .. import _oineus
from .. import vr_filtration as non_diff_vr_filtration

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# to copy docstring and name from the wrapped _oineus (C++) methods


class DiffFiltration:
    def __init__(self, fil, values):
        self.under_fil = fil
        self.values = values

    def __len__(self):
        return len(self.under_fil)

    def __repr__(self):
        return f"DiffFil(under_fil={self.under_fil}, values={self.values})"

    def __iter__(self):
        return iter(self.under_fil)

    #@ft.wraps(_oineus.Filtration.max_dim)
    def max_dim(self):
        return self.under_fil.max_dim()

    #@ft.wraps(_oineus.Filtration.size)
    def size(self):
        return self.under_fil.size()

    #@ft.wraps(_oineus.Filtration.size_in_dimension)
    def size_in_dimension(self, dim):
        return self.under_fil.size(dim)

    #@ft.wraps(_oineus.Filtration.n_vertices)
    def n_vertices(self):
        return self.under_fil.n_vertices()

    #@ft.wraps(_oineus.Filtration.cells)
    def cells(self):
        return self.under_fil.cells()

    #@ft.wraps(_oineus.Filtration.get_id_by_sorted_id)
    def id_by_sorted_id(self, sorted_id):
        return self.under_fil.id_by_sorted_id(sorted_id)

    #@ft.wraps(_oineus.Filtration.get_sorted_id_by_id)
    def sorted_id_by_id(self, id):
        return self.under_fil.sorted_id_by_id(id)

    #@ft.wraps(_oineus.Filtration.get_cell)
    def cell(self, sorted_idx):
        return self.under_fil.cell(sorted_idx)

    #@ft.wraps(_oineus.Filtration.get_simplex)
    def simplex(self, sorted_idx):
        return self.under_fil.simplex(sorted_idx)

    #@ft.wraps(_oineus.Filtration.get_sorting_permutation)
    def sorting_permutation(self):
        return self.under_fil.sorting_permutation()

    #@ft.wraps(_oineus.Filtration.get_inv_sorting_permutation)
    def get_inv_sorting_permutation(self):
        return self.under_fil.inv_sorting_permutation()

    #@ft.wraps(_oineus.Filtration.cell_by_uid)
    def cell_by_uid(self, uid):
        return self.under_fil.cell_by_uid(uid)

    #@ft.wraps(_oineus.Filtration.boundary_matrix)
    def boundary_matrix(self, uid):
        return self.under_fil.boundary_matrix(uid)

    #@ft.wraps(_oineus.Filtration.simplex_value_by_sorted_id)
    def simplex_value_by_sorted_id(self, sorted_id):
        return self.under_fil.simplex_value_by_sorted_id(sorted_id)

    #@ft.wraps(_oineus.Filtration.simplex_value_by_vertices)
    def simplex_value_by_uid(self, uid):
        return self.under_fil.simplex_value_by_uid(uid)

    def cell_value_by_uid(self, uid):
        return self.under_fil.cell_value_by_uid(uid)

    #@ft.wraps(_oineus.Filtration.get_sorted_id_by_vertices)
    def sorted_id_by_uid(self, uid):
        return self.under_fil.sorted_id_by_uid(uid)

    #@ft.wraps(_oineus.Filtration.reset_ids_to_sorted_ids)
    def reset_ids_to_sorted_ids(self):
        self.under_fil.reset_ids_to_sorted_ids()


class TopologyOptimizer:
    """
    A wrapper around C++ topology optimizer
    The C++ class is templated by the Cell type,
    the purpose of this class is to allow the user
    not to worry about the type of a filtration
    """
    def __init__(self, fil):
        if isinstance(fil, DiffFiltration):
            fil = fil.under_fil
        if type(fil) is _oineus.Filtration:
            self.under_opt = _oineus.TopologyOptimizer(fil)
        elif type(fil) is _oineus.ProdFiltration:
            self.under_opt = _oineus.TopologyOptimizerProd(fil)
        else:
            raise RuntimeError("unknown filtration type in oineus.diff.TopologyOptimizer constructor")

    #@ft.wraps(_oineus.TopologyOptimizer.compute_diagram)
    def compute_diagram(self, include_inf_points: bool):
        return self.under_opt.compute_diagram(include_inf_points)

    #@ft.wraps(_oineus.TopologyOptimizer.simplify)
    def simplify(self, epsilon: float, strategy, dim: int):
        return self.under_opt.simplify(epsilon, strategy, dim)

    #@ft.wraps(_oineus.TopologyOptimizer.get_nth_persistence)
    def get_nth_persistence(self, dim: int, n: int):
        return self.under_opt.get_nth_persistence(dim=dim, n=n)

    #@ft.wraps(_oineus.TopologyOptimizer.match)
    def match(self, template_dgm, dim: int, wasserstein_q: float=1.0, return_wasserstein_distance: bool=False):
        return self.under_opt.match(temlate_dgm=template_dgm, dim=dim, wasserstein_q=wasserstein_q,
                                    return_wasserstein_distance=return_wasserstein_distance)

    @property
    def homology_decomposition(self):
        return self.under_opt.homology_decomposition

    @property
    def cohomology_decomposition(self):
        return self.under_opt.cohomology_decomposition

    #@ft.wraps(_oineus.TopologyOptimizer.singleton)
    def singleton(self, index: int, value: float):
        return self.under_opt.singleton(index, value)

    #@ft.wraps(_oineus.TopologyOptimizer.singletons)
    def singletons(self, indices, values):
        return self.under_opt.singletons(indices, values)

    #@ft.wraps(_oineus.TopologyOptimizer.reduce_all)
    def reduce_all(self):
        return self.under_opt.reduce_all()

    #@ft.wraps(_oineus.TopologyOptimizer.update)
    def update(self):
        return self.under_opt.update()

    #@ft.wraps(_oineus.TopologyOptimizer.increase_death)
    def increase_death(self, negative_simplex_idx: int):
        return self.under_opt.increase_death(negative_simplex_idx)

    #@ft.wraps(_oineus.TopologyOptimizer.decrease_death)
    def decrease_death(self, negative_simplex_idx: int):
        return self.under_opt.decrease_death(negative_simplex_idx)

    #@ft.wraps(_oineus.TopologyOptimizer.increase_birth)
    def increase_birth(self, positive_simplex_idx: int):
        return self.under_opt.increase_birth(positive_simplex_idx)

    #@ft.wraps(_oineus.TopologyOptimizer.decrease_birth)
    def decrease_birth(self, positive_simplex_idx: int):
        return self.under_opt.decrease_birth(positive_simplex_idx)

    #@ft.wraps(_oineus.TopologyOptimizer.combine_loss)
    def combine_loss(self, critical_sets, strategy):
        return self.under_opt.combine_loss(critical_sets, strategy)


def min_filtration(fil_1: DiffFiltration, fil_2: DiffFiltration) -> DiffFiltration:
    fil_1_under = fil_1.under_fil
    fil_2_under = fil_2.under_fil

    min_fil_under, inds_1, inds_2 = _oineus._min_filtration_with_indices(fil_1_under, fil_2_under)

    inds_1 = np.array(inds_1)
    inds_2 = np.array(inds_2)

    vals_1 = epy.astensor(fil_1.values)[inds_1]
    vals_2 = epy.astensor(fil_2.values)[inds_2]

    min_fil_values = epy.min(epy.stack((vals_1, vals_2)), axis=0).raw

    return DiffFiltration(min_fil_under, min_fil_values)


def freudenthal_filtration(data, negate, wrap, max_dim, n_threads):
    data = epy.astensor(data)
    np_data = data.float64().numpy()
    fil, cv = _oineus.get_freudenthal_filtration_and_crit_vertices(np_data, negate, wrap, max_dim, n_threads)
    cv = np.array(cv, dtype=np.int64)
    values = data.flatten()[cv].raw
    return DiffFiltration(fil, values)


def vr_filtration(data, from_pwdists: bool=False, max_dim: int=-1, max_diameter: float=-1.0, eps=1e-6, n_threads=8) -> DiffFiltration:

    data = epy.astensor(data)
    data_np = data.float64().numpy()
    assert(data.ndim == 2)

    fil, edges = non_diff_vr_filtration(data=data_np, from_pwdists=from_pwdists, with_critical_edges=True,
                                        max_dim=max_dim, max_diameter=max_diameter, n_threads=n_threads)

    if not from_pwdists:
        sqdists = epy.sum((data[edges[:, 0].flatten()] - data[edges[:, 1].flatten()]) ** 2, axis=1) + eps
        diff_dists = epy.sqrt(sqdists).raw

        return DiffFiltration(fil, diff_dists)
    else:
        edges = epy.astensor(edges)
        diff_dists = data[edges[:, 0], edges[:, 1]].raw
        return DiffFiltration(fil, diff_dists)


def triangle_meb(p0, p1, p2, eps=0.0):
    """
    Compute minimum enclosing ball center and radius squared for triangles.

    Args:
        p0, p1, p2: Tensor of shape (n, d) for n triangles in d dimensions
        eps: Small value for numerical stability

    Returns:
        centers: Tensor of shape (n, d) - MEB centers
        radii_sq: Tensor of shape (n,) - MEB radii squared
    """
    # Edge vectors
    a = p1 - p0
    b = p2 - p0
    c = p2 - p1

    # Edge lengths squared
    a_sq = torch.sum(a ** 2, dim=1)
    b_sq = torch.sum(b ** 2, dim=1)
    c_sq = torch.sum(c ** 2, dim=1)

    # Area of triangle using cross product (2 * area)
    d = p0.shape[1]
    if d == 2:
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        area_2_sq = cross ** 2
    else:  # 3D case
        cross = torch.cross(a, b, dim=1)
        area_2_sq = torch.sum(cross ** 2, dim=1)

    # Circumradius squared: R^2 = (a^2 * b^2 * c^2) / (16 * Area^2)
    circum_radii_sq = (a_sq * b_sq * c_sq + eps) / (4 * area_2_sq + eps)

    # Circumcenter computation
    # Using barycentric coordinates: circumcenter = u*p0 + v*p1 + w*p2
    # where u = |c|²(a·b), v = |b|²(a·c), w = |a|²(b·c) (before normalization)
    # But easier: circumcenter = p0 + (|a|²(b·b) - |b|²(a·b)) * (b×(b×a)) / (2|a×b|²) ...
    # Actually, let's use the standard formula:
    # circumcenter = p0 + ((|a|²)(b) - (|b|²)(a)) × (a × b) / (2|a × b|²)

    # For 3D points:
    if d == 3:
        cross_ab = torch.cross(a, b, dim=1)
        cross_ab_sq = torch.sum(cross_ab ** 2, dim=1, keepdim=True)

        a_dot_a = a_sq.unsqueeze(1)
        b_dot_b = b_sq.unsqueeze(1)

        b_cross_axb = torch.cross(b, cross_ab, dim=1)
        axb_cross_a = torch.cross(cross_ab, a, dim=1)

        circum_centers = p0 + (a_dot_a * b_cross_axb + b_dot_b * axb_cross_a) / (2 * cross_ab_sq + eps)
    else:
        # 2D case - use perpendicular bisector intersection
        # circumcenter = p0 + ((|a|²)(b_perp) - (|b|²)(a_perp)) where perp rotates 90 deg
        # Actually simpler: use the formula with 2x2 system
        a_dot_a = a_sq.unsqueeze(1)
        b_dot_b = b_sq.unsqueeze(1)

        # D = 2(a_x * b_y - a_y * b_x) = 2 * cross (scalar)
        D = 2 * (a[:, 0:1] * b[:, 1:2] - a[:, 1:2] * b[:, 0:1])

        # circumcenter = p0 + (1/D) * [b_y * |a|² - a_y * |b|², a_x * |b|² - b_x * |a|²]
        ux = (b[:, 1:2] * a_dot_a - a[:, 1:2] * b_dot_b) / (D + eps)
        uy = (a[:, 0:1] * b_dot_b - b[:, 0:1] * a_dot_a) / (D + eps)
        circum_centers = p0 + torch.cat([ux, uy], dim=1)

    # For obtuse triangles, MEB center is midpoint of longest edge
    abc_sq = torch.stack((a_sq, b_sq, c_sq), dim=0)
    s_abc_sq, sort_idx = torch.sort(abc_sq, dim=0)

    # Triangle is obtuse if longest² > sum of other two squared
    obtuse_mask = s_abc_sq[2, :] > s_abc_sq[0, :] + s_abc_sq[1, :]

    # Determine which edge is longest and compute its midpoint
    # Edge 0 (a): p0-p1, midpoint = (p0+p1)/2
    # Edge 1 (b): p0-p2, midpoint = (p0+p2)/2
    # Edge 2 (c): p1-p2, midpoint = (p1+p2)/2
    longest_edge_idx = sort_idx[2, :]  # Index of longest edge for each triangle

    midpoint_a = (p0 + p1) / 2
    midpoint_b = (p0 + p2) / 2
    midpoint_c = (p1 + p2) / 2

    # Initialize with circumcenter values
    centers = circum_centers.clone()
    radii_sq = circum_radii_sq.clone()

    # Update obtuse triangles
    if obtuse_mask.any():
        obtuse_longest = longest_edge_idx[obtuse_mask]

        # For obtuse triangles, set center to midpoint of longest edge
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

        # Radius squared = longest edge squared / 4
        radii_sq[obtuse_mask] = s_abc_sq[2, obtuse_mask] / 4

    return centers, radii_sq


def cech_delaunay_filtration(alpha_fil, points, eps=0.0):
    """
    :param alpha_fil: Alpha filtration from diode or oineus
    :param points: Tensor of point coordinates
    :param eps: Small value for numerical stability
    :return: differentiable Cech-Delaunay filtration
    """
    if type(alpha_fil) is not _oineus.Filtration:
        alpha_fil = _oineus.Filtration([_oineus.Simplex(vs, val) for vs, val in alpha_fil])

    values_in_dim = [torch.zeros(alpha_fil.size_in_dimension(0), requires_grad=True)]

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

            _, radii_sq = triangle_meb(p0, p1, p2, eps)
            assert triangles.shape[0] == radii_sq.shape[0]

        elif dim == 3:
            tetra = torch.LongTensor(alpha_fil.get_tetrahedra().astype(np.uint64))
            p0 = points[tetra[:, 0]]
            p1 = points[tetra[:, 1]]
            p2 = points[tetra[:, 2]]
            p3 = points[tetra[:, 3]]

            # Compute circumradius of tetrahedron
            a = p1 - p0
            b = p2 - p0
            c = p3 - p0

            a_sq = torch.sum(a ** 2, dim=1, keepdim=True)
            b_sq = torch.sum(b ** 2, dim=1, keepdim=True)
            c_sq = torch.sum(c ** 2, dim=1, keepdim=True)

            cross_bc = torch.cross(b, c, dim=1)
            cross_ca = torch.cross(c, a, dim=1)
            cross_ab = torch.cross(a, b, dim=1)

            # 6V = a · (b × c)
            volume_6 = torch.sum(a * cross_bc, dim=1)
            volume_6_abs = torch.abs(volume_6)

            # Circumcenter displacement from p0
            numerator_vec = a_sq * cross_bc + b_sq * cross_ca + c_sq * cross_ab
            circum_disp = numerator_vec / (2 * volume_6.unsqueeze(1) + eps)
            circum_center = p0 + circum_disp

            # Circumradius squared
            circum_radii_sq = torch.sum(circum_disp ** 2, dim=1)

            # Compute MEB for each of the 4 faces and check if opposite vertex is inside
            # Face 0: p1, p2, p3 (opposite to p0)
            # Face 1: p0, p2, p3 (opposite to p1)
            # Face 2: p0, p1, p3 (opposite to p2)
            # Face 3: p0, p1, p2 (opposite to p3)

            face_centers_0, face_radii_sq_0 = triangle_meb(p1, p2, p3, eps)
            face_centers_1, face_radii_sq_1 = triangle_meb(p0, p2, p3, eps)
            face_centers_2, face_radii_sq_2 = triangle_meb(p0, p1, p3, eps)
            face_centers_3, face_radii_sq_3 = triangle_meb(p0, p1, p2, eps)

            # Check if opposite vertex is contained in face's MEB
            # Vertex is contained if distance² from center <= radius²
            dist_sq_0 = torch.sum((p0 - face_centers_0) ** 2, dim=1)  # p0 to face 0's MEB
            dist_sq_1 = torch.sum((p1 - face_centers_1) ** 2, dim=1)  # p1 to face 1's MEB
            dist_sq_2 = torch.sum((p2 - face_centers_2) ** 2, dim=1)  # p2 to face 2's MEB
            dist_sq_3 = torch.sum((p3 - face_centers_3) ** 2, dim=1)  # p3 to face 3's MEB

            # Masks for faces whose MEB contains the opposite vertex
            contains_0 = dist_sq_0 <= face_radii_sq_0 + eps
            contains_1 = dist_sq_1 <= face_radii_sq_1 + eps
            contains_2 = dist_sq_2 <= face_radii_sq_2 + eps
            contains_3 = dist_sq_3 <= face_radii_sq_3 + eps

            # Set radii to infinity where face MEB doesn't contain opposite vertex
            inf_val = torch.tensor(float('inf'), dtype=points.dtype, device=points.device)

            face_radii_sq_0_masked = torch.where(contains_0, face_radii_sq_0, inf_val)
            face_radii_sq_1_masked = torch.where(contains_1, face_radii_sq_1, inf_val)
            face_radii_sq_2_masked = torch.where(contains_2, face_radii_sq_2, inf_val)
            face_radii_sq_3_masked = torch.where(contains_3, face_radii_sq_3, inf_val)

            # Stack all candidate radii and take minimum
            all_radii_sq = torch.stack([
                circum_radii_sq,
                face_radii_sq_0_masked,
                face_radii_sq_1_masked,
                face_radii_sq_2_masked,
                face_radii_sq_3_masked
            ], dim=0)

            radii_sq = torch.min(all_radii_sq, dim=0).values

            assert tetra.shape[0] == radii_sq.shape[0]

        values_in_dim.append(radii_sq)

    # this will sort the simplices in the filtration correctly, cd_vals_np is not monotonic
    cd_vals = torch.cat(values_in_dim)
    cd_vals_list = [ float(x) for x in cd_vals.clone().detach().cpu() ]
    alpha_fil.set_values(cd_vals_list)
    sorted_vals = torch.cat([ torch.sort(vals)[0] for vals in values_in_dim])
    return DiffFiltration(alpha_fil, sorted_vals)


def mapping_cylinder_filtration(fil_domain: DiffFiltration, fil_codomain: DiffFiltration, v_domain, v_codomain) -> DiffFiltration:
    assert(type(fil_domain) is DiffFiltration)
    assert(type(fil_codomain) is DiffFiltration)

    if isinstance(v_domain, _oineus.Simplex):
        v_domain = v_domain.combinatorial_cell()

    if isinstance(v_codomain, _oineus.Simplex):
        v_codomain = v_codomain.combinatorial_cell()

    under_fil_dom = fil_domain.under_fil
    under_fil_cod = fil_codomain.under_fil

    under_cyl_fil, cyl_val_inds = _oineus._mapping_cylinder_with_indices(under_fil_dom, under_fil_cod, v_domain, v_codomain)

    cyl_val_inds = epy.astensor(np.array(cyl_val_inds, dtype=np.int64))

    concat_vals = epy.concatenate((epy.astensor(fil_domain.values), epy.astensor(fil_codomain.values)))

    assert(concat_vals.ndim == 1 and concat_vals.shape[0] == fil_domain.size() + fil_codomain.size())

    cyl_values = concat_vals[cyl_val_inds].raw

    return DiffFiltration(under_cyl_fil, cyl_values)


# =============================================================================
# Differentiable Persistence Diagrams with PyTorch integration
# =============================================================================

# if TORCH_AVAILABLE:

class PersistenceDiagramHelper(torch.autograd.Function):
    """
    Autograd function for extracting a single dimension's diagram.

    Forward: subscripts fil.values at birth/death indices
    Backward: accumulates gradients at those indices (dgm-loss)
              or expands to critical sets (crit-sets)
    """

    @staticmethod
    def forward(ctx, fil_values, fil, dcmp_hom, dcmp_coh, dgms, dim, include_inf_points,
                gradient_method, lr, conflict_strategy):
        """
        Extract diagram for dimension `dim` as a differentiable tensor.
        """
        # Get index diagram (list of IndexDiagramPoint)
        index_dgm = dgms.index_diagram_in_dimension(dim, as_numpy=True).astype(np.int64)
        index_dgm = torch.from_numpy(index_dgm).to(fil_values.device)

        fil_len = fil_values.shape[0]

        ctx.gradient_method = gradient_method
        ctx.fil_len = fil_len
        ctx.include_inf_points = include_inf_points
        ctx.lr = lr
        ctx.conflict_strategy = conflict_strategy
        ctx.dim = dim

        n_total = len(index_dgm)

        if n_total == 0:
            return torch.zeros((0, 2), dtype=fil_values.dtype, device=fil_values.device)

        if include_inf_points:
            fin_mask = (index_dgm[:, 0] >= 0) & (index_dgm[:, 0] < fil_len) & (index_dgm[:, 1] >= 0) & (index_dgm[:, 1] < fil_len)
            fin_idx_dgm = index_dgm[fin_mask]
            inf_births_inds = index_dgm[~fin_mask][:, 0]
            finite_dgm = fil_values[fin_idx_dgm]
            inf_births = fil_values[inf_births_inds]
            inf_deaths = torch.full_like(inf_births, float('inf'), dtype=fil_values.dtype, device=fil_values.device)
            inf_dgm = torch.stack([inf_births, inf_deaths], dim=1)
            diagram = torch.cat([finite_dgm, inf_dgm], dim=0)
        else:
            diagram = fil_values[index_dgm]

        # Save for backward
        ctx.save_for_backward(index_dgm)

        return diagram

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: propagate gradients to fil_values.
        """
        if ctx.gradient_method == "dgm-loss":
            # w.r.t. fil_values, fil, dcmp_hom, dcmp_coh, dgms, dim,  include_inf_points, gradient_method, lr, conflict_strategy
            fil_len = ctx.fil_len
            index_dgm = ctx.saved_tensors[0]
            grad_vals = torch.zeros(fil_len, dtype=grad_output.dtype, device=grad_output.device)
            grad_vals[index_dgm.flatten()] = grad_output.flatten()
            return grad_vals, None, None, None, None, None, None, None, None, None
        else:
            raise RuntimeError("Gradient method not implemented yet.")


class PersistenceDiagrams:
    """
    Container for differentiable persistence diagrams in all dimensions.
    Provides access to diagrams in each dimension as differentiable tensors.

    Usage:
        dgms = persistence_diagram(fil, dualize=True)
        dgm1 = dgms[1]  # H1 diagram as tensor (N, 2)
        loss = dgm1[:, 1].sum()
        loss.backward()
    """

    def __init__(self, fil: DiffFiltration, dualize: bool, include_inf_points: bool,
                 gradient_method: str, lr: float, conflict_strategy: str, rp = None):
        if not isinstance(fil.values, torch.Tensor):
            raise TypeError("fil.values must be a torch.Tensor for differentiable diagrams")

        if rp is None:
            rp = _oineus.ReductionParams()

        self._fil = fil
        self._dualize = dualize
        dcmp = _oineus.Decomposition(fil.under_fil, dualize=dualize)
        dcmp.reduce(rp)
        if dualize:
            self._dcmp_coh, self._dcmp_hom = dcmp, None
        else:
            self._dcmp_coh, self._dcmp_hom = None, dcmp
        self._include_inf_points = include_inf_points
        self._gradient_method = gradient_method
        self._lr = lr
        self._conflict_strategy = conflict_strategy

        nondiff_dgms = dcmp.diagram(fil.under_fil, include_inf_points=include_inf_points)

        self._diagrams = { dim : PersistenceDiagramHelper.apply(
                self._fil.values,
                self._fil.under_fil,
                self._dcmp_hom,
                self._dcmp_coh,
                nondiff_dgms,
                dim,
                self._include_inf_points,
                self._gradient_method,
                self._lr,
                self._conflict_strategy
            )
        for dim in range(self._fil.max_dim()) }


    def __getitem__(self, dim: int) -> torch.Tensor:
        """Get diagram in dimension dim."""
        if dim not in self._diagrams:
            raise KeyError(f"No diagram for dimension {dim}. Available: {list(self._diagrams.keys())}")
        return self._diagrams[dim]

    def __contains__(self, dim: int) -> bool:
        return dim in self._diagrams

    def __len__(self) -> int:
        return len(self._diagrams)

    def __iter__(self):
        return iter(self._diagrams)

    def keys(self):
        return self._diagrams.keys()

    def values(self):
        return self._diagrams.values()

    def items(self):
        return self._diagrams.items()

    def in_dimension(self, dim: int) -> torch.Tensor:
        """Alias for __getitem__."""
        return self[dim]

    @property
    def max_dim(self) -> int:
        """Maximum dimension available."""
        return max(self._diagrams.keys())


def persistence_diagram(
    fil: DiffFiltration,
    dualize: bool = False,
    include_inf_points: bool = False,
    gradient_method: str = "dgm-loss",
    lr: float = 1.0,
    conflict_strategy: str = "avg"
) -> PersistenceDiagrams:
    """
    Compute differentiable persistence diagrams from a DiffFiltration.

    Efficiently computes the decomposition once and returns diagrams for
    all dimensions. The returned object can be indexed by dimension.

    Args:
        fil: DiffFiltration with differentiable `values` tensor
        dualize: If True, compute cohomology (default). If False, homology.
        include_inf_points: If True, include infinite points in diagrams.
                           Infinite deaths are represented as float('inf').
        gradient_method: "dgm-loss" (gradient to critical simplices only)
                        or "crit-sets" (gradient to all simplices in critical sets)
        lr: Learning rate for crit-sets target computation (default: 1.0)
        conflict_strategy: Conflict resolution for crit-sets: "avg", "max", "sum"

    Returns:
        PersistenceDiagrams: dict-like object mapping dimension -> Tensor (N, 2)
        Each tensor has birth values in column 0, death values in column 1.
        Gradients flow back to fil.values automatically.

    Example:
        # Create VR filtration from points
        pts = torch.tensor([[0., 0.], [1., 0.], [0.5, 1.]], requires_grad=True)
        fil = oin.diff.vr_filtration(pts, max_dim=2)

        # Get all diagrams
        dgms = oin.diff.persistence_diagram(fil, dualize=True)

        # Access specific dimension
        dgm1 = dgms[1]  # H1 diagram

        # Compute loss and backpropagate
        loss = (dgm1[:, 1] - 2.0).pow(2).sum()
        loss.backward()
        print(pts.grad)
    """
    return PersistenceDiagrams(
        fil=fil,
        dualize=dualize,
        include_inf_points=include_inf_points,
        gradient_method=gradient_method,
        lr=lr,
        conflict_strategy=conflict_strategy
    )
