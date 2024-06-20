from __future__ import absolute_import

import typing
import numpy as np
import scipy.sparse

from . import _oineus

from ._oineus import *

import warnings

try:
    from . import diff
except:
    warnings.warn("oineus.diff import failed, probably, because eagerpy is not installed")


# __all__ = ["compute_diagrams_ls", "compute_diagrams_and_v_ls", "get_boundary_matrix", "to_scipy_matrix", "is_reduced", "get_freudenthal_filtration"]


def to_scipy_matrix(sparse_cols, shape=None):
    if shape is None:
        shape = (len(sparse_cols), len(sparse_cols))
    row_ind = [j for i in range(len(sparse_cols)) for j in sparse_cols[i]]
    col_ind = [i for i in range(len(sparse_cols)) for _ in sparse_cols[i]]
    assert (len(row_ind) == len(col_ind))
    data = [1 for _ in range(len(row_ind))]
    return scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=shape)


def is_reduced(a):
    lowest_ones = []
    for col_idx in range(a.shape[1]):
        if np.any(a[:, col_idx] % 2 == 1):
            lowest_ones.append(np.max(np.where(a[:, col_idx] % 2 == 1)))
    return len(lowest_ones) == len(set(lowest_ones))


def get_real_type(fil):
    if "_double" in str(type(fil)):
        return "double"
    elif "_float" in str(type(fil)):
        return "float"
    else:
        raise RuntimeError(f"Unknown type: {type(fil)}")


def get_type_dim(data: np.ndarray, points=False):
    if data.dtype == np.float32:
        type_part = "float"
    elif data.dtype == np.float64:
        type_part = "double"
    else:
        raise RuntimeError(f"Type not supported: {data.dtype}")

    if points:
        if data.ndim == 2 and data.shape[1] in [1, 2, 3, 4]:
            dim_part = str(data.shape[1])
        else:
            raise RuntimeError(f"Dimension not supported: shape = {data.shape}")
    else:
        if data.ndim in [1, 2, 3]:
            dim_part = str(data.ndim)
        else:
            raise RuntimeError(f"Dimension not supported: shape = {data.shape}")

    return type_part, dim_part


def get_freudenthal_filtration(data, negate, wrap, max_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"get_fr_filtration_{type_part}_{dim_part}")
    return func(data, negate, wrap, max_dim, n_threads)

def get_freudenthal_filtration_and_critical_vertices(data, negate, wrap, max_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"get_fr_filtration_and_critical_vertices_{type_part}_{dim_part}")
    fil, cv = func(data, negate, wrap, max_dim, n_threads)
    return fil, np.array(cv)


def get_vr_filtration_from_pwdists(pwdists, max_dim, max_radius, n_threads):
    type_part, _ = get_type_dim(pwdists, False)
    func = getattr(_oineus, f"get_vr_filtration_from_pwdists_{type_part}")
    return func(pwdists, max_dim, max_radius, n_threads)

def get_vr_filtration_and_critical_edges_from_pwdists(pwdists, max_dim, max_radius, n_threads):
    type_part, _ = get_type_dim(pwdists, False)
    func = getattr(_oineus, f"get_vr_filtration_and_critical_edges_from_pwdists_{type_part}")
    fil, edges = func(pwdists, max_dim, max_radius, n_threads)
    edges = np.array([[e.x, e.y] for e in edges])
    return fil, edges


def get_vr_filtration(points, max_dim, max_radius, n_threads):
    type_part, dim_part = get_type_dim(points, True)
    func = getattr(_oineus, f"get_vr_filtration_{type_part}_{dim_part}")
    return func(points, max_dim, max_radius, n_threads)

def get_vr_filtration_and_critical_edges(points, max_dim, max_radius, n_threads):
    type_part, dim_part = get_type_dim(points, True)
    func = getattr(_oineus, f"get_vr_filtration_and_critical_edges_{type_part}_{dim_part}")
    fil, edges = func(points, max_dim, max_radius, n_threads)
    edges = np.array([[e.x, e.y] for e in edges])
    return fil, edges


def get_boundary_matrix(data, negate, wrap, max_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"get_boundary_matrix_{type_part}_{dim_part}")
    bm = func(data, negate, wrap, max_dim, n_threads)
    return to_scipy_matrix(bm)


def compute_diagrams_ls(data, negate, wrap, max_dim, params, include_inf_points, dualize):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"compute_diagrams_ls_{type_part}_{dim_part}")
    return func(data, negate, wrap, max_dim, params, include_inf_points, dualize)


def get_denoise_target(d, fil, rv, eps, strat):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_denoise_target_{type_part}")
    return func(d, fil, rv, eps, strat)


def get_well_group_target(d, fil, rv, t):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_well_group_target_{type_part}")
    return func(d, fil, rv, t)


def get_ls_target_values_diagram_loss(d, dtv, fil):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_target_values_diagram_loss_{type_part}")
    return func(dtv, False)


def get_vr_target_values_diagram_loss(d, dtv, fil):
    type_part = get_real_type(fil)
    death_only = d == 0
    func = getattr(_oineus, f"get_target_values_diagram_loss_{type_part}")
    return func(dtv, death_only)



def get_ls_target_values_x(d, dtv, fil, decmp, decmp_coh, conflict_strategy):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_ls_target_values_x_{type_part}")
    return func(d, dtv, fil, decmp, decmp_coh, conflict_strategy, False)


def get_vr_target_values_x(d, dtv, fil, decmp, decmp_coh, conflict_strategy):
    type_part = get_real_type(fil)
    death_only = d == 0
    func = getattr(_oineus, f"get_vr_target_values_x_{type_part}")
    return func(d, dtv, fil, decmp, decmp_coh, conflict_strategy, death_only)



def get_ls_wasserstein_matching_target_values(dgm, fil, rv, d: int, q: float, mip: bool, mdp: bool):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_ls_wasserstein_matching_target_values_{type_part}")

    if type(dgm) is np.ndarray:
        DgmPt = getattr(_oineus, f"DiagramPoint_{type_part}")
        dgm_1 = []
        assert len(dgm.shape) == 2 and dgm.shape[1] == 2
        for p in dgm:
            dgm_1.append(DgmPt(p[0], p[1]))
        dgm = dgm_1

    return func(dgm, fil, rv, d, q, mip, mdp)



def get_bruelle_target(fil, rv, p, q, i_0, d, minimize, min_birth, max_death):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_bruelle_target_{type_part}")
    return func(fil, rv, p, q, i_0, d, minimize, min_birth, max_death)


def get_barycenter_target(fil, rv, d):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_barycenter_target_{type_part}")
    return func(fil, rv, d)


def get_nth_persistence(fil, rv, d, n):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_nth_persistence_{type_part}")
    return func(fil, rv, d, n)


def get_permutation(target_values, fil):
    if len(target_values) == 0:
        return {}
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_permutation_{type_part}")
    return func(target_values, fil)


def list_to_filtration(simplex_list): #take a list which contains data for simplices and convert it to a filtration
    string_type = str(type(simplex_list[0][2]))
    if "int" in string_type:
        func = getattr(_oineus, f"list_to_filtration_int")
        return func(simplex_list)
    elif "float" in string_type:
        func = getattr(_oineus, f"list_to_filtration_float")
        return func(simplex_list)
    elif "double" in string_type:
        func = getattr(_oineus, f"list_to_filtration_double")
        return func(simplex_list)

def compute_kernel_image_cokernel_reduction(K_, L_, IdMap, n_threads): #
    string_type = str(type(K_[0][2]))
    func = getattr(_oineus, f"compute_kernel_image_cokernel_reduction_float")
    return func(K_, L_, IdMap, n_threads)


def get_ls_filtration(simplices: typing.List[typing.List[int]], vertex_values: np.ndarray, negate: bool, n_threads: int):
    if vertex_values.dtype == np.float32:
        func = getattr(_oineus, f"get_ls_filtration_float")
    elif vertex_values.dtype == np.float64:
        func = getattr(_oineus, f"get_ls_filtration_double")
    return func(simplices, vertex_values, negate, n_threads)


def compute_ker_im_cok_reduction_cyl(fil_2, fil_3):
    fil_min = _oineus.min_filtration(fil_2, fil_3)

    id_domain = fil_3.size() + fil_min.size() + 1
    id_codomain = id_domain + 1

    # id_domain: id of vertex at the top of the cylinder,
    # i.e., we multiply fil_3 with id_domain
    # id_codomain: id of vertex at the bottom of the cylinder
    # i.e, we multiply fil_min with id_codomain

    v0 = _oineus.Simplex(id_domain, [id_domain])
    v1 = _oineus.Simplex(id_codomain, [id_codomain])

    fil_cyl = _oineus.mapping_cylinder(fil_3, fil_min, v0, v1)

    # to get a subcomplex, we multiply each fil_3 with id_domain
    fil_3_prod = _oineus.multiply_filtration(fil_3, v0)

    params = _oineus.ReductionParams()
    params.kernel = params.cokernel = True
    params.image = False

    kicr_reduction = _oineus.KerImCokReducedProd_double(fil_cyl, fil_3_prod, params)
    return kicr_reduction

def compute_relative_diagrams(fil, rel, include_inf_points=True):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"compute_relative_diagrams_{type_part}")
    return func(fil, rel, include_inf_points)

#def compute_cokernel_diagrams(K_, L_, IdMap, n_threads): #
#    string_type = str(type(K_[0][2]))
#    func = getattr(_oineus, f"compute_cokernel_diagrams_float")
#    return func(K_, L_, IdMap, n_threads)
#

#def lists_to_paired_filtrations(simplex_list_1, simplex_list_2)
