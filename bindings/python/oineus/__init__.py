from __future__ import absolute_import

import numpy as np
import scipy.sparse

from . import _oineus

from ._oineus import *

# __all__ = ["compute_diagrams_ls", "compute_diagrams_and_v_ls", "get_boundary_matrix", "to_scipy_matrix", "is_reduced", "get_freudenthal_filtration"]


def to_scipy_matrix(sparse_cols, shape=None):
    if shape is None:
        shape = (len(sparse_cols), len(sparse_cols))
    row_ind = [ j for i in range(len(sparse_cols)) for j in sparse_cols[i]]
    col_ind = [ i for i in range(len(sparse_cols)) for _ in sparse_cols[i]]
    assert(len(row_ind) == len(col_ind))
    data = [ 1 for _ in range(len(row_ind)) ]
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
        return "double"
    else:
        raise RuntimeError(f"Unknown type: {type(fil)}")


def get_type_dim(data):
    if data.dtype == np.float32:
        type_part = "float"
    elif data.dtype == np.float64:
        type_part = "double"
    else:
        raise RuntimeError(f"Type not supported: {data.dtype}")

    if data.ndim in [1, 2, 3]:
        dim_part = str(data.ndim)
    else:
        raise RuntimeError(f"Dimension not supported: {data.ndim}")

    return type_part, dim_part

def get_freudenthal_filtration(data, negate, wrap, max_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"get_fr_filtration_{type_part}_{dim_part}")
    return func(data, negate, wrap, max_dim, n_threads)

def get_vr_filtration(points, max_dim, max_radius, n_threads):
    type_part, dim_part = get_type_dim(points)
    func = getattr(_oineus, f"get_vr_filtration_{type_part}_{dim_part}")
    return func(points, max_dim, max_radius, n_threads)


def get_boundary_matrix(data, negate, wrap, max_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"get_boundary_matrix_{type_part}_{dim_part}")
    bm = func(data, negate, wrap, max_dim, n_threads)
    return to_scipy_matrix(bm)


def compute_diagrams_ls(data, negate, wrap, max_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"compute_diagrams_ls_{type_part}_{dim_part}")
    return func(data, negate, wrap, max_dim, n_threads)


def compute_diagrams_and_v_ls(data, negate, wrap, max_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"compute_diagrams_and_v_ls_{type_part}_{dim_part}")
    dgms, v = func(data, negate, wrap, max_dim, n_threads)
    v = to_scipy_matrix(v)
    return dgms, v

def compute_diagrams_and_rv_ls(data, negate, wrap, max_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"compute_diagrams_and_rv_ls_{type_part}_{dim_part}")
    dgms, r, v = func(data, negate, wrap, max_dim, n_threads)
    v = to_scipy_matrix(v)
    r = to_scipy_matrix(r)
    return dgms, r, v


def get_denoise_target(d, fil, rv, eps, strat):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_denoise_target_{type_part}")
    return func(d, fil, rv, eps, strat)


def get_ls_target_values(d, dtv, fil, rv):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_ls_target_values_{type_part}")
    return func(d, dtv, fil, rv)


def get_ls_target_values_diagram_loss(d, dtv, fil, rv):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_ls_target_values_diagram_loss_{type_part}")
    return func(d, dtv, fil)


def get_vr_target_values(d, dtv, fil, rv):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_vr_target_values_{type_part}")
    return func(d, dtv, fil, rv)


def get_ls_wasserstein_matching_target_values(dgm, fil, rv, d, q):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_ls_wasserstein_matching_target_values_{type_part}")

    if type(dgm) is np.ndarray:
        DgmPt = getattr(_oineus, f"DiagramPoint_{type_part}")
        dgm_1 = []
        assert len(dgm.shape) == 2 and dgm.shape[1] == 2
        for p in dgm:
            dgm_1.append(DgmPt(p[0], p[1]))
        dgm = dgm_1

    return func(dgm, fil, rv, d, q)


def get_vr_target_values_diagram_loss(d, dtv, fil, rv):
    type_part = get_real_type(fil)
    func = getattr(_oineus, f"get_vr_target_values_diagram_loss_{type_part}")
    return func(d, dtv, fil)