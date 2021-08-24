from __future__ import absolute_import

import numpy as np
import scipy.sparse

from . import _oineus

__all__ = ["compute_diagrams_ls", "compute_diagrams_and_v_ls", "get_boundary_matrix", "to_scipy_matrix", "is_reduced"]


def to_scipy_matrix(sparse_cols, shape=None):
    if shape is None:
        shape = (len(sparse_cols), len(sparse_cols))
    row_ind = [ i for i in range(len(sparse_cols)) for _ in sparse_cols[i]]
    col_ind = [ j for i in range(len(sparse_cols)) for j in sparse_cols[i]]
    assert(len(row_ind) == len(col_ind))
    data = [ 1 for _ in range(len(row_ind)) ]
    return scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=shape)


def is_reduced(a):
    lowest_ones = []
    for col_idx in range(a.shape[1]):
        if np.any(a[:, col_idx] % 2 == 1):
            lowest_ones.append(np.max(np.where(a[:, col_idx] % 2 == 1)))
    lowest_ones.sort()
    print(lowest_ones)
    return len(lowest_ones) == len(set(lowest_ones))



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


def get_boundary_matrix(data, negate, wrap, top_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"get_boundary_matrix_{type_part}_{dim_part}")
    bm = func(data, negate, wrap, top_dim, n_threads)
    return to_scipy_matrix(bm)


def compute_diagrams_ls(data, negate, wrap, top_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"compute_diagrams_ls_{type_part}_{dim_part}")
    return func(data, negate, wrap, top_dim, n_threads)


def compute_diagrams_and_v_ls(data, negate, wrap, top_dim, n_threads):
    type_part, dim_part = get_type_dim(data)
    func = getattr(_oineus, f"compute_diagrams_and_v_ls_{type_part}_{dim_part}")
    dgms, v = func(data, negate, wrap, top_dim, n_threads)
    v = to_scipy_matrix(v)
    return dgms, v