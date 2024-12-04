from __future__ import absolute_import

import typing
import numpy as np
import scipy.sparse

from . import _oineus

from ._oineus import *

import warnings

from icecream import ic

try:
    from . import diff
except:
    warnings.warn("oineus.diff import failed, probably, because eagerpy is not installed")


__all__ = ["compute_diagrams_ls", "compute_diagrams_vr", "get_boundary_matrix", "is_reduced"]


def to_scipy_matrix(sparse_cols, shape=None):
    if shape is None:
        shape = (len(sparse_cols), len(sparse_cols))
    row_ind = [j for i in range(len(sparse_cols)) for j in sparse_cols[i]]
    col_ind = [i for i in range(len(sparse_cols)) for _ in sparse_cols[i]]
    assert (len(row_ind) == len(col_ind))
    data = [1 for _ in range(len(row_ind))]
    return scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=shape)


def max_distance(data: np.ndarray, from_pwdists: bool=False):
    if from_pwdists:
        return 1.00001 * np.min(np.max(data, axis=1))
    else:
        assert data.ndim == 2 and data.shape[0] >= 2
        diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
        squared_distances = np.sum(diff**2, axis=2)
        return 1.00001 * np.sqrt(np.min(np.max(squared_distances, axis=1)))


def freudenthal_filtration(data: np.ndarray,
                           negate: bool=False,
                           wrap: bool=False,
                           max_dim: int = 3,
                           with_critical_vertices: bool=False,
                           n_threads: int=1):
    max_dim = min(max_dim, data.ndim)
    if with_critical_vertices:
        return _oineus.get_freudenthal_filtration_and_critical_vertices(data=data, negate=negate, wrap=wrap, max_dim=max_dim, n_threads=n_threads)
    else:
        return _oineus.get_freudenthal_filtration(data=data, negate=negate, wrap=wrap, max_dim=max_dim, n_threads=n_threads)


def vr_filtration(data: np.ndarray,
                  from_pwdists: bool = False,
                  max_dim: int = -1,
                  max_diameter: float = -1.0,
                  with_critical_edges: bool = False,
                  n_threads: int = 1):
    assert data.ndim == 2

    if from_pwdists:
        assert data.shape[0] == data.shape[1]

    if max_diameter < 0:
        max_diameter = max_distance(data, from_pwdists)

    ic(max_dim, max_diameter)

    if max_dim < 0:
        if from_pwdists:
            raise RuntimeError("vr_filtration: if input is pairwise distance matrix, max_dim must be specified")
        else:
            max_dim = data.shape[1]

    if from_pwdists:
        if with_critical_edges:
            func = _oineus.get_vr_filtration_and_critical_edges_from_pwdists
        else:
            func = _oineus.get_vr_filtration_from_pwdists
    else:
        if with_critical_edges:
            func = _oineus.get_vr_filtration_and_critical_edges
        else:
            func = _oineus.get_vr_filtration

    return func(data, max_dim=max_dim, max_diameter=max_diameter, n_threads=n_threads)


def is_reduced(a):
    lowest_ones = []
    for col_idx in range(a.shape[1]):
        if np.any(a[:, col_idx] % 2 == 1):
            lowest_ones.append(np.max(np.where(a[:, col_idx] % 2 == 1)))
    return len(lowest_ones) == len(set(lowest_ones))


def get_dim(data: np.ndarray, points=False):
    if points:
        if data.ndim == 2 and data.shape[1] in [1, 2, 3, 4]:
            return data.shape[1]
        else:
            raise RuntimeError(f"Dimension not supported: shape = {data.shape}")
    else:
        if data.ndim in [1, 2, 3]:
            return data.ndim
        else:
            raise RuntimeError(f"Dimension not supported: shape = {data.shape}")


def get_boundary_matrix(data, negate, wrap, max_dim, n_threads):
    dim_part = get_dim(data)
    func = getattr(_oineus, f"get_boundary_matrix_{dim_part}")
    bm = func(data, negate, wrap, max_dim, n_threads)
    return to_scipy_matrix(bm)


def compute_diagrams_ls(data: np.ndarray, negate: bool=False, wrap: bool=False,
                        max_dim: typing.Optional[int]=None, params: typing.Optional[ReductionParams]=None,
                        include_inf_points: bool=True, dualize: bool=False):
    if max_dim is None:
        max_dim = data.ndim - 1
    if params is None:
        params = _oineus.ReductionParams()
    # max_dim is maximal dimension of the _diagram_, we need simplices one dimension higher, hence +1
    fil = freudenthal_filtration(data=data, negate=negate, wrap=wrap, max_dim=max_dim + 1, n_threads=params.n_threads)
    dcmp = _oineus.Decomposition(fil, dualize)
    dcmp.reduce(params)
    return dcmp.diagram(fil=fil, include_inf_points=include_inf_points)


def compute_diagrams_vr(data: np.ndarray, from_pwdists: bool=False, max_dim: int=-1, max_diameter: float = -1.0, params: typing.Optional[ReductionParams]=None, include_inf_points: bool=True, dualize: bool=True):
    if params is None:
        params = _oineus.ReductionParams()
    # max_dim is maximal dimension of the _diagram_, we need simplices one dimension higher, hence +1
    fil = vr_filtration(data, from_pwdists, max_dim=max_dim, max_diameter=max_diameter, with_critical_edges=False, n_threads=params.n_threads)
    dcmp = _oineus.Decomposition(fil, dualize)
    dcmp.reduce(params)
    return dcmp.diagram(fil=fil, include_inf_points=include_inf_points)


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


def get_permutation(target_values, fil):
    if len(target_values) == 0:
        return {}
    func = getattr(_oineus, f"get_permutation")
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
    func = getattr(_oineus, f"compute_kernel_image_cokernel_reduction")
    return func(K_, L_, IdMap, n_threads)


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
