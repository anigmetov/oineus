from __future__ import absolute_import

import typing
import numpy as np
import scipy.sparse

from . import _oineus

from ._oineus import ConflictStrategy, DenoiseStrategy, VREdge
from ._oineus import CombinatorialProdSimplex, CombinatorialSimplex,Simplex, ProdSimplex
from ._oineus import Filtration, ProdFiltration
from ._oineus import Decomposition, IndexDiagramPoint, DiagramPoint, Diagrams
from ._oineus import ReductionParams, KICRParams, KerImCokReduced, KerImCokReducedProd
from ._oineus import IndicesValues, IndicesValuesProd, TopologyOptimizer, TopologyOptimizerProd
from ._oineus import compute_relative_diagrams, get_boundary_matrix, get_denoise_target, get_induced_matching
from ._oineus import get_nth_persistence, get_permutation_dtv, list_to_filtration

# import warnings
#
# try:
#     from . import diff
# except:
#     warnings.warn("oineus.diff import failed, probably, because eagerpy is not installed")
#

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
        fil, vertices = _oineus.get_freudenthal_filtration_and_crit_vertices(data=data, negate=negate, wrap=wrap, max_dim=max_dim, n_threads=n_threads)
        vertices = np.array(vertices, dtype=np.int64)
        return fil, vertices
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

    result = func(data, max_dim=max_dim, max_diameter=max_diameter, n_threads=n_threads)
    if with_critical_edges:
        # convert list of VREdges to numpy array
        edges = [ [ e.x, e.y] for e in result[1] ]
        edges = np.array(edges, dtype=np.int64)
        return result[0], edges
    else:
        return result


def is_reduced(a):
    lowest_ones = []
    for col_idx in range(a.shape[1]):
        if np.any(a[:, col_idx] % 2 == 1):
            lowest_ones.append(np.max(np.where(a[:, col_idx] % 2 == 1)))
    return len(lowest_ones) == len(set(lowest_ones))

def mapping_cylinder(fil_domain, fil_codomain, v_domain, v_codomain, with_indices=False):
    if isinstance(v_codomain, _oineus.Simplex) or isinstance(v_codomain, _oineus.ProdSimplex):
        v_codomain = v_codomain.combinatorial_cell()
    if isinstance(v_domain, _oineus.Simplex) or isinstance(v_domain, _oineus.ProdSimplex):
        v_domain = v_domain.combinatorial_cell()
    if with_indices:
        return _oineus._mapping_cylinder_with_indices(fil_domain, fil_codomain, v_domain, v_codomain)
    else:
        return _oineus._mapping_cylinder(fil_domain, fil_codomain, v_domain, v_codomain)

def multiply_filtration(fil, sigma):
    if isinstance(sigma, _oineus.Simplex):
        sigma = sigma.combinatorial_cell()
    return _oineus._multiply_filtration(fil, sigma)

def min_filtration(fil_1, fil_2, with_indices=False):
    if with_indices:
        return _oineus._min_filtration_with_indices(fil_1, fil_2)
    else:
        return _oineus._min_filtration(fil_1, fil_2)

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
    func = getattr(_oineus, f"get_ls_wasserstein_matching_target_values")

    if type(dgm) is np.ndarray:
        dgm_1 = []
        assert len(dgm.shape) == 2 and dgm.shape[1] == 2
        for p in dgm:
            dgm_1.append(DiagramPoint(p[0], p[1]))
        dgm = dgm_1

    return func(dgm, fil, rv, d, q, mip, mdp)


def compute_kernel_image_cokernel_reduction(K, L, params=None, reduction_params=None):
    # simplicial filtrations can be supplied as lists,
    # convert to Oineus filtrations if necessary
    if isinstance(K, list):
        K = _oineus.list_to_filtration(K)
    if isinstance(L, list):
        L = _oineus.list_to_filtration(L)

    # KICR class is templatized by cell type in C++
    # different instantiations have different class names in Python
    # figure out the right one by type
    if isinstance(K[0], _oineus.Simplex):
        KICR_Class = _oineus.KerImCokReduced
    elif isinstance(K[0], _oineus.ProdSimplex):
        KICR_Class = _oineus.KerImCokReducedProd

    if params is None:
        # compute all by default
        params = _oineus.KICRParams()
        params.kernel = True
        params.cokernel = True
        params.image = True

    if reduction_params != None:
        assert type(params) == _oineus.ReductionParams
        params.params_f = reduction_params
        params.params_g = reduction_params
        params.params_ker = reduction_params
        params.params_im = reduction_params
        params.params_cok = reduction_params

    return KICR_Class(K, L, params)


def compute_ker_cok_reduction_cyl(fil_2, fil_3):
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

    params = _oineus.KICRParams()
    params.kernel = params.cokernel = True
    params.image = False

    kicr_reduction = _oineus.KerImCokReducedProd(fil_cyl, fil_3_prod, params)

    return kicr_reduction