from __future__ import absolute_import

__version__ = "0.9.26"

import typing
import numpy as np
import scipy.sparse

from . import _oineus

from ._oineus import ConflictStrategy, DenoiseStrategy, VREdge
from ._oineus import DiagramPlaneDomain, FrechetMeanInit
from ._oineus import CombinatorialProdSimplex, CombinatorialSimplex,Simplex, ProdSimplex
from ._oineus import Filtration, ProdFiltration
from ._oineus import Decomposition, IndexDiagramPoint, DiagramPoint, Diagrams
from ._oineus import ReductionParams, KICRParams, KerImCokReduced, KerImCokReducedProd
from ._oineus import IndicesValues, IndicesValuesProd, TopologyOptimizer, TopologyOptimizerProd
from ._oineus import compute_relative_diagrams, get_boundary_matrix, get_denoise_target, get_induced_matching
from ._oineus import get_nth_persistence, get_permutation_dtv
from ._oineus import bottleneck_distance as _bottleneck_distance_cpp
from ._oineus import wasserstein_distance as _wasserstein_distance_cpp
from ._oineus import frechet_mean as _frechet_mean_cpp
from ._oineus import GridDomain_1D, Grid_1D, CombinatorialCube_1D, Cube_1D, CubeFiltration_1D
from ._oineus import GridDomain_2D, Grid_2D, CombinatorialCube_2D, Cube_2D, CubeFiltration_2D
from ._oineus import GridDomain_3D, Grid_3D, CombinatorialCube_3D, Cube_3D, CubeFiltration_3D
from .vis_utils import plot_persistence_diagram
# from ._oineus import Z2_Column, Z2_Matrix

try:
    import diode
    _HAS_DIODE = True
except:
    _HAS_DIODE = False


__all__ = ["compute_diagrams_ls", "compute_diagrams_vr", "compute_diagrams_alpha", "get_boundary_matrix", "is_reduced", "plot_persistence_diagram", "bottleneck_distance", "wasserstein_distance", "frechet_mean"]


def _diagram_from_oineus_diagrams(dgms, dim: typing.Optional[int]):
    if dim is None:
        if len(dgms) == 1:
            dim = 0
        else:
            raise ValueError("When passing oineus.Diagrams, specify dim=...")
    return np.ascontiguousarray(dgms.in_dimension(dim, as_numpy=True), dtype=np.float64)


def _normalize_diagram_for_distance(dgm, *, dim: typing.Optional[int]=None):
    if hasattr(dgm, "in_dimension") and hasattr(dgm, "__len__"):
        return _diagram_from_oineus_diagrams(dgm, dim)
    if isinstance(dgm, np.ndarray):
        dgm = np.asarray(dgm, dtype=np.float64)
        if dgm.ndim != 2 or dgm.shape[1] != 2:
            raise ValueError("Expected NumPy array with shape (n_points, 2)")
        return np.ascontiguousarray(dgm)
    return dgm


def _normalize_diagram_collection(diagrams):
    return [_normalize_diagram_for_distance(dgm) for dgm in diagrams]


def bottleneck_distance(dgm_1, dgm_2, delta: float=0.01, dim: typing.Optional[int]=None):
    """Compute the bottleneck distance between two persistence diagrams.

    Args:
        dgm_1: Persistence diagram as either `list[DiagramPoint]`, an Oineus
            `Diagrams` object, or a NumPy array with shape `(n_points, 2)`.
        dgm_2: Persistence diagram as either `list[DiagramPoint]`, an Oineus
            `Diagrams` object, or a NumPy array with shape `(n_points, 2)`.
        delta: Relative error requested from Hera. Set `delta=0.0` to request
            the exact bottleneck distance.
        dim: Homology dimension to extract when `dgm_1` or `dgm_2` is an
            Oineus `Diagrams` object. If omitted, dimension 0 is used only when
            the diagrams container has length 1.

    Returns:
        The bottleneck distance as a Python float.
    """
    return _bottleneck_distance_cpp(_normalize_diagram_for_distance(dgm_1, dim=dim),
                                    _normalize_diagram_for_distance(dgm_2, dim=dim),
                                    delta=delta)


def _diagram_arrays_equal_for_zero_check(dgm_1, dgm_2):
    if isinstance(dgm_1, np.ndarray):
        arr_1 = dgm_1
    else:
        arr_1 = np.array([[p[0], p[1]] for p in dgm_1], dtype=np.float64).reshape((-1, 2))

    if isinstance(dgm_2, np.ndarray):
        arr_2 = dgm_2
    else:
        arr_2 = np.array([[p[0], p[1]] for p in dgm_2], dtype=np.float64).reshape((-1, 2))

    if arr_1.shape != arr_2.shape:
        return False

    if arr_1.size == 0:
        return True

    sort_idx_1 = np.lexsort((arr_1[:, 1], arr_1[:, 0]))
    sort_idx_2 = np.lexsort((arr_2[:, 1], arr_2[:, 0]))
    arr_1 = arr_1[sort_idx_1]
    arr_2 = arr_2[sort_idx_2]

    finite_mask = np.isfinite(arr_1) & np.isfinite(arr_2)
    matching_inf_mask = np.isinf(arr_1) & np.isinf(arr_2) & (np.signbit(arr_1) == np.signbit(arr_2))

    if not np.all(finite_mask | matching_inf_mask):
        return False

    diff = np.zeros_like(arr_1)
    diff[finite_mask] = np.abs(arr_1[finite_mask] - arr_2[finite_mask])

    return np.all(diff < np.finfo(np.float64).eps)


def wasserstein_distance(dgm_1, dgm_2, q: float=2.0, delta: float=0.01, internal_p: float=np.inf,
                         wasserstein_q: typing.Optional[float]=None,
                         check_for_zero: bool=True,
                         dim: typing.Optional[int]=None):
    """Compute the q-Wasserstein distance between two persistence diagrams.

    Args:
        dgm_1: Persistence diagram as either `list[DiagramPoint]`, an Oineus
            `Diagrams` object, or a NumPy array with shape `(n_points, 2)`.
        dgm_2: Persistence diagram as either `list[DiagramPoint]`, an Oineus
            `Diagrams` object, or a NumPy array with shape `(n_points, 2)`.
        q: Wasserstein exponent.
        delta: Relative error requested from Hera.
        internal_p: Ground-metric norm in the plane. Use `np.inf` for the
            `L_infinity` norm.
        wasserstein_q: Alias for `q`, kept for API compatibility.
        check_for_zero: If `True`, do a quick exact equality check in Python and
            return `0.0` before calling Hera when the diagrams coincide.
        dim: Homology dimension to extract when `dgm_1` or `dgm_2` is an
            Oineus `Diagrams` object. If omitted, dimension 0 is used only when
            the diagrams container has length 1.

    Returns:
        The Wasserstein distance as a Python float.
    """
    dgm_1 = _normalize_diagram_for_distance(dgm_1, dim=dim)
    dgm_2 = _normalize_diagram_for_distance(dgm_2, dim=dim)
    if check_for_zero:
        if _diagram_arrays_equal_for_zero_check(dgm_1, dgm_2):
            return 0.0

    if wasserstein_q is not None:
        q = wasserstein_q
    if np.isinf(internal_p):
        internal_p = -1.0

    return _wasserstein_distance_cpp(dgm_1,
                                     dgm_2,
                                     q=q, delta=delta, internal_p=internal_p)


def frechet_mean(diagrams,
                 *,
                 weights=None,
                 max_iter: int = 100,
                 tol: float = 1e-7,
                 wasserstein_delta: float = 0.01,
                 internal_p: float = np.inf,
                 init_strategy=FrechetMeanInit.Grid,
                 domain=DiagramPlaneDomain.AboveDiagonal,
                 ignore_infinite_points: bool = False,
                 random_noise_scale: float = 1.0,
                 random_seed: int = 42,
                 grid_n_x_bins: int = 16,
                 grid_n_y_bins: int = 16,
                 custom_initial_barycenter=None):
    diagrams = _normalize_diagram_collection(diagrams)
    weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    if weights is not None:
        assert weights.ndim == 1, "weights must be a 1D array"
        assert weights.shape[0] == len(diagrams), "weights must have same length as diagrams"
    custom_initial_barycenter = None if custom_initial_barycenter is None else _normalize_diagram_for_distance(custom_initial_barycenter)

    if np.isinf(internal_p):
        internal_p = -1.0

    return _frechet_mean_cpp(diagrams,
                             weights=weights,
                             max_iter=max_iter,
                             tol=tol,
                             wasserstein_delta=wasserstein_delta,
                             internal_p=internal_p,
                             init_strategy=init_strategy,
                             domain=domain,
                             ignore_infinite_points=ignore_infinite_points,
                             random_noise_scale=random_noise_scale,
                             random_seed=random_seed,
                             grid_n_x_bins=grid_n_x_bins,
                             grid_n_y_bins=grid_n_y_bins,
                             custom_initial_barycenter=custom_initial_barycenter)


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
        v_codomain = v_codomain.combinatorial_cell
    if isinstance(v_domain, _oineus.Simplex) or isinstance(v_domain, _oineus.ProdSimplex):
        v_domain = v_domain.combinatorial_cell
    if with_indices:
        return _oineus._mapping_cylinder_with_indices(fil_domain, fil_codomain, v_domain, v_codomain)
    else:
        return _oineus._mapping_cylinder(fil_domain, fil_codomain, v_domain, v_codomain)

def multiply_filtration(fil, sigma):
    if isinstance(sigma, _oineus.Simplex):
        sigma = sigma.combinatorial_cell
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


def compute_diagrams_alpha(points: np.ndarray,
                           weights: typing.Optional[np.ndarray]=None,
                           params: typing.Optional[ReductionParams]=None,
                           include_inf_points: bool=True,
                           dualize: bool=False,
                           exact: bool=False,
                           periodic: bool=False,
                           compute_bounding_box: bool=True,
                           bbox_min: typing.Optional[typing.Union[np.ndarray, typing.List[float]]]=None,
                           bbox_max: typing.Optional[typing.Union[np.ndarray, typing.List[float]]]=None,
                           ):
    """Compute alpha-shape persistence diagrams.

    Args:
        points: NumPy array of shape (n, 2) or (n, 3).
        weights: Optional 1D array of length n. If provided, computes weighted
            alpha-shapes (currently 3D only).
        params: Reduction parameters. Defaults to ReductionParams().
        include_inf_points: Include points at infinity in output diagrams.
        dualize: If True, compute cohomology; otherwise homology.
        exact: Passed to diode. If True, uses exact CGAL kernel.
        periodic: If True, uses periodic alpha-shapes. Duplicate simplices
            reported by diode are deduplicated before building the filtration.
        compute_bounding_box: If True, use the bounding box of the data for periodic
            otherwise diode defaults (unit box) will be used.
        bbox_min: lexicographically smallest point of the bounding box.
             NumPy array or list of floats. Ignored, if compute_bounding_box is True.
             Origin will be used, if compute_bounding_box is False and bbox_min is None.
        bbox_max: lexicographically largest point of the bounding box.
             NumPy array or list of floats. Ignored, if compute_bounding_box is True

    Returns:
        Diagrams object indexed by homology dimension.
    """
    if params is None:
        params = _oineus.ReductionParams()
    assert _HAS_DIODE, "Cannot compute alpha-shapes without diode"
    assert points.ndim == 2
    assert points.shape[1] in [2, 3], "Alpha-shapes only support 2D and 3D point clouds"

    # Diode wants lists, we accept NumPy array for convenience
    if isinstance(bbox_min, np.ndarray):
        bbox_min = [ float(x) for x in bbox_min ]
    if isinstance(bbox_max, np.ndarray):
        bbox_max = [ float(x) for x in bbox_max ]

    if compute_bounding_box:
        bbox_min = [ float(np.min(points[:, d])) for d in range(points.shape[1]) ]
        bbox_max = [ float(np.max(points[:, d])) for d in range(points.shape[1]) ]
    else:
        if bbox_max is None:
            bbox_max = [ 1.0 for d in range(points.shape[1]) ]
        if  bbox_min is None:
            bbox_min = [ 0.0 for d in range(points.shape[1]) ]

    if weights is not None:
        weights = np.asarray(weights)
        assert weights.ndim == 1, "weights must be a 1D array"
        assert weights.shape[0] == points.shape[0], "weights must have same length as points"
        assert points.shape[1] == 3, "Weighted alpha-shapes require 3D points"

        weighted_points = np.column_stack((points, weights))

        if periodic:
            if not hasattr(diode, "fill_weighted_periodic_alpha_shapes"):
                raise RuntimeError("diode.fill_weighted_periodic_alpha_shapes is not available in this diode build")
            fil_diode = diode.fill_weighted_periodic_alpha_shapes(weighted_points, exact, bbox_min, bbox_max)
        else:
            fil_diode = diode.fill_weighted_alpha_shapes(weighted_points, exact=exact)
    else:
        if periodic:
            fil_diode = diode.fill_periodic_alpha_shapes(points, exact, bbox_min, bbox_max)
        else:
            fil_diode = diode.fill_alpha_shapes(points, exact=exact)

    fil = _oineus.Filtration(
        fil_diode,
        duplicates_possible=periodic,
        n_threads=params.n_threads
    )
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


def list_to_filtration(data: typing.List[typing.Tuple[int, typing.List[int], float]]):
    simplices = [ Simplex(id, vertices, val) for id, vertices, val in data ]
    return Filtration(simplices)


def compute_kernel_image_cokernel_reduction(K, L, params=None, reduction_params=None):
    # simplicial filtrations can be supplied as lists,
    # convert to Oineus filtrations if necessary
    if isinstance(K, list):
        K = list_to_filtration(K)
    if isinstance(L, list):
        L = list_to_filtration(L)

    # KICR class is templatized by cell type in C++
    # different instantiations have different class names in Python
    # figure out the right one by type
    if isinstance(K[0], _oineus.Simplex):
        KICR_Class = _oineus.KerImCokReduced
    elif isinstance(K[0], _oineus.ProdSimplex):
        KICR_Class = _oineus.KerImCokReducedProd
    else:
        raise TypeError(f"Unsupported filtration cell type: {type(K[0])}")

    if params is None:
        # compute all by default
        params = _oineus.KICRParams(kernel=True, image=True, cokernel=True)
    elif not isinstance(params, _oineus.KICRParams):
        raise TypeError("params must be a KICRParams instance")

    if reduction_params is not None:
        if not isinstance(reduction_params, _oineus.ReductionParams):
            raise TypeError("reduction_params must be a ReductionParams instance")
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


def cube_filtration(data: np.ndarray,
                    negate: bool=False,
                    wrap: bool=False,
                    max_dim: typing.Optional[int]=None,
                    values_on: str="vertices",
                    n_threads: int=1,):
    if wrap:
        raise RuntimeError("cube_filtration: wrap=True is not implemented yet")
    if max_dim is None:
        max_dim = data.ndim
    dim = data.ndim
    if dim == 1:
        grid = _oineus.Grid_1D(data, wrap=wrap, values_on=values_on)
    elif dim == 2:
        grid = _oineus.Grid_2D(data, wrap=wrap, values_on=values_on)
    elif dim == 3:
        grid = _oineus.Grid_3D(data, wrap=wrap, values_on=values_on)
    else:
        raise RuntimeError(f"cube_filtration: dim={data.ndim} not supported, recompile from sources")
    fil = grid.cube_filtration(max_dim=max_dim, n_threads=n_threads, negate=negate)
    return fil
