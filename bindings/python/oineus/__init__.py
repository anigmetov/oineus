from __future__ import absolute_import

__version__ = "0.9.31"

import typing
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.sparse

from . import _oineus

from ._oineus import ConflictStrategy, DenoiseStrategy, VREdge, FiltrationKind
from ._oineus import DiagramPlaneDomain, FrechetMeanInit
from ._oineus import CombinatorialProdSimplex, CombinatorialSimplex,Simplex, ProdSimplex
from ._oineus import Decomposition, IndexDiagramPoint
# Diagrams / DiagramPoint are Real-templated (a float32 build registers distinct classes under
# _f32), so the bare top-module class would fail isinstance on a float32 diagram. They are
# re-exposed below as cross-backend markers, mirroring Filtration / ProdFiltration.
# IndexDiagramPoint is Real-independent (indices are ints), so the single shared class is fine.
from ._oineus import reduce
from ._oineus import DecompositionManipStats
from ._oineus import ReductionParams, ReductionTimings, KICRParams, KerImCokReduced, KerImCokReducedProd
from ._oineus import ColumnRepr
from ._oineus import IndicesValues, IndicesValuesProd, TopologyOptimizerProd
from ._oineus import TopologyOptimizerCube_1D, TopologyOptimizerCube_2D, TopologyOptimizerCube_3D, TopologyOptimizerCube_4D
from ._oineus import get_boundary_matrix
# compute_relative_diagrams / get_denoise_target / get_induced_matching / get_nth_persistence /
# get_permutation(_dtv) are dtype- and cell-type-routed Python wrappers defined below: the C++
# overloads live per-Real (top module for float64, _f32 for float32) and per cell type, so the
# bare top-module symbol would reject the now-default packed/slim and any float32 filtration.
from ._oineus import bottleneck_distance as _bottleneck_distance_cpp
from ._oineus import wasserstein_distance as _wasserstein_distance_cpp
from ._oineus import init_frechet_mean_first_diagram as _init_frechet_mean_first_diagram_cpp
from ._oineus import init_frechet_mean_random_diagram as _init_frechet_mean_random_diagram_cpp
from ._oineus import init_frechet_mean_medoid_diagram as _init_frechet_mean_medoid_diagram_cpp
from ._oineus import init_frechet_mean_diagonal_grid as _init_frechet_mean_diagonal_grid_cpp
from ._oineus import frechet_mean as _frechet_mean_cpp
from ._oineus import GridDomain_1D, Grid_1D, CombinatorialCube_1D, Cube_1D
from ._oineus import GridDomain_2D, Grid_2D, CombinatorialCube_2D, Cube_2D
from ._oineus import GridDomain_3D, Grid_3D, CombinatorialCube_3D, Cube_3D
from ._oineus import GridDomain_4D, Grid_4D, CombinatorialCube_4D, Cube_4D

from ._dtype import (REAL_DTYPE, DEFAULT_REAL_DTYPE, REAL_MODULES, as_real_numpy,
                     detect_real_dtype, real_module_for, module_of_oineus_obj)


def _merge_over_reals(per_module):
    """Merge a per-submodule dict over every compiled Real (float64 = the top
    module, float32 = _f32 when present). Class names are identical across
    submodules, but a float32 _Filtration is a distinct Python class from the
    float64 one, so the merged map dispatches both dtypes from one lookup."""
    merged = {}
    for sub in REAL_MODULES.values():
        merged.update(per_module(sub))
    return merged


# Maps the fat cell-with-value type a user hands to Filtration(...) to the concrete C++
# filtration class that consumes it. The per-encoding internal filtrations are distinct C++
# types (cube vs simplex vs product), but the user sees one Filtration.
_FIL_CLASS_BY_CELL_TYPE = _merge_over_reals(lambda s: {
    s.Simplex:      s._Filtration,        # fat Simplex (VR / alpha / user-built)
    s.ProdSimplex:  s._ProdFiltration,    # product cells (mapping cylinders)
    s.Cube_1D:      s._CubeFiltration_1D,  # fat cubes (hand-built cubical complexes)
    s.Cube_2D:      s._CubeFiltration_2D,
    s.Cube_3D:      s._CubeFiltration_3D,
    s.Cube_4D:      s._CubeFiltration_4D,
})

# Every concrete filtration C++ type (both dtypes), for isinstance(x, oineus.Filtration).
# Includes the factory-produced slim Freudenthal / bit-packed ones, which a user never
# constructs by hand but should still recognize as filtrations.
_ALL_FILTRATION_TYPES = tuple(
    f for s in REAL_MODULES.values() for f in (
        s._Filtration, s._ProdFiltration,
        s._CubeFiltration_1D, s._CubeFiltration_2D, s._CubeFiltration_3D, s._CubeFiltration_4D,
        s._FreudenthalFiltration_1D, s._FreudenthalFiltration_2D, s._FreudenthalFiltration_3D, s._FreudenthalFiltration_4D,
        s._PackedSimplexFiltration_64, s._PackedSimplexFiltration_128,
    ))


class _FiltrationMeta(type):
    # isinstance(x, oineus.Filtration) is True for any concrete filtration the library builds,
    # even though Filtration() returns the concrete C++ object (not a _FiltrationMeta instance).
    def __instancecheck__(cls, obj):
        return isinstance(obj, _ALL_FILTRATION_TYPES)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, _ALL_FILTRATION_TYPES)


class Filtration(metaclass=_FiltrationMeta):
    """A filtration: an ordered list of cells, each with a filtration value.

    Construct one from a list of fat cells with values, dispatching on the cell type::

        oineus.Filtration([oineus.Simplex([0], 0.0), oineus.Simplex([0, 1], 1.0), ...])  # simplicial
        oineus.Filtration([oineus.Cube_2D(...), ...])                                     # cubical
        oineus.Filtration([oineus.ProdSimplex(...), ...])                                 # product cells

    For the common constructions use the factory functions instead, which build the cells for
    you (and pick an efficient internal cell encoding): vr_filtration / alpha_filtration for
    point clouds, freudenthal_filtration / cube_filtration for functions on grids.

    isinstance(x, oineus.Filtration) is True for any filtration the library produces, including
    the factory-built ones whose concrete C++ type is an internal detail.
    """

    def __new__(cls, cells, *args, **kwargs):
        try:
            first = cells[0]
        except IndexError:
            # empty list -> the universal fat Simplex filtration (historical default; there is
            # no cell to dispatch on)
            return _oineus._Filtration(cells, *args, **kwargs)
        except TypeError:
            raise ValueError(
                "Filtration(cells): cells must be a list of fat cells with values "
                "(Simplex / Cube_1D/2D/3D / ProdSimplex). For point clouds use vr_filtration or "
                "alpha_filtration; for functions on grids use freudenthal_filtration or "
                "cube_filtration.")
        fil_cls = _FIL_CLASS_BY_CELL_TYPE.get(type(first))
        if fil_cls is None:
            if isinstance(first, tuple):
                # (vertices, value) pairs -> the universal simplicial constructor, which builds
                # the Simplex cells itself (used by the diode alpha / Cech-Delaunay paths)
                return _oineus._Filtration(cells, *args, **kwargs)
            raise TypeError(
                f"Filtration(cells): unsupported cell type {type(first).__name__}; expected one "
                f"of {[t.__name__ for t in _FIL_CLASS_BY_CELL_TYPE]} or (vertices, value) tuples.")
        return fil_cls(cells, *args, **kwargs)


# Product-cell filtrations: those whose cells are oineus.ProdSimplex (ProductCell), i.e. the
# output of mapping_cylinder / multiply_filtration and of Filtration([ProdSimplex, ...]). Both
# Real backends register the concrete C++ type under the private name _ProdFiltration.
_PROD_FILTRATION_TYPES = tuple(s._ProdFiltration for s in REAL_MODULES.values())


class _ProdFiltrationMeta(type):
    # isinstance(x, oineus.ProdFiltration) is True for any product-cell filtration, in either
    # Real backend. ProdFiltration is a marker only -- it never instantiates (see __new__).
    def __instancecheck__(cls, obj):
        return isinstance(obj, _PROD_FILTRATION_TYPES)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, _PROD_FILTRATION_TYPES)


class ProdFiltration(metaclass=_ProdFiltrationMeta):
    """Marker for product-cell filtrations -- those whose cells are oineus.ProdSimplex
    (ProductCell), e.g. the result of mapping_cylinder or multiply_filtration.

    Use it for membership tests only::

        isinstance(fil, oineus.ProdFiltration)   # True for any product-cell filtration

    It is a cell-type marker, distinct from the filtration's FiltrationKind (which records how
    the filtration was built). It is NOT a constructor; build a product filtration through the
    unified facade::

        oineus.Filtration([oineus.ProdSimplex([0], [0], 0.0), ...])
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "oineus.ProdFiltration is an isinstance marker, not a constructor. Build a product "
            "filtration with oineus.Filtration([oineus.ProdSimplex(...), ...]).")


# Diagrams / DiagramPoint are Real-templated: a float32 build registers distinct classes under
# _f32, so the bare top-module class fails isinstance on a float32 diagram. These markers make
# isinstance(x, oineus.Diagrams) / oineus.DiagramPoint True in either backend, mirroring
# Filtration / ProdFiltration. Construction defaults to the float64 concrete class (birth/death
# and the dimension count are plain Python numbers, with no input dtype to route on).
_ALL_DIAGRAMS_TYPES = tuple(s.Diagrams for s in REAL_MODULES.values())
_ALL_DIAGRAM_POINT_TYPES = tuple(s.DiagramPoint for s in REAL_MODULES.values())


class _DiagramsMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _ALL_DIAGRAMS_TYPES)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, _ALL_DIAGRAMS_TYPES)


class Diagrams(metaclass=_DiagramsMeta):
    """Persistence diagrams indexed by homology dimension.

    isinstance(x, oineus.Diagrams) is True for a diagram from either Real backend (float64 or
    float32). Extract one dimension with ``dgm.in_dimension(d)`` (NumPy ``(n, 2)`` array) or
    ``dgm.in_dimension(d, as_numpy=False)`` (list of DiagramPoint). Constructing
    ``oineus.Diagrams(max_dim)`` returns a float64 diagram container.
    """

    def __new__(cls, *args, **kwargs):
        return _oineus.Diagrams(*args, **kwargs)


class _DiagramPointMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _ALL_DIAGRAM_POINT_TYPES)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, _ALL_DIAGRAM_POINT_TYPES)


class DiagramPoint(metaclass=_DiagramPointMeta):
    """A single persistence-diagram point with ``birth`` and ``death`` attributes.

    isinstance(x, oineus.DiagramPoint) is True for a point from either Real backend.
    Constructing ``oineus.DiagramPoint(birth, death)`` returns a float64 point.
    """

    def __new__(cls, *args, **kwargs):
        return _oineus.DiagramPoint(*args, **kwargs)


# Visualization helpers require matplotlib, an optional extra
# (`pip install oineus[vis]`). When it is absent, the plot_* helpers and
# style constants are simply unavailable; the rest of oineus works normally.
try:
    from .vis import (
        plot_diagram,
        plot_diagram_gradient,
        plot_matching,
        plot_chain,
        default_point_style,
        default_diagram_a_point_style,
        default_diagram_b_point_style,
        default_matching_edge_style,
        default_longest_edge_style,
        default_diagonal_style,
        default_diagonal_projection_a_style,
        default_diagonal_projection_b_style,
        default_inf_line_style,
        default_inf_point_style,
        default_diagram_gradient_style,
        default_density_style,
        default_grid_style,
        default_chain_vertex_style,
        default_chain_edge_style,
        default_chain_triangle_style,
        default_chain_tetrahedron_style,
        default_point_cloud_style,
        DEFAULT_POINT_STYLE,
        DEFAULT_DIAGRAM_A_POINT_STYLE,
        DEFAULT_DIAGRAM_B_POINT_STYLE,
        DEFAULT_MATCHING_EDGE_STYLE,
        DEFAULT_LONGEST_EDGE_STYLE,
        DEFAULT_DIAGONAL_STYLE,
        DEFAULT_DIAGONAL_PROJECTION_A_STYLE,
        DEFAULT_DIAGONAL_PROJECTION_B_STYLE,
        DEFAULT_INF_LINE_STYLE,
        DEFAULT_INF_POINT_STYLE,
        DEFAULT_DIAGRAM_GRADIENT_STYLE,
        DEFAULT_DENSITY_STYLE,
        DEFAULT_DENSITY_THRESHOLD,
        DEFAULT_GRID_STYLE,
        DEFAULT_MATCHING_EDGE_QUANTILE,
        DEFAULT_GRADIENT_TOP_K_ARROWS,
        DEFAULT_CHAIN_VERTEX_STYLE,
        DEFAULT_CHAIN_EDGE_STYLE,
        DEFAULT_CHAIN_TRIANGLE_STYLE,
        DEFAULT_CHAIN_TETRAHEDRON_STYLE,
        DEFAULT_POINT_CLOUD_STYLE,
        OKABE_ITO_BLUE,
        OKABE_ITO_VERMILLION,
    )
    # Keep vis_utils as a backward-compat alias.
    from . import vis_utils  # noqa: F401
except ImportError:
    pass
from .matching import (
    DiagramMatching,
    BottleneckMatching,
    InfKind,
    EssentialMatches,
    EssentialLongestEdges,
    LongestEdges,
    FiniteLongestEdge,
    EssentialLongestEdge,
    point_to_diagonal,
    wasserstein_matching,
    bottleneck_matching,
)
from .sliced_wasserstein import (
    sliced_wasserstein_distance,
    sliced_wasserstein_distance_diag_corrected,
)
# from ._oineus import Z2_Column, Z2_Matrix

try:
    import diode
    _HAS_DIODE = True
except:
    _HAS_DIODE = False

# Newer diode builds add the structured-array exporters (combinatorics/values as
# NumPy arrays instead of one Python tuple per simplex). Probed once at import;
# when False, the code falls back to the list-of-(vertices, value) API.
_HAS_DIODE_ARRAYS = _HAS_DIODE and hasattr(diode, "fill_delaunay_arrays") \
    and hasattr(diode, "fill_alpha_shapes_arrays")


# Maps each filtration cell encoding to its C++ TopologyOptimizer instantiation. The
# reduction core is cell-agnostic, but the optimizer is templated on the cell type, so
# there is one bound class per encoding (universal Simplex, product, slim cube, slim
# Freudenthal, bit-packed VR/alpha). Single source of truth: oineus.diff reuses it.
_OPT_CLASS_BY_FIL_TYPE = _merge_over_reals(lambda s: {
    s._Filtration:               s.TopologyOptimizer,
    s._ProdFiltration:           s.TopologyOptimizerProd,
    s._CubeFiltration_1D:        s.TopologyOptimizerCube_1D,
    s._CubeFiltration_2D:        s.TopologyOptimizerCube_2D,
    s._CubeFiltration_3D:        s.TopologyOptimizerCube_3D,
    s._CubeFiltration_4D:        s.TopologyOptimizerCube_4D,
    s._FreudenthalFiltration_1D: s.TopologyOptimizerFreudenthal_1D,
    s._FreudenthalFiltration_2D: s.TopologyOptimizerFreudenthal_2D,
    s._FreudenthalFiltration_3D: s.TopologyOptimizerFreudenthal_3D,
    s._FreudenthalFiltration_4D: s.TopologyOptimizerFreudenthal_4D,
    s._PackedSimplexFiltration_64:  s.TopologyOptimizerPacked_64,
    s._PackedSimplexFiltration_128: s.TopologyOptimizerPacked_128,
})


# Maps each filtration cell encoding to its C++ KerImCokReduced (kernel/image/cokernel)
# instantiation. kernel.h is cell-agnostic, so KICR is wired for every encoding; the two
# fat classes keep their public names, the rest are hidden underscore names. Keyed by the
# filtration type (type(K)) -- NOT by K[0], whose materialized fat cell would misdispatch
# slim/packed filtrations into the fat ctor.
_KICR_CLASS_BY_FIL_TYPE = _merge_over_reals(lambda s: {
    s._Filtration:               s.KerImCokReduced,
    s._ProdFiltration:           s.KerImCokReducedProd,
    s._CubeFiltration_1D:        s._KerImCokReduced_Cube_1D,
    s._CubeFiltration_2D:        s._KerImCokReduced_Cube_2D,
    s._CubeFiltration_3D:        s._KerImCokReduced_Cube_3D,
    s._CubeFiltration_4D:        s._KerImCokReduced_Cube_4D,
    s._FreudenthalFiltration_1D: s._KerImCokReduced_Fr_1D,
    s._FreudenthalFiltration_2D: s._KerImCokReduced_Fr_2D,
    s._FreudenthalFiltration_3D: s._KerImCokReduced_Fr_3D,
    s._FreudenthalFiltration_4D: s._KerImCokReduced_Fr_4D,
    s._PackedSimplexFiltration_64:  s._KerImCokReduced_Packed_64,
    s._PackedSimplexFiltration_128: s._KerImCokReduced_Packed_128,
})


class TopologyOptimizer:
    """Topology optimizer for a filtration of any cell encoding.

    Dispatches on the filtration's cell type -- universal Simplex (VR / alpha / user),
    product, slim cube, slim Freudenthal, or bit-packed VR/alpha -- and returns the
    matching C++ optimizer instance directly, so its full native API (reduce_all,
    compute_diagram, simplify, match, singletons, combine_loss, ...) is available
    unchanged. Constructor keywords (with_crit_sets, dims_to_restore_elz, n_threads,
    u_strategy) are forwarded verbatim.

    oineus.diff.TopologyOptimizer is the differentiable-pipeline wrapper built on the
    same dispatch; use this bare class for direct (non-autograd) topology optimization.
    """

    def __new__(cls, fil, *args, **kwargs):
        opt_cls = _OPT_CLASS_BY_FIL_TYPE.get(type(fil))
        if opt_cls is None:
            raise TypeError(
                f"TopologyOptimizer: unsupported filtration type "
                f"{type(fil).__name__}; expected one of "
                f"{[t.__name__ for t in _OPT_CLASS_BY_FIL_TYPE]}"
            )
        return opt_cls(fil, *args, **kwargs)


def _check_numpy_diagram_shape(dgm):
    """If ``dgm`` is a numpy array, assert shape (n, 2). Otherwise pass through."""
    if isinstance(dgm, np.ndarray) and (dgm.ndim != 2 or dgm.shape[1] != 2):
        raise ValueError("Expected NumPy array with shape (n_points, 2)")
    return dgm


def _normalize_frechet_weights(n_diagrams: int, weights):
    if n_diagrams == 0:
        return np.empty((0,), dtype=REAL_DTYPE)

    if weights is None:
        return np.full(n_diagrams, 1.0 / n_diagrams, dtype=REAL_DTYPE)

    arr = np.asarray(weights)
    if arr.ndim != 1:
        raise ValueError("weights must be a 1D array")
    if arr.shape[0] != n_diagrams:
        raise ValueError("weights must have same length as diagrams")
    if np.any(arr < 0.0):
        raise ValueError("weights must be nonnegative")

    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")

    return arr / total


def _diagram_persistences(dgm: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return np.empty((0,), dtype=dgm.dtype)

    pers = np.empty(dgm.shape[0], dtype=dgm.dtype)
    finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
    pers[finite_mask] = np.abs(dgm[finite_mask, 1] - dgm[finite_mask, 0])
    pers[~finite_mask] = np.inf
    return pers


def _threshold_diagram_by_persistence(dgm: np.ndarray, min_persistence: float, *, include_infinite_points: bool = True):
    if dgm.size == 0:
        return dgm.reshape((0, 2))

    pers = _diagram_persistences(dgm)
    finite_mask = np.isfinite(pers)
    keep_mask = np.zeros(dgm.shape[0], dtype=bool)
    keep_mask[finite_mask] = pers[finite_mask] >= min_persistence
    if include_infinite_points:
        keep_mask |= ~finite_mask
    return np.ascontiguousarray(dgm[keep_mask])


def _newly_active_diagram_points(dgm: np.ndarray, previous_threshold: float, current_threshold: float):
    if dgm.size == 0:
        return dgm.reshape((0, 2))

    pers = _diagram_persistences(dgm)
    finite_mask = np.isfinite(pers)
    keep_mask = finite_mask & (pers >= current_threshold) & (pers < previous_threshold)
    return np.ascontiguousarray(dgm[keep_mask])


def _diagrams_to_numpy_list(diagrams):
    """Convert each diagram in the sequence to a Real-dtype ``(n, 2)`` numpy array.

    Used by helpers that manipulate diagram points in Python (thresholding,
    persistence scheduling, pairwise distances, ...). For direct pass-through
    to the C++ Hera bindings, prefer :func:`as_real_numpy` instead — nanobind
    picks the correct overload based on input type.
    """
    result = []
    for dgm in diagrams:
        if isinstance(dgm, np.ndarray):
            result.append(as_real_numpy(_check_numpy_diagram_shape(dgm)))
        elif hasattr(dgm, "in_dimension"):  # oineus.Diagrams
            if len(dgm) != 1:
                raise ValueError(
                    "Cannot convert multi-dimensional oineus.Diagrams: specify dim=... "
                    "or extract .in_dimension(d) before calling"
                )
            result.append(dgm.in_dimension(0, as_numpy=True))
        else:  # list of DiagramPoint or similar
            if len(dgm) == 0:
                result.append(np.empty((0, 2), dtype=REAL_DTYPE))
            else:
                result.append(
                    np.array([[p[0], p[1]] for p in dgm], dtype=REAL_DTYPE).reshape((-1, 2))
                )
    return result


def _resolve_multistart_seed(diagrams,
                             seed,
                             *,
                             weights,
                             domain,
                             random_noise_scale,
                             random_seed,
                             grid_n_x_bins,
                             grid_n_y_bins,
                             wasserstein_delta,
                             internal_p,
                             n_threads: int = 1):
    if not isinstance(seed, str):
        return as_real_numpy(_check_numpy_diagram_shape(seed))

    if seed == "first":
        return init_frechet_mean_first_diagram(diagrams)
    if seed == "medoid":
        return init_frechet_mean_medoid_diagram(diagrams, weights=weights, n_threads=n_threads)
    if seed == "grid":
        return init_frechet_mean_diagonal_grid(
            diagrams,
            weights=weights,
            domain=domain,
            grid_n_x_bins=grid_n_x_bins,
            grid_n_y_bins=grid_n_y_bins,
        )
    if seed == "random":
        return init_frechet_mean_random_diagram(
            diagrams,
            domain=domain,
            random_noise_scale=random_noise_scale,
            random_seed=random_seed,
        )

    # Need numpy arrays for .copy() in the "farthest_from_medoid"/"second_medoid" paths.
    diagrams = _diagrams_to_numpy_list(diagrams)
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)
    n = len(diagrams)
    d2 = np.zeros((n, n), dtype=REAL_DTYPE)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def pair_dist(ij):
        i, j = ij
        return wasserstein_distance(
            diagrams[i], diagrams[j],
            q=2.0, delta=wasserstein_delta, internal_p=internal_p,
        )

    if n_threads <= 1 or len(pairs) <= 1:
        dists = [pair_dist(p) for p in pairs]
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            dists = list(executor.map(pair_dist, pairs))
    for (i, j), dist in zip(pairs, dists):
        d2[i, j] = dist * dist
        d2[j, i] = d2[i, j]

    medoid_idx = int(np.argmin(d2 @ normalized_weights))

    if seed == "farthest_from_medoid":
        return diagrams[int(np.argmax(d2[:, medoid_idx]))].copy()

    if seed == "second_medoid":
        provisional = int(np.argmax(d2[:, medoid_idx]))
        c2 = np.where(d2[:, provisional] < d2[:, medoid_idx])[0]
        if c2.size == 0:
            return diagrams[provisional].copy()
        restricted_scores = np.array([
            np.sum(normalized_weights[c2] * d2[idx, c2])
            for idx in c2
        ])
        return diagrams[int(c2[int(np.argmin(restricted_scores))])].copy()

    raise ValueError(f"Unknown Fréchet-mean seed '{seed}'")


def _prepare_distance_args(dgm_1, dgm_2):
    """Coerce two single-dimension diagram inputs for the C++ distance entry.

    Both inputs must be a numpy ``(n, 2)`` array or a ``list[DiagramPoint]``.
    A multi-dimensional ``oineus.Diagrams`` is rejected — the caller must
    extract a single dimension first via ``dgm.in_dimension(d)``.
    """
    def coerce(dgm):
        if isinstance(dgm, Diagrams):
            raise TypeError(
                "Pass a single-dimension diagram: use `dgm.in_dimension(d)` "
                "instead of passing the multi-dimensional Diagrams object.")
        return as_real_numpy(_check_numpy_diagram_shape(dgm))
    return (coerce(dgm_1), coerce(dgm_2))


def bottleneck_distance(dgm_1, dgm_2, delta: float=0.01):
    """Compute the bottleneck distance between two persistence diagrams.

    Args:
        dgm_1: Single-dimension persistence diagram: a NumPy array of shape
            ``(n_points, 2)`` or a ``list[DiagramPoint]``. To pass an Oineus
            ``Diagrams`` object, extract the dimension first via
            ``dgm.in_dimension(d)``.
        dgm_2: Same forms as ``dgm_1``.
        delta: Relative error requested from Hera. Set `delta=0.0` to request
            the exact bottleneck distance.

    Returns:
        The bottleneck distance as a Python float.
    """
    return _bottleneck_distance_cpp(*_prepare_distance_args(dgm_1, dgm_2), delta=delta)


def _diagram_arrays_equal_for_zero_check(dgm_1, dgm_2):
    if isinstance(dgm_1, np.ndarray):
        arr_1 = dgm_1
    else:
        arr_1 = np.array([[p[0], p[1]] for p in dgm_1], dtype=REAL_DTYPE).reshape((-1, 2))

    if isinstance(dgm_2, np.ndarray):
        arr_2 = dgm_2
    else:
        arr_2 = np.array([[p[0], p[1]] for p in dgm_2], dtype=REAL_DTYPE).reshape((-1, 2))

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

    return np.all(diff < np.finfo(arr_1.dtype).eps)


def wasserstein_distance(dgm_1, dgm_2, q: float=1.0, delta: float=0.01, internal_p: float=np.inf,
                         wasserstein_q: typing.Optional[float]=None,
                         check_for_zero: bool=True):
    """Compute the q-Wasserstein distance between two persistence diagrams.

    Args:
        dgm_1: Single-dimension persistence diagram: a NumPy array of shape
            ``(n_points, 2)`` or a ``list[DiagramPoint]``. To pass an Oineus
            ``Diagrams`` object, extract the dimension first via
            ``dgm.in_dimension(d)``.
        dgm_2: Same forms as ``dgm_1``.
        q: Wasserstein exponent.
        delta: Relative error requested from Hera.
        internal_p: Ground-metric norm in the plane. Use `np.inf` for the
            `L_infinity` norm.
        wasserstein_q: Alias for `q`, kept for API compatibility.
        check_for_zero: If `True`, skip Hera when the two inputs are numpy
            arrays of equal points.

    Returns:
        The Wasserstein distance as a Python float.
    """
    if wasserstein_q is not None:
        q = wasserstein_q
    if np.isinf(internal_p):
        internal_p = -1.0

    prepared = _prepare_distance_args(dgm_1, dgm_2)

    if check_for_zero and _diagram_arrays_equal_for_zero_check(*prepared):
        return 0.0

    return _wasserstein_distance_cpp(*prepared, q=q, delta=delta, internal_p=internal_p)


def init_frechet_mean_first_diagram(diagrams):
    return _init_frechet_mean_first_diagram_cpp([as_real_numpy(d) for d in diagrams])


def init_frechet_mean_random_diagram(diagrams,
                                     *,
                                     domain=DiagramPlaneDomain.AboveDiagonal,
                                     random_noise_scale: float = 1.0,
                                     random_seed: int = 42):
    return _init_frechet_mean_random_diagram_cpp(
        [as_real_numpy(d) for d in diagrams],
        domain=domain,
        random_noise_scale=random_noise_scale,
        random_seed=random_seed,
    )


def init_frechet_mean_medoid_diagram(diagrams, *, weights=None, n_threads: int = 1):
    return _init_frechet_mean_medoid_diagram_cpp(
        [as_real_numpy(d) for d in diagrams], weights=weights, n_threads=n_threads
    )


def init_frechet_mean_diagonal_grid(diagrams,
                                    *,
                                    weights=None,
                                    domain=DiagramPlaneDomain.AboveDiagonal,
                                    grid_n_x_bins: int = 16,
                                    grid_n_y_bins: int = 16):
    return _init_frechet_mean_diagonal_grid_cpp(
        [as_real_numpy(d) for d in diagrams],
        weights=weights,
        domain=domain,
        grid_n_x_bins=grid_n_x_bins,
        grid_n_y_bins=grid_n_y_bins,
    )


def frechet_mean_objective(diagrams,
                           barycenter,
                           *,
                           weights=None,
                           wasserstein_delta: float = 0.01,
                           internal_p: float = np.inf,
                           n_threads: int = 1):
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)

    def term(i_diagram):
        i, diagram = i_diagram
        return normalized_weights[i] * wasserstein_distance(
            barycenter,
            diagram,
            q=2.0,
            delta=wasserstein_delta,
            internal_p=internal_p,
        ) ** 2

    if n_threads <= 1 or len(diagrams) <= 1:
        return float(sum(term((i, d)) for i, d in enumerate(diagrams)))

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        terms = list(executor.map(term, enumerate(diagrams)))
    return float(sum(terms))


def make_frechet_mean_persistence_schedule(diagrams,
                                           *,
                                           initial_threshold_fraction: float = 0.5,
                                           max_active_growth: float = 0.10,
                                           min_persistence: float = 0.0):
    diagrams = _diagrams_to_numpy_list(diagrams)
    if initial_threshold_fraction <= 0.0:
        raise ValueError("initial_threshold_fraction must be positive")
    if max_active_growth < 0.0:
        raise ValueError("max_active_growth must be nonnegative")

    finite_persistences = []
    for dgm in diagrams:
        pers = _diagram_persistences(dgm)
        finite_persistences.extend(pers[np.isfinite(pers)].tolist())

    if not finite_persistences:
        return [float(min_persistence)]

    values = np.array(sorted(set(float(p) for p in finite_persistences if p >= min_persistence), reverse=True))
    if values.size == 0:
        return [float(min_persistence)]

    start_target = max(float(min_persistence), float(initial_threshold_fraction) * float(values[0]))
    start_idx = int(np.where(values >= start_target)[0][-1])
    schedule = [float(values[start_idx])]
    current_idx = start_idx

    counts = np.array([
        sum(int(np.count_nonzero(np.isfinite(_diagram_persistences(dgm)) & (_diagram_persistences(dgm) >= thr))) for dgm in diagrams)
        for thr in values
    ], dtype=np.int64)

    while current_idx + 1 < values.size:
        current_count = max(int(counts[current_idx]), 1)
        max_count = int(np.floor((1.0 + max_active_growth) * current_count))
        next_idx = current_idx + 1
        valid = np.where(counts[current_idx + 1:] <= max_count)[0]
        if valid.size > 0:
            next_idx = current_idx + 1 + int(valid[-1])
        schedule.append(float(values[next_idx]))
        current_idx = next_idx

    if schedule[-1] > float(min_persistence):
        schedule.append(float(min_persistence))

    deduped = []
    for threshold in schedule:
        if not deduped or threshold < deduped[-1]:
            deduped.append(threshold)
    return deduped


def frechet_mean_newborn_points_from_newly_active(newly_active_diagrams, *, weights=None):
    newly_active_diagrams = _diagrams_to_numpy_list(newly_active_diagrams)
    normalized_weights = _normalize_frechet_weights(len(newly_active_diagrams), weights)

    new_points = []
    for diagram_weight, dgm in zip(normalized_weights, newly_active_diagrams):
        if dgm.size == 0:
            continue
        finite_mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1])
        finite_points = dgm[finite_mask]
        if finite_points.size == 0:
            continue
        midpoints = 0.5 * (finite_points[:, 0] + finite_points[:, 1])
        births = diagram_weight * finite_points[:, 0] + (1.0 - diagram_weight) * midpoints
        deaths = diagram_weight * finite_points[:, 1] + (1.0 - diagram_weight) * midpoints
        new_points.append(np.column_stack((births, deaths)))

    if not new_points:
        return np.empty((0, 2), dtype=REAL_DTYPE)

    return np.ascontiguousarray(np.vstack(new_points))


def frechet_mean_multistart(diagrams,
                            *,
                            weights=None,
                            starts=("medoid", "second_medoid", "farthest_from_medoid"),
                            return_details: bool = False,
                            n_threads: int = 1,
                            **kwargs):
    diagrams = [as_real_numpy(_check_numpy_diagram_shape(d)) for d in diagrams]
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)
    if not starts:
        raise ValueError("starts must be non-empty")

    results = []
    for start_idx, start in enumerate(starts):
        if isinstance(start, dict):
            local_kwargs = dict(kwargs)
            local_kwargs.update(start)
            local_kwargs.pop("init_strategy", None)
            local_kwargs.pop("custom_initial_barycenter", None)
            seed = _resolve_multistart_seed(
                diagrams,
                local_kwargs.pop("seed", "medoid"),
                weights=normalized_weights,
                domain=local_kwargs.get("domain", DiagramPlaneDomain.AboveDiagonal),
                random_noise_scale=local_kwargs.get("random_noise_scale", 1.0),
                random_seed=local_kwargs.get("random_seed", 42 + start_idx),
                grid_n_x_bins=local_kwargs.get("grid_n_x_bins", 16),
                grid_n_y_bins=local_kwargs.get("grid_n_y_bins", 16),
                wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
                internal_p=local_kwargs.get("internal_p", np.inf),
                n_threads=n_threads,
            )
        else:
            local_kwargs = dict(kwargs)
            local_kwargs.pop("init_strategy", None)
            local_kwargs.pop("custom_initial_barycenter", None)
            seed = _resolve_multistart_seed(
                diagrams,
                start,
                weights=normalized_weights,
                domain=local_kwargs.get("domain", DiagramPlaneDomain.AboveDiagonal),
                random_noise_scale=local_kwargs.get("random_noise_scale", 1.0),
                random_seed=local_kwargs.get("random_seed", 42 + start_idx),
                grid_n_x_bins=local_kwargs.get("grid_n_x_bins", 16),
                grid_n_y_bins=local_kwargs.get("grid_n_y_bins", 16),
                wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
                internal_p=local_kwargs.get("internal_p", np.inf),
                n_threads=n_threads,
            )

        barycenter = frechet_mean(
            diagrams,
            weights=normalized_weights,
            init_strategy=FrechetMeanInit.Custom,
            custom_initial_barycenter=seed,
            n_threads=n_threads,
            **local_kwargs,
        )
        objective = frechet_mean_objective(
            diagrams,
            barycenter,
            weights=normalized_weights,
            wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
            internal_p=local_kwargs.get("internal_p", np.inf),
            n_threads=n_threads,
        )
        results.append({"start": start, "barycenter": barycenter, "objective": objective})

    best = min(results, key=lambda item: item["objective"])
    if return_details:
        return best["barycenter"], {"objective": best["objective"], "runs": results}
    return best["barycenter"]


def progressive_frechet_mean(diagrams,
                             *,
                             weights=None,
                             thresholds=None,
                             initial_threshold_fraction: float = 0.5,
                             max_active_growth: float = 0.10,
                             min_persistence: float = 0.0,
                             initial_seed="medoid",
                             support_update_predicate=None,
                             support_update_fn=None,
                             return_details: bool = False,
                             n_threads: int = 1,
                             **kwargs):
    diagrams = _diagrams_to_numpy_list(diagrams)
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)
    ignore_infinite_points = bool(kwargs.get("ignore_infinite_points", False))

    if thresholds is None:
        thresholds = make_frechet_mean_persistence_schedule(
            diagrams,
            initial_threshold_fraction=initial_threshold_fraction,
            max_active_growth=max_active_growth,
            min_persistence=min_persistence,
        )
    else:
        thresholds = [float(t) for t in thresholds]
        if not thresholds:
            raise ValueError("thresholds must be non-empty")

    local_kwargs = dict(kwargs)
    local_kwargs.pop("init_strategy", None)
    local_kwargs.pop("custom_initial_barycenter", None)

    barycenter = None
    history = []
    previous_threshold = np.inf

    for stage_idx, threshold in enumerate(thresholds):
        active_diagrams = [
            _threshold_diagram_by_persistence(
                dgm,
                threshold,
                include_infinite_points=not ignore_infinite_points,
            )
            for dgm in diagrams
        ]

        if barycenter is None:
            seed = _resolve_multistart_seed(
                active_diagrams,
                initial_seed,
                weights=normalized_weights,
                domain=local_kwargs.get("domain", DiagramPlaneDomain.AboveDiagonal),
                random_noise_scale=local_kwargs.get("random_noise_scale", 1.0),
                random_seed=local_kwargs.get("random_seed", 42),
                grid_n_x_bins=local_kwargs.get("grid_n_x_bins", 16),
                grid_n_y_bins=local_kwargs.get("grid_n_y_bins", 16),
                wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
                internal_p=local_kwargs.get("internal_p", np.inf),
                n_threads=n_threads,
            )
        else:
            seed = barycenter
            if support_update_predicate is not None and support_update_fn is not None:
                newly_active_diagrams = [
                    _newly_active_diagram_points(dgm, previous_threshold, threshold)
                    for dgm in diagrams
                ]
                should_update = bool(support_update_predicate(
                    stage_index=stage_idx,
                    threshold=threshold,
                    previous_threshold=previous_threshold,
                    current_barycenter=seed,
                    active_diagrams=active_diagrams,
                    newly_active_diagrams=newly_active_diagrams,
                    weights=normalized_weights,
                ))
                if should_update:
                    new_points = support_update_fn(
                        stage_index=stage_idx,
                        threshold=threshold,
                        previous_threshold=previous_threshold,
                        current_barycenter=seed,
                        active_diagrams=active_diagrams,
                        newly_active_diagrams=newly_active_diagrams,
                        weights=normalized_weights,
                    )
                    if new_points is not None:
                        new_points = as_real_numpy(_check_numpy_diagram_shape(new_points))
                        if new_points.size != 0:
                            seed = np.ascontiguousarray(np.vstack([seed, new_points]))

        barycenter = frechet_mean(
            active_diagrams,
            weights=normalized_weights,
            init_strategy=FrechetMeanInit.Custom,
            custom_initial_barycenter=seed,
            n_threads=n_threads,
            **local_kwargs,
        )

        objective = frechet_mean_objective(
            active_diagrams,
            barycenter,
            weights=normalized_weights,
            wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
            internal_p=local_kwargs.get("internal_p", np.inf),
            n_threads=n_threads,
        )
        history.append({
            "stage_index": stage_idx,
            "threshold": threshold,
            "n_active_points": int(sum(dgm.shape[0] for dgm in active_diagrams)),
            "barycenter": barycenter,
            "objective": objective,
        })
        previous_threshold = threshold

    if return_details:
        return barycenter, {"thresholds": thresholds, "history": history}
    return barycenter


def progressive_frechet_mean_multistart(diagrams,
                                        *,
                                        weights=None,
                                        starts=("medoid", "second_medoid", "farthest_from_medoid"),
                                        return_details: bool = False,
                                        n_threads: int = 1,
                                        **kwargs):
    diagrams = [as_real_numpy(_check_numpy_diagram_shape(d)) for d in diagrams]
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)
    if not starts:
        raise ValueError("starts must be non-empty")

    results = []
    for start_idx, start in enumerate(starts):
        local_kwargs = dict(kwargs)
        initial_seed = local_kwargs.pop("initial_seed", start)

        if isinstance(start, dict):
            local_kwargs.update(start)
            initial_seed = local_kwargs.pop("initial_seed", local_kwargs.pop("seed", "medoid"))

        barycenter, details = progressive_frechet_mean(
            diagrams,
            weights=normalized_weights,
            initial_seed=initial_seed,
            return_details=True,
            n_threads=n_threads,
            **local_kwargs,
        )
        objective = frechet_mean_objective(
            diagrams,
            barycenter,
            weights=normalized_weights,
            wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
            internal_p=local_kwargs.get("internal_p", np.inf),
            n_threads=n_threads,
        )
        results.append({
            "start": start,
            "barycenter": barycenter,
            "objective": objective,
            "progressive_details": details,
        })

    best = min(results, key=lambda item: item["objective"])
    if return_details:
        return best["barycenter"], {
            "objective": best["objective"],
            "runs": results,
            "thresholds": best["progressive_details"]["thresholds"],
            "history": best["progressive_details"]["history"],
        }
    return best["barycenter"]


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
                 custom_initial_barycenter=None,
                 n_threads: int = 1):
    diagrams = [as_real_numpy(_check_numpy_diagram_shape(d)) for d in diagrams]
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if weights.shape[0] != len(diagrams):
            raise ValueError("weights must have same length as diagrams")
    custom_initial_barycenter = (
        None if custom_initial_barycenter is None
        else as_real_numpy(_check_numpy_diagram_shape(custom_initial_barycenter))
    )

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
                             custom_initial_barycenter=custom_initial_barycenter,
                             n_threads=n_threads)


def to_scipy_matrix(sparse_cols, shape=None):
    if shape is None:
        shape = (len(sparse_cols), len(sparse_cols))
    row_ind = [j for i in range(len(sparse_cols)) for j in sparse_cols[i]]
    col_ind = [i for i in range(len(sparse_cols)) for _ in sparse_cols[i]]
    assert (len(row_ind) == len(col_ind))
    data = [1 for _ in range(len(row_ind))]
    return scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape=shape)


def max_distance(data: np.ndarray, from_pwdists: bool=False):
    # 1.00001 is a small fudge factor so the returned bound is strictly
    # greater than every pairwise distance even after floating-point
    # rounding; callers use this as a max_diameter that must enclose all
    # edges
    if from_pwdists:
        return 1.00001 * np.min(np.max(data, axis=1))
    else:
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("max_distance: data must be a 2D array with at least 2 rows")
        diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
        squared_distances = np.sum(diff**2, axis=2)
        return 1.00001 * np.sqrt(np.min(np.max(squared_distances, axis=1)))


_FR_GRID_CLASS_BY_NDIM = {1: Grid_1D, 2: Grid_2D, 3: Grid_3D, 4: Grid_4D}


def freudenthal_filtration(data: np.ndarray,
                           negate: bool=False,
                           wrap: bool=False,
                           max_dim: int = 3,
                           with_critical_vertices: bool=False,
                           *,
                           slim: bool=True,
                           n_threads: int=1):
    # route to the float32 or float64 backend by the input dtype (float32 numpy/torch
    # arrays build a genuine float32 filtration; everything else defaults to float64)
    dt = detect_real_dtype(data)
    sub = REAL_MODULES[dt]
    data = as_real_numpy(data, dtype=dt)
    max_dim = min(max_dim, data.ndim)
    # slim (the default) returns the compact (anchor,type) _FreudenthalFiltration_ND (one shared
    # FrGeometry, fat simplices materialized on access) for D=1,2,3,4 on non-wrap grids; it reduces,
    # produces diagrams, optimizes (oineus.TopologyOptimizer dispatches it) and supports KICR
    # identically to the fat path but with a far smaller boundary-build footprint. wrap grids and
    # D>=5 always fall back to the fat universal Filtration (FrGeometry rejects wrap; the slim
    # builder is bound only for D=1,2,3,4). Pass slim=False to force the fat path.
    use_slim = slim and (not wrap) and (1 <= data.ndim <= 4)
    if use_slim:
        grid_cls = {1: sub.Grid_1D, 2: sub.Grid_2D, 3: sub.Grid_3D, 4: sub.Grid_4D}[data.ndim]
        grid = grid_cls(data, wrap=wrap, values_on="vertices")
        if with_critical_vertices:
            fil, vertices = grid.freudenthal_filtration_and_critical_vertices_slim(max_dim=max_dim, negate=negate, n_threads=n_threads)
            vertices = np.array(vertices, dtype=np.int64)
            return fil, vertices
        return grid.freudenthal_filtration_slim(max_dim=max_dim, negate=negate, n_threads=n_threads)
    if with_critical_vertices:
        fil, vertices = sub.get_freudenthal_filtration_and_crit_vertices(data=data, negate=negate, wrap=wrap, max_dim=max_dim, n_threads=n_threads)
        vertices = np.array(vertices, dtype=np.int64)
        return fil, vertices
    return sub.get_freudenthal_filtration(data=data, negate=negate, wrap=wrap, max_dim=max_dim, n_threads=n_threads)


def _vr_packed_word_suffix(n_points, max_dim):
    # Smallest packed word that holds a (max_dim)-simplex over n_points vertices:
    # bits = ceil(log2(n_points)) per field (== C++ packed_vertex_bits), (max_dim+1)
    # fields. Returns "64", "128", or None (too wide -> fat fallback).
    bits = _packed_bits(n_points)
    width = (int(max_dim) + 1) * bits
    if width <= 64:
        return "64"
    if width <= 128:
        return "128"
    return None


def _packed_bits(n_points):
    # bits per packed vertex field == C++ oin::packed_vertex_bits(n_points);
    # passed to the packed array builders so they skip a full max-id scan.
    return max(1, (int(n_points) - 1).bit_length())


def vr_filtration(data: np.ndarray,
                  from_pwdists: bool = False,
                  max_dim: int = -1,
                  max_diameter: float = -1.0,
                  with_critical_edges: bool = False,
                  *,
                  packed: bool = True,
                  n_threads: int = 1):
    """Build a Vietoris-Rips filtration from points or pairwise distances.

    Construction uses the in-order generation (VRE) algorithm of
    Vejdemo-Johansson, Matuszewski & Bauer ("In-order generation of
    Vietoris-Rips Complexes", arXiv:2411.05495).

    Parameters
    ----------
    data : np.ndarray
        (n, d) array of points, or (n, n) pairwise distance matrix.
    from_pwdists : bool
        Treat ``data`` as a pairwise distance matrix.
    max_dim : int
        Largest simplex dimension to enumerate. Default: data dimensionality.
    max_diameter : float
        Truncation threshold; only simplices with diameter <= this value are
        kept. Default: enclosing radius of the point cloud.
    with_critical_edges : bool
        Also return an array (one per simplex) of an edge whose length equals
        the simplex's filtration value.
    packed : bool
        Use the compact bit-packed cell encoding (the default) when the vertex
        ids fit a 64- or 128-bit word; falls back to the fat encoding otherwise.
        Pass packed=False to force the fat universal Simplex filtration.
    n_threads : int
        Threads used for the (parallel) sort inside the Filtration ctor.
        Enumeration itself is single-threaded.
    """
    # route to the float32 or float64 backend by the input dtype, then coerce the
    # point/distance array to that Real so the matching get_vr_* builder accepts it
    dt = detect_real_dtype(data)
    sub = REAL_MODULES[dt]
    data = as_real_numpy(data, dtype=dt)

    if data.ndim != 2:
        raise ValueError("data must be a 2D array")

    if from_pwdists and data.shape[0] != data.shape[1]:
        raise ValueError("from_pwdists=True requires a square pairwise-distance matrix")

    if max_diameter < 0:
        max_diameter = max_distance(data, from_pwdists)

    if max_dim < 0:
        if from_pwdists:
            raise RuntimeError("vr_filtration: if input is pairwise distance matrix, max_dim must be specified")
        else:
            max_dim = data.shape[1]

    # packed (the default) returns a bit-packed _PackedSimplexFiltration_64/128 (compact cells,
    # one shared PackedGeom) when the vertex ids fit a 64- or 128-bit word; if they do not fit
    # (very large/high-dim complex) it transparently falls back to the fat path. It reduces,
    # produces diagrams, optimizes (oineus.TopologyOptimizer dispatches it), supports KICR and
    # the uid-keyed accessors (via the combinatorial-uid translation) identically to fat but with
    # a smaller footprint. Pass packed=False to force the fat universal Simplex filtration.
    suffix = _vr_packed_word_suffix(data.shape[0], max_dim) if packed else None

    if from_pwdists:
        if suffix is not None:
            base = "get_vr_filtration_and_critical_edges_packed" if with_critical_edges else "get_vr_filtration_packed"
            func = getattr(sub, base + suffix + "_from_pwdists")
        elif with_critical_edges:
            func = sub.get_vr_filtration_and_critical_edges_from_pwdists
        else:
            func = sub.get_vr_filtration_from_pwdists
    else:
        if suffix is not None:
            base = "get_vr_filtration_and_critical_edges_packed" if with_critical_edges else "get_vr_filtration_packed"
            func = getattr(sub, base + suffix)
        elif with_critical_edges:
            func = sub.get_vr_filtration_and_critical_edges
        else:
            func = sub.get_vr_filtration

    result = func(data, max_dim=max_dim, max_diameter=max_diameter, n_threads=n_threads)
    if with_critical_edges:
        # convert list of VREdges to numpy array
        edges = [ [ e.x, e.y] for e in result[1] ]
        edges = np.array(edges, dtype=np.int64)
        return result[0], edges
    else:
        return result


def is_reduced(a):
    """Check whether a Z_2 boundary matrix is reduced.

    A column is treated as nonzero in row ``i`` iff ``a[i, col] % 2 == 1``,
    so any integer dtype with mod-2 semantics works (binary 0/1 matrices,
    or unreduced count matrices). Returns ``True`` iff every nonzero
    column has a distinct lowest-1 row index, which is the definition of
    a reduced matrix in the standard persistence reduction.

    Args:
        a: 2D array-like with ``.shape[1]`` columns and integer entries.

    Returns:
        bool: True if the matrix is reduced.
    """
    lowest_ones = []
    for col_idx in range(a.shape[1]):
        if np.any(a[:, col_idx] % 2 == 1):
            lowest_ones.append(np.max(np.where(a[:, col_idx] % 2 == 1)))
    return len(lowest_ones) == len(set(lowest_ones))

_SLIM_SIMPLEX_FIL_TYPES = tuple(
    f for s in REAL_MODULES.values() for f in (
        s._FreudenthalFiltration_1D, s._FreudenthalFiltration_2D,
        s._FreudenthalFiltration_3D, s._FreudenthalFiltration_4D,
        s._PackedSimplexFiltration_64, s._PackedSimplexFiltration_128,
    ))


def _to_fat_simplex_filtration(fil):
    """Materialize a slim/packed simplicial filtration into a fat _Filtration.

    mapping_cylinder / multiply_filtration build fat ProductCell<Simplex, Simplex>
    filtrations, so they need fat Simplex factors. A slim Freudenthal / bit-packed
    filtration materializes fat Simplex cells on access, so rebuild a fat _Filtration
    from them. Fat Simplex / product filtrations pass through unchanged; cube (and
    anything else) is returned as-is so the C++ overload rejects it as before.
    """
    if isinstance(fil, _SLIM_SIMPLEX_FIL_TYPES):
        # fil.cells() materializes fat Simplex cells in fil's own (float32/float64)
        # submodule, so build the fat _Filtration from the matching backend
        return module_of_oineus_obj(fil)._Filtration(fil.cells(), fil.negate)
    return fil


def mapping_cylinder(fil_domain, fil_codomain, v_domain, v_codomain,
                     v_domain_value=None, v_codomain_value=None, with_indices=False):
    """Build the mapping cylinder of the inclusion fil_domain -> fil_codomain.

    The auxiliary vertex values default to filtration-order -inf
    (``fil_domain.neg_infinity()`` / ``fil_codomain.neg_infinity()``), which
    keeps the cylinder's persistent homology equivalent to the inclusion's.
    Pass explicit values only if you intentionally want the auxiliary vertices
    to enter at a finite point in the filtration.

    Slim Freudenthal / bit-packed inputs are materialized to the fat encoding
    first (the cylinder is built over fat product cells); the cells keep their
    combinatorial uids, so uid-keyed lookups against the original filtrations
    still resolve.
    """
    fil_domain = _to_fat_simplex_filtration(fil_domain)
    fil_codomain = _to_fat_simplex_filtration(fil_codomain)
    # Accept either valued (oin.Simplex / SimplexValue) or bare (CombinatorialSimplex)
    # auxiliary vertices. Strip the value -- we use the explicit *_value args
    # below, so any value baked into the simplex would only confuse readers.
    if isinstance(v_codomain, _oineus.Simplex) or isinstance(v_codomain, _oineus.ProdSimplex):
        v_codomain = v_codomain.combinatorial_cell
    if isinstance(v_domain, _oineus.Simplex) or isinstance(v_domain, _oineus.ProdSimplex):
        v_domain = v_domain.combinatorial_cell
    if v_domain_value is None:
        v_domain_value = fil_domain.neg_infinity()
    if v_codomain_value is None:
        v_codomain_value = fil_codomain.neg_infinity()
    # route to the backend matching the (fattened) filtrations' dtype
    sub = module_of_oineus_obj(fil_domain)
    if with_indices:
        return sub._mapping_cylinder_with_indices(fil_domain, fil_codomain, v_domain, v_codomain,
                                                  v_domain_value, v_codomain_value)
    else:
        return sub._mapping_cylinder(fil_domain, fil_codomain, v_domain, v_codomain,
                                     v_domain_value, v_codomain_value)

def multiply_filtration(fil, sigma, sigma_value=None):
    """Multiply every cell in fil by the auxiliary simplex sigma.

    Each product cell receives value ``fil.fil_max(cell.value, sigma_value)``.
    sigma_value defaults to ``fil.neg_infinity()`` so each product cell
    inherits its primary factor's value unchanged.

    A slim Freudenthal / bit-packed fil is materialized to the fat encoding first
    (the product cells are fat).
    """
    fil = _to_fat_simplex_filtration(fil)
    if isinstance(sigma, _oineus.Simplex):
        sigma = sigma.combinatorial_cell
    if sigma_value is None:
        sigma_value = fil.neg_infinity()
    return module_of_oineus_obj(fil)._multiply_filtration(fil, sigma, sigma_value)

def min_filtration(fil_1, fil_2, with_indices=False):
    # route to the backend matching the input filtrations' dtype
    sub = module_of_oineus_obj(fil_1)
    if with_indices:
        return sub._min_filtration_with_indices(fil_1, fil_2)
    else:
        return sub._min_filtration(fil_1, fil_2)


# The five helpers below take a filtration argument and were historically bound only for the
# fat float64 Simplex. The C++ overloads are now folded over every cell type and registered in
# both Real backends; these wrappers route each call to the (sub)module matching the
# filtration's dtype, so they work on the now-default packed VR / slim Freudenthal / cube
# filtrations and on float32 filtrations -- mirroring min_filtration / mapping_cylinder.

def get_nth_persistence(fil, rv_matrix, dim, n):
    """The n-th largest persistence value in the given homology dimension."""
    return module_of_oineus_obj(fil).get_nth_persistence(fil, rv_matrix, dim, n)


def get_denoise_target(dim, fil, rv_matrix, eps, strategy):
    """Target values for topological denoising (DiagramToValues)."""
    return module_of_oineus_obj(fil).get_denoise_target(dim, fil, rv_matrix, eps, strategy)


def get_permutation(target_matching, fil):
    """Permutation realizing the requested simplex-to-value targets (warm starts)."""
    return module_of_oineus_obj(fil).get_permutation(target_matching, fil)


def get_permutation_dtv(diagram_to_values, fil):
    """Permutation realizing the requested diagram-to-value targets (warm starts)."""
    return module_of_oineus_obj(fil).get_permutation_dtv(diagram_to_values, fil)


def compute_relative_diagrams(fil, rel, include_inf_points=True):
    """Relative persistence diagrams of the pair (fil, rel). fil and rel must share
    the same cell type and Real dtype."""
    return module_of_oineus_obj(fil).compute_relative_diagrams(
        fil, rel, include_inf_points=include_inf_points)


def get_induced_matching(included_filtration, containing_filtration, dim=-1, n_threads=1):
    """Induced matching between the diagrams of two filtrations on the same complex.

    dim < 0 (the default) matches across all homology dimensions."""
    sub = module_of_oineus_obj(included_filtration)
    # dim_type is unsigned in C++; -1 ("all dims") is its SIZE_MAX default, which a Python -1
    # cannot convert to -- so omit the argument and let the C++ default apply.
    if dim is None or dim < 0:
        return sub.get_induced_matching(included_filtration, containing_filtration, n_threads=n_threads)
    return sub.get_induced_matching(
        included_filtration, containing_filtration, dim=dim, n_threads=n_threads)

def remove_simplices(fil, dcmp, seeds, *, close_star=True, stats=None, n_threads=1):
    """SiRUP: remove a coface-closed set of cells from a reduced decomposition.

    Updates ``dcmp`` in place to the reduced R = D V decomposition of the
    filtration with the requested cells removed, updating both the barcode and
    the representative cycles, instead of recomputing from scratch (Giunti and
    Lazovskis, "Pruning vineyards: updating barcodes and representative cycles
    by removing simplices").

    Parameters
    ----------
    fil
        The filtration ``dcmp`` was reduced from.
    dcmp
        A reduced Decomposition with V (reduce with compute_v = True), from the
        classic ``oin.Decomposition(fil); dcmp.reduce(params)`` path. Homology
        only. Mutated in place.
    seeds
        sorted_ids of the cells to remove. By default their coface up-closure
        (union of stars) is taken so that the result is a valid filtration; set
        ``close_star=False`` if ``seeds`` is already coface-closed.
    close_star
        Whether to expand ``seeds`` to ``fil.star_closure(seeds)`` first.
    stats
        Optional DecompositionManipStats collecting column-op counts and timings.
    n_threads
        Threads for the internal row-index / closure build.

    Returns
    -------
    A new filtration on the surviving cells, in the same order as the updated
    decomposition's columns, so ``dcmp.diagram(new_fil)`` gives the updated
    diagram.
    """
    seeds = [int(s) for s in seeds]
    cells = fil.star_closure(seeds, n_threads) if close_star else seeds
    dcmp.remove_simplices(cells, stats, n_threads)
    return fil.without_cells(cells)

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
    return dcmp.diagram(fil=fil, include_inf_points=include_inf_points, n_threads=params.n_threads)


def compute_diagrams_vr(data: np.ndarray, from_pwdists: bool=False, max_dim: int=-1, max_diameter: float = -1.0, params: typing.Optional[ReductionParams]=None, include_inf_points: bool=True, dualize: bool=True):
    if params is None:
        params = _oineus.ReductionParams()
    # max_dim is maximal dimension of the _diagram_, we need simplices one dimension higher, hence +1
    fil = vr_filtration(data, from_pwdists, max_dim=max_dim, max_diameter=max_diameter, with_critical_edges=False, n_threads=params.n_threads)
    dcmp = _oineus.Decomposition(fil, dualize)
    dcmp.reduce(params)
    return dcmp.diagram(fil=fil, include_inf_points=include_inf_points, n_threads=params.n_threads)


def alpha_filtration(points: np.ndarray,
                     weights: typing.Optional[np.ndarray]=None,
                     exact: bool=False,
                     periodic: bool=False,
                     compute_bounding_box: bool=True,
                     bbox_min=None,
                     bbox_max=None,
                     *,
                     packed: bool=True,
                     n_threads: int=1):
    """Build an alpha-shape filtration from a 2D or 3D point cloud.

    Combinatorics come from diode (CGAL Delaunay); filtration values are the
    alpha values returned by diode. For one-shot diagrams use
    :func:`compute_diagrams_alpha`; the differentiable Cech-Delaunay path
    reuses the same combinatorics with autograd-attached values.

    Args:
        points: NumPy array of shape (n, 2) or (n, 3).
        weights: Optional 1D array of length n. If provided, computes
            weighted (regular) alpha-shapes; currently 3D only.
        exact: Use CGAL's exact kernel. Slower but robust against numerical
            pathologies.
        periodic: Use periodic alpha-shapes on a torus.
        compute_bounding_box: If True, use the bounding box of the data.
        bbox_min, bbox_max: Bounding box if compute_bounding_box=False.
        packed: Use the compact bit-packed cell encoding (the default) on the
            fast unweighted/non-periodic array path when the vertex ids fit a
            64/128-bit word. Pass packed=False to force the fat encoding.
        n_threads: Threads used inside the Filtration constructor.

    Returns:
        A Filtration over alpha-shape simplices.
    """
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n_points, dim)")
    if points.shape[1] not in (2, 3):
        raise ValueError("Alpha-shapes only support 2D and 3D point clouds")
    if not _HAS_DIODE:
        raise ImportError(
            "Alpha-shape construction requires the `diode` package "
            "(https://github.com/mrzv/diode). Install it via "
            "`pip install diode` or build from source."
        )

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
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if weights.shape[0] != points.shape[0]:
            raise ValueError("weights must have same length as points")
        if points.shape[1] != 3:
            raise ValueError("Weighted alpha-shapes require 3D points")

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
        elif _HAS_DIODE_ARRAYS:
            # Fast array path: simplices and alpha values come back as NumPy
            # arrays, avoiding one Python tuple per simplex. Same combinatorics
            # and values as fill_alpha_shapes.
            verts_by_dim, vals_by_dim = diode.fill_alpha_shapes_arrays(points, exact=exact)
            # Route the filtration to the backend matching the points' dtype (float32 points ->
            # a genuine float32 alpha filtration, like vr_filtration / freudenthal_filtration).
            # CGAL/diode compute the alpha values in double, so narrow them to the target Real.
            dt = detect_real_dtype(points)
            sub = REAL_MODULES[dt]
            if dt != DEFAULT_REAL_DTYPE:
                vals_by_dim = [np.ascontiguousarray(v, dtype=dt) for v in vals_by_dim]
            # packed (the default) returns a bit-packed PackedSimplexFiltration when the vertex
            # ids fit a 64/128-bit word (only this fast array path supports it; the
            # weighted/periodic/list fallbacks below stay fat). Reduces and produces
            # diagrams identically to fat, with a smaller footprint.
            suffix = _vr_packed_word_suffix(points.shape[0], points.shape[1]) if packed else None
            if suffix is not None:
                # bits passed directly (skips the C++ max-id scan); diode rows are
                # not vertex-sorted, so assume_sorted stays False.
                fil = getattr(sub, "_filtration_from_arrays_packed" + suffix)(
                    verts_by_dim, vals_by_dim, n_threads=n_threads, bits=_packed_bits(points.shape[0]))
            else:
                fil = sub._filtration_from_arrays(verts_by_dim, vals_by_dim, n_threads=n_threads)
            fil.kind = _oineus.FiltrationKind.Alpha
            return fil
        else:
            fil_diode = diode.fill_alpha_shapes(points, exact=exact)

    # Route the weighted / periodic / non-array-exporter fallbacks to the backend matching the
    # points' dtype, like the fast array path above (and vr_filtration / freudenthal_filtration).
    # diode computes alpha values in double; the float32 _Filtration ctor narrows them, same as
    # the array path's explicit narrowing.
    sub = REAL_MODULES[detect_real_dtype(points)]
    fil = sub._Filtration(
        fil_diode,
        duplicates_possible=periodic,
        n_threads=n_threads,
    )
    fil.kind = _oineus.FiltrationKind.Alpha
    return fil


def _delaunay_combinatorics(points: np.ndarray, exact: bool=False, packed: bool=False, n_threads: int=1):
    """Build the Delaunay complex as a Filtration, for its combinatorics only.

    For callers that recompute and set their own values (the differentiable
    Cech-Delaunay and weak-alpha filtrations), so the filtration values here are
    not meaningful and must be overwritten via set_values. The fast path
    (diode's fill_delaunay_arrays) leaves them at 0; the fallback
    (alpha_filtration, used when the array exporters are absent) carries alpha
    values instead. Either way the simplex set is identical for full-dimensional
    input.

    Args:
        points: NumPy array of shape (n, 2) or (n, 3).
        exact: Use CGAL's exact kernel.
        packed: Use the compact bit-packed cell encoding when the vertex ids fit
            a 64/128-bit word (only the fast diode-array path supports it; the
            alpha_filtration fallback honors packed too).
        n_threads: Threads used inside the Filtration constructor.

    Returns:
        A Filtration over the Delaunay simplices; its values are not meaningful
        and are expected to be overwritten by the caller.
    """
    if _HAS_DIODE_ARRAYS:
        verts_by_dim = diode.fill_delaunay_arrays(points, exact=exact)
        # route to the backend matching the points' dtype so a float32 point cloud yields a
        # float32 Delaunay filtration (the differentiable cech/weak paths then set float32
        # values into it via real_buffer_for). Only the vertex arrays are used here; the
        # caller overwrites the values, so no value-array dtype handling is needed.
        sub = real_module_for(points)
        suffix = _vr_packed_word_suffix(points.shape[0], points.shape[1]) if packed else None
        if suffix is not None:
            return getattr(sub, "_filtration_from_arrays_packed" + suffix)(
                verts_by_dim, None, n_threads=n_threads, bits=_packed_bits(points.shape[0]))
        return sub._filtration_from_arrays(verts_by_dim, None, n_threads=n_threads)
    return alpha_filtration(points, exact=exact, packed=packed, n_threads=n_threads)


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
    fil = alpha_filtration(
        points,
        weights=weights,
        exact=exact,
        periodic=periodic,
        compute_bounding_box=compute_bounding_box,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        n_threads=params.n_threads,
    )
    dcmp = _oineus.Decomposition(fil, dualize)
    dcmp.reduce(params)
    return dcmp.diagram(fil=fil, include_inf_points=include_inf_points, n_threads=params.n_threads)


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

    # KICR is templated on the cell type in C++; each encoding has its own bound class.
    # Dispatch on the FILTRATION type, not K[0]: a slim/packed filtration materializes a
    # fat Simplex on K[0], which would misdispatch it into the fat ctor. type(K) is the
    # stable C++ class for every encoding (and is correct for empty filtrations too).
    KICR_Class = _KICR_CLASS_BY_FIL_TYPE.get(type(K))
    if KICR_Class is None:
        raise TypeError(
            f"compute_kernel_image_cokernel_reduction: unsupported filtration type "
            f"{type(K).__name__}; expected one of "
            f"{[t.__name__ for t in _KICR_CLASS_BY_FIL_TYPE]}")

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
    fil_min = min_filtration(fil_2, fil_3)

    id_domain = fil_3.size() + fil_min.size() + 1
    id_codomain = id_domain + 1

    # id_domain: id of vertex at the top of the cylinder,
    # i.e., we multiply fil_3 with id_domain
    # id_codomain: id of vertex at the bottom of the cylinder
    # i.e, we multiply fil_min with id_codomain

    # The wrapper functions below default the auxiliary vertex values to
    # fil.neg_infinity(), so the value carried on these Simplex objects is
    # discarded. We still need a Simplex/CombinatorialSimplex to specify
    # the vertex labels.
    v0 = _oineus.CombinatorialSimplex(id_domain, [id_domain])
    v1 = _oineus.CombinatorialSimplex(id_codomain, [id_codomain])

    fil_cyl = mapping_cylinder(fil_3, fil_min, v0, v1)

    # to get a subcomplex, we multiply each fil_3 with id_domain
    fil_3_prod = multiply_filtration(fil_3, v0)

    params = _oineus.KICRParams()
    params.kernel = params.cokernel = True
    params.image = False

    # route the product KICR to the backend matching the cylinder's dtype
    kicr_reduction = module_of_oineus_obj(fil_cyl).KerImCokReducedProd(fil_cyl, fil_3_prod, params)

    return kicr_reduction


def cube_filtration(data: np.ndarray,
                    negate: bool=False,
                    wrap: bool=False,
                    max_dim: typing.Optional[int]=None,
                    values_on: str="vertices",
                    n_threads: int=1,):
    if wrap:
        raise RuntimeError("cube_filtration: wrap=True is not implemented yet")
    # route to the float32 or float64 backend by the input dtype
    dt = detect_real_dtype(data)
    sub = REAL_MODULES[dt]
    data = as_real_numpy(data, dtype=dt)
    if max_dim is None:
        max_dim = data.ndim
    dim = data.ndim
    if dim == 1:
        grid = sub.Grid_1D(data, wrap=wrap, values_on=values_on)
    elif dim == 2:
        grid = sub.Grid_2D(data, wrap=wrap, values_on=values_on)
    elif dim == 3:
        grid = sub.Grid_3D(data, wrap=wrap, values_on=values_on)
    elif dim == 4:
        grid = sub.Grid_4D(data, wrap=wrap, values_on=values_on)
    else:
        raise RuntimeError(f"cube_filtration: dim={data.ndim} not supported, recompile from sources")
    fil = grid.cube_filtration(max_dim=max_dim, n_threads=n_threads, negate=negate)
    return fil


_PUBLIC_API_NAMES = [
    "ConflictStrategy",
    "DenoiseStrategy",
    "VREdge",
    "FiltrationKind",
    "DiagramPlaneDomain",
    "FrechetMeanInit",
    "CombinatorialProdSimplex",
    "CombinatorialSimplex",
    "Simplex",
    "ProdSimplex",
    "Filtration",
    "ProdFiltration",
    "Decomposition",
    "IndexDiagramPoint",
    "DiagramPoint",
    "Diagrams",
    "reduce",
    "DecompositionManipStats",
    "ReductionParams",
    "ReductionTimings",
    "KICRParams",
    "KerImCokReduced",
    "KerImCokReducedProd",
    "ColumnRepr",
    "IndicesValues",
    "IndicesValuesProd",
    "TopologyOptimizer",
    "TopologyOptimizerProd",
    "TopologyOptimizerCube_1D",
    "TopologyOptimizerCube_2D",
    "TopologyOptimizerCube_3D",
    "TopologyOptimizerCube_4D",
    "compute_relative_diagrams",
    "get_boundary_matrix",
    "get_denoise_target",
    "get_induced_matching",
    "get_nth_persistence",
    "get_permutation_dtv",
    "get_permutation",
    "GridDomain_1D",
    "Grid_1D",
    "CombinatorialCube_1D",
    "Cube_1D",
    "GridDomain_2D",
    "Grid_2D",
    "CombinatorialCube_2D",
    "Cube_2D",
    "GridDomain_3D",
    "Grid_3D",
    "CombinatorialCube_3D",
    "Cube_3D",
    "GridDomain_4D",
    "Grid_4D",
    "CombinatorialCube_4D",
    "Cube_4D",
    "DiagramMatching",
    "BottleneckMatching",
    "InfKind",
    "EssentialMatches",
    "EssentialLongestEdges",
    "LongestEdges",
    "FiniteLongestEdge",
    "EssentialLongestEdge",
    "point_to_diagonal",
    "wasserstein_matching",
    "bottleneck_matching",
    "sliced_wasserstein_distance",
    "sliced_wasserstein_distance_diag_corrected",
    "REAL_DTYPE",
    "as_real_numpy",
    "bottleneck_distance",
    "wasserstein_distance",
    "init_frechet_mean_first_diagram",
    "init_frechet_mean_random_diagram",
    "init_frechet_mean_medoid_diagram",
    "init_frechet_mean_diagonal_grid",
    "frechet_mean_objective",
    "make_frechet_mean_persistence_schedule",
    "frechet_mean_newborn_points_from_newly_active",
    "frechet_mean_multistart",
    "progressive_frechet_mean",
    "progressive_frechet_mean_multistart",
    "frechet_mean",
    "to_scipy_matrix",
    "max_distance",
    "freudenthal_filtration",
    "vr_filtration",
    "is_reduced",
    "mapping_cylinder",
    "multiply_filtration",
    "min_filtration",
    "compute_diagrams_ls",
    "compute_diagrams_vr",
    "alpha_filtration",
    "compute_diagrams_alpha",
    "list_to_filtration",
    "compute_kernel_image_cokernel_reduction",
    "compute_ker_cok_reduction_cyl",
    "cube_filtration",
    "plot_diagram",
    "plot_diagram_gradient",
    "plot_matching",
    "plot_chain",
    "default_point_style",
    "default_diagram_a_point_style",
    "default_diagram_b_point_style",
    "default_matching_edge_style",
    "default_longest_edge_style",
    "default_diagonal_style",
    "default_diagonal_projection_a_style",
    "default_diagonal_projection_b_style",
    "default_inf_line_style",
    "default_inf_point_style",
    "default_diagram_gradient_style",
    "default_density_style",
    "default_grid_style",
    "default_chain_vertex_style",
    "default_chain_edge_style",
    "default_chain_triangle_style",
    "default_chain_tetrahedron_style",
    "default_point_cloud_style",
    "DEFAULT_POINT_STYLE",
    "DEFAULT_DIAGRAM_A_POINT_STYLE",
    "DEFAULT_DIAGRAM_B_POINT_STYLE",
    "DEFAULT_MATCHING_EDGE_STYLE",
    "DEFAULT_LONGEST_EDGE_STYLE",
    "DEFAULT_DIAGONAL_STYLE",
    "DEFAULT_DIAGONAL_PROJECTION_A_STYLE",
    "DEFAULT_DIAGONAL_PROJECTION_B_STYLE",
    "DEFAULT_INF_LINE_STYLE",
    "DEFAULT_INF_POINT_STYLE",
    "DEFAULT_DIAGRAM_GRADIENT_STYLE",
    "DEFAULT_DENSITY_STYLE",
    "DEFAULT_DENSITY_THRESHOLD",
    "DEFAULT_GRID_STYLE",
    "DEFAULT_MATCHING_EDGE_QUANTILE",
    "DEFAULT_GRADIENT_TOP_K_ARROWS",
    "DEFAULT_CHAIN_VERTEX_STYLE",
    "DEFAULT_CHAIN_EDGE_STYLE",
    "DEFAULT_CHAIN_TRIANGLE_STYLE",
    "DEFAULT_CHAIN_TETRAHEDRON_STYLE",
    "DEFAULT_POINT_CLOUD_STYLE",
    "OKABE_ITO_BLUE",
    "OKABE_ITO_VERMILLION",
]

__all__ = [name for name in _PUBLIC_API_NAMES if name in globals()]
