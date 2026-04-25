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
from ._oineus import init_frechet_mean_first_diagram as _init_frechet_mean_first_diagram_cpp
from ._oineus import init_frechet_mean_random_diagram as _init_frechet_mean_random_diagram_cpp
from ._oineus import init_frechet_mean_medoid_diagram as _init_frechet_mean_medoid_diagram_cpp
from ._oineus import init_frechet_mean_diagonal_grid as _init_frechet_mean_diagonal_grid_cpp
from ._oineus import frechet_mean as _frechet_mean_cpp
from ._oineus import GridDomain_1D, Grid_1D, CombinatorialCube_1D, Cube_1D, CubeFiltration_1D
from ._oineus import GridDomain_2D, Grid_2D, CombinatorialCube_2D, Cube_2D, CubeFiltration_2D
from ._oineus import GridDomain_3D, Grid_3D, CombinatorialCube_3D, Cube_3D, CubeFiltration_3D
from .vis_utils import (
    plot_diagram,
    plot_matching,
    default_point_style,
    default_diagram_a_point_style,
    default_diagram_b_point_style,
    default_matching_edge_style,
    default_longest_edge_style,
    default_diagonal_style,
    default_diagonal_projection_a_style,
    default_diagonal_projection_b_style,
    default_inf_line_style,
    DEFAULT_POINT_STYLE,
    DEFAULT_DIAGRAM_A_POINT_STYLE,
    DEFAULT_DIAGRAM_B_POINT_STYLE,
    DEFAULT_MATCHING_EDGE_STYLE,
    DEFAULT_LONGEST_EDGE_STYLE,
    DEFAULT_DIAGONAL_STYLE,
    DEFAULT_DIAGONAL_PROJECTION_A_STYLE,
    DEFAULT_DIAGONAL_PROJECTION_B_STYLE,
    DEFAULT_INF_LINE_STYLE,
)
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
from ._dtype import REAL_DTYPE, as_real_numpy
# from ._oineus import Z2_Column, Z2_Matrix

try:
    import diode
    _HAS_DIODE = True
except:
    _HAS_DIODE = False


__all__ = [
    "compute_diagrams_ls",
    "compute_diagrams_vr",
    "compute_diagrams_alpha",
    "get_boundary_matrix",
    "is_reduced",
    "plot_diagram",
    "plot_matching",
    "bottleneck_distance",
    "wasserstein_distance",
    "wasserstein_matching",
    "bottleneck_matching",
    "DiagramMatching",
    "BottleneckMatching",
    "InfKind",
    "EssentialMatches",
    "EssentialLongestEdges",
    "LongestEdges",
    "FiniteLongestEdge",
    "EssentialLongestEdge",
    "point_to_diagonal",
    "default_point_style",
    "default_diagram_a_point_style",
    "default_diagram_b_point_style",
    "default_matching_edge_style",
    "default_longest_edge_style",
    "default_diagonal_style",
    "default_diagonal_projection_a_style",
    "default_diagonal_projection_b_style",
    "default_inf_line_style",
    "DEFAULT_POINT_STYLE",
    "DEFAULT_DIAGRAM_A_POINT_STYLE",
    "DEFAULT_DIAGRAM_B_POINT_STYLE",
    "DEFAULT_MATCHING_EDGE_STYLE",
    "DEFAULT_LONGEST_EDGE_STYLE",
    "DEFAULT_DIAGONAL_STYLE",
    "DEFAULT_DIAGONAL_PROJECTION_A_STYLE",
    "DEFAULT_DIAGONAL_PROJECTION_B_STYLE",
    "DEFAULT_INF_LINE_STYLE",
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
]


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
                             internal_p):
    if not isinstance(seed, str):
        return as_real_numpy(_check_numpy_diagram_shape(seed))

    if seed == "first":
        return init_frechet_mean_first_diagram(diagrams)
    if seed == "medoid":
        return init_frechet_mean_medoid_diagram(diagrams, weights=weights)
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
    for i in range(n):
        for j in range(i + 1, n):
            dist = wasserstein_distance(
                diagrams[i],
                diagrams[j],
                q=2.0,
                delta=wasserstein_delta,
                internal_p=internal_p,
            )
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


def _prepare_distance_args(dgm_1, dgm_2, dim):
    """Coerce inputs into one of the C++ overloads' supported shapes.

    Returns either ``(Diagrams, Diagrams, dim)`` — when both are Diagrams —
    or ``(x, y)`` where each of ``x``, ``y`` is a list/numpy and nanobind will
    pick the matching overload. Mixed Diagrams + numpy/list is resolved by
    extracting the named ``dim`` slice from the Diagrams side.
    """
    dgm_1 = as_real_numpy(_check_numpy_diagram_shape(dgm_1))
    dgm_2 = as_real_numpy(_check_numpy_diagram_shape(dgm_2))
    is_d1 = isinstance(dgm_1, Diagrams)
    is_d2 = isinstance(dgm_2, Diagrams)

    if is_d1 and is_d2:
        if dim is None:
            raise ValueError("When passing oineus.Diagrams, specify dim=...")
        return (dgm_1, dgm_2, dim)

    if is_d1:
        if dim is None:
            raise ValueError("When passing oineus.Diagrams, specify dim=...")
        dgm_1 = dgm_1.in_dimension(dim, as_numpy=True)
    if is_d2:
        if dim is None:
            raise ValueError("When passing oineus.Diagrams, specify dim=...")
        dgm_2 = dgm_2.in_dimension(dim, as_numpy=True)

    return (dgm_1, dgm_2)


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
            Oineus `Diagrams` object.

    Returns:
        The bottleneck distance as a Python float.
    """
    return _bottleneck_distance_cpp(*_prepare_distance_args(dgm_1, dgm_2, dim), delta=delta)


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
        check_for_zero: If `True`, skip Hera when the two inputs are numpy
            arrays of equal points.
        dim: Homology dimension to extract when `dgm_1` or `dgm_2` is an
            Oineus `Diagrams` object.

    Returns:
        The Wasserstein distance as a Python float.
    """
    if wasserstein_q is not None:
        q = wasserstein_q
    if np.isinf(internal_p):
        internal_p = -1.0

    prepared = _prepare_distance_args(dgm_1, dgm_2, dim)

    if check_for_zero and len(prepared) == 2 and _diagram_arrays_equal_for_zero_check(*prepared):
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


def init_frechet_mean_medoid_diagram(diagrams, *, weights=None):
    return _init_frechet_mean_medoid_diagram_cpp(
        [as_real_numpy(d) for d in diagrams], weights=weights
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
                           internal_p: float = np.inf):
    normalized_weights = _normalize_frechet_weights(len(diagrams), weights)
    return float(sum(
        normalized_weights[i] * wasserstein_distance(
            barycenter,
            diagram,
            q=2.0,
            delta=wasserstein_delta,
            internal_p=internal_p,
        ) ** 2
        for i, diagram in enumerate(diagrams)
    ))


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
            )

        barycenter = frechet_mean(
            diagrams,
            weights=normalized_weights,
            init_strategy=FrechetMeanInit.Custom,
            custom_initial_barycenter=seed,
            **local_kwargs,
        )
        objective = frechet_mean_objective(
            diagrams,
            barycenter,
            weights=normalized_weights,
            wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
            internal_p=local_kwargs.get("internal_p", np.inf),
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
            **local_kwargs,
        )

        objective = frechet_mean_objective(
            active_diagrams,
            barycenter,
            weights=normalized_weights,
            wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
            internal_p=local_kwargs.get("internal_p", np.inf),
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
            **local_kwargs,
        )
        objective = frechet_mean_objective(
            diagrams,
            barycenter,
            weights=normalized_weights,
            wasserstein_delta=local_kwargs.get("wasserstein_delta", 0.01),
            internal_p=local_kwargs.get("internal_p", np.inf),
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
                 custom_initial_barycenter=None):
    diagrams = [as_real_numpy(_check_numpy_diagram_shape(d)) for d in diagrams]
    if weights is not None:
        weights = np.asarray(weights)
        assert weights.ndim == 1, "weights must be a 1D array"
        assert weights.shape[0] == len(diagrams), "weights must have same length as diagrams"
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
