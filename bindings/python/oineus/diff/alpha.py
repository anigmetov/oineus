"""Differentiable alpha filtration.

Combinatorics come from CGAL via diode (with attachment information).
Critical values are recomputed differentiably (torch or jax, via eagerpy)
as squared circumradii of each simplex's *attacher* tau (a Gabriel
coface), so gradients flow back to the input point coordinates. Vertices
are immovable: dim-0 values are zeros without grad.
"""
import inspect
import time
from typing import Optional

import numpy as np
import eagerpy as epy

from .. import _oineus
from .._dtype import real_module_for
from ._backend import concrete_numpy
from ._tensor_utils import real_buffer_for
from .alpha_utils import (
    edge_circumradius_sq,
    tetrahedron_circumradius_sq,
    triangle_circumradius_sq,
)
from .diff_filtration import DiffFiltration


_GUARD_RESULT: Optional[bool] = None


def _diode_supports_attachment() -> bool:
    """Return True iff the installed diode exposes ``with_attachment=True``.

    Cached after first call. Falls back from signature introspection (which
    pybind11 sometimes hides) to a small probe call.
    """
    global _GUARD_RESULT
    if _GUARD_RESULT is not None:
        return _GUARD_RESULT
    try:
        import diode
    except ImportError:
        _GUARD_RESULT = False
        return _GUARD_RESULT

    try:
        sig = inspect.signature(diode.fill_alpha_shapes)
        if "with_attachment" in sig.parameters:
            _GUARD_RESULT = True
            return _GUARD_RESULT
    except (ValueError, TypeError):
        pass

    probe_points = np.array([[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
    try:
        result = diode.fill_alpha_shapes(probe_points, with_attachment=True)
    except TypeError:
        _GUARD_RESULT = False
        return _GUARD_RESULT
    except Exception:
        _GUARD_RESULT = False
        return _GUARD_RESULT

    _GUARD_RESULT = bool(result) and len(result[0]) == 3
    return _GUARD_RESULT


def _bucket_indices_by_tau_dim(tau_rows):
    """Group sigma-row indices by tau dimension.

    Returns: dict[tau_dim -> (indices: list[int], tau_arr: list[tuple[int]])]
    where tau_dim is len(tau)-1 and indices are positions within the dim block.
    """
    buckets = {}
    for i, t in enumerate(tau_rows):
        td = len(t) - 1
        bucket = buckets.setdefault(td, ([], []))
        bucket[0].append(i)
        bucket[1].append(t)
    return buckets


def _compute_values_for_dim(points, sigma_rows, tau_by_sigma_tuple, eps):
    """Compute the differentiable critical-value tensor for one dim block.

    points: eagerpy tensor of the input coordinates.
    sigma_rows: numpy ``(n_d, d+1)`` of vertex indices in Oineus sorted order.
    tau_by_sigma_tuple: dict mapping ``tuple(sigma_verts)`` -> ``tuple(tau_verts)``.

    Returns a 1-D eagerpy tensor of shape ``(n_d,)`` with autograd connected
    back through ``points`` for every nonzero entry.
    """
    n_d = sigma_rows.shape[0]
    sigma_tuples = [tuple(int(v) for v in row) for row in sigma_rows]
    tau_rows = [tau_by_sigma_tuple[s] for s in sigma_tuples]
    buckets = _bucket_indices_by_tau_dim(tau_rows)

    # Every sigma has exactly one tau, so the buckets partition the rows:
    # concatenate the per-bucket values and apply the inverse permutation
    # (a pure gather, backend-neutral) instead of scattering into zeros
    pieces = []
    positions = []
    for tau_dim, (indices, taus) in buckets.items():
        if not indices:
            continue
        tau_arr = np.asarray(taus, dtype=np.int64)
        if tau_dim == 0:
            vals = epy.zeros(points, len(indices))
        elif tau_dim == 1:
            vals = epy.astensor(edge_circumradius_sq(
                points[tau_arr[:, 0]], points[tau_arr[:, 1]]))
        elif tau_dim == 2:
            vals = epy.astensor(triangle_circumradius_sq(
                points[tau_arr[:, 0]], points[tau_arr[:, 1]],
                points[tau_arr[:, 2]], eps))
        elif tau_dim == 3:
            vals = epy.astensor(tetrahedron_circumradius_sq(
                points[tau_arr[:, 0]], points[tau_arr[:, 1]],
                points[tau_arr[:, 2]], points[tau_arr[:, 3]], eps))
        else:
            raise RuntimeError(f"alpha_filtration: tau_dim={tau_dim} not supported")
        pieces.append(vals)
        positions.append(np.asarray(indices, dtype=np.int64))

    if not pieces:
        return epy.zeros(points, 0)
    order = np.concatenate(positions)
    inv = np.empty(n_d, dtype=np.int64)
    inv[order] = np.arange(n_d, dtype=np.int64)
    return epy.concatenate(pieces)[inv]


def alpha_filtration(points, eps: float = 1e-12, exact: bool = False,
                     print_time: bool = False) -> DiffFiltration:
    """Build a differentiable alpha filtration from a point cloud.

    Combinatorics and per-simplex *attacher* (a Gabriel coface tau whose
    squared circumradius equals alpha(sigma)) are obtained from diode
    (CGAL, via ``fill_alpha_shapes(..., with_attachment=True)``). Critical
    values are recomputed differentiably as squared circumradii of tau, so
    gradients flow back to ``points``.

    Vertices are immovable: dim-0 values are zeros without grad.

    Args:
        points: ``(n, d)`` torch tensor or jax array with ``d in {2, 3}``.
            Differentiable; the returned values are in the same framework.
        eps: small value for numerical stability in the closed-form formulas.
        exact: forwarded to diode (selects the exact CGAL kernel).
        print_time: if True, print per-stage timings.

    Returns:
        DiffFiltration whose ``values`` tensor matches CGAL's alpha values
        and is wired into the autograd graph.

    Raises:
        RuntimeError if the installed diode does not support
        ``with_attachment=True``.
    """
    if not _diode_supports_attachment():
        raise RuntimeError(
            "alpha_filtration requires a build of diode that exposes the "
            "`with_attachment` keyword argument in `fill_alpha_shapes`. "
            "Rebuild diode from the branch that adds attachment information."
        )

    import diode  # known to be importable since the guard passed

    if print_time:
        t0 = time.time()

    tensor = epy.astensor(points)
    points_np = concrete_numpy(points)
    triples = diode.fill_alpha_shapes(points_np, exact=exact, with_attachment=True)

    if print_time:
        print(f"diode fill_alpha_shapes elapsed: {time.time() - t0:.3f}")
        t0 = time.time()

    pairs = [(s, a) for s, a, _ in triples]
    # Route to the backend matching the point dtype so a float32 cloud builds a genuine float32
    # under-filtration (diode's alpha values are double; the float32 _Filtration ctor narrows
    # them). The recomputed values below already follow the points dtype, so the whole
    # DiffFiltration stays single-dtype instead of float32 values on a float64 filtration.
    sub = real_module_for(points_np)
    alpha_fil = sub._Filtration(pairs, duplicates_possible=False, n_threads=1)
    alpha_fil.kind = _oineus.FiltrationKind.Alpha

    if print_time:
        print(f"build _oineus._Filtration elapsed: {time.time() - t0:.3f}")
        t0 = time.time()

    # Diode may emit sigma vertices unsorted; Oineus stores them sorted (Simplex
    # ctor at include/oineus/simplex.h:116-128). Key the lookup on sorted tuples
    # so it matches the rows returned by get_edges()/get_triangles()/etc.
    tau_by_sigma_tuple = {tuple(sorted(int(v) for v in s)): tuple(int(v) for v in t)
                          for s, _, t in triples}
    if len(tau_by_sigma_tuple) != len(triples):
        raise RuntimeError(
            "alpha_filtration: diode returned duplicate sigma simplices; "
            "duplicates are not supported in non-periodic mode"
        )

    if print_time:
        print(f"build tau_by_sigma_tuple elapsed: {time.time() - t0:.3f}")

    n_v = alpha_fil.size_in_dimension(0)
    values_in_dim = [epy.zeros(tensor, n_v)]

    for dim in range(1, alpha_fil.max_dim + 1):
        if print_time:
            t_dim = time.time()
        if dim == 1:
            sigma_rows = alpha_fil.get_edges()
        elif dim == 2:
            sigma_rows = alpha_fil.get_triangles()
        elif dim == 3:
            sigma_rows = alpha_fil.get_tetrahedra()
        else:
            raise RuntimeError(f"alpha_filtration: dim={dim} not supported")
        sigma_rows = sigma_rows.astype(np.int64)
        vals = _compute_values_for_dim(tensor, sigma_rows, tau_by_sigma_tuple, eps)
        values_in_dim.append(vals)
        if print_time:
            print(f"dim {dim} elapsed: {time.time() - t_dim:.3f}")

    if print_time:
        t0 = time.time()

    cd_vals = epy.concatenate(values_in_dim)
    # contiguous buffer in the filtration's Real dtype -- read directly by set_values
    alpha_fil.set_values(real_buffer_for(alpha_fil, cd_vals.raw))
    sorted_vals = epy.concatenate([epy.sort(v) for v in values_in_dim]).raw

    if print_time:
        print(f"finalize elapsed: {time.time() - t0:.3f}")

    return DiffFiltration(alpha_fil, sorted_vals)
