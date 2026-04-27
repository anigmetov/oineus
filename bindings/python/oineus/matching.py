"""Matchings between persistence diagrams (Wasserstein and bottleneck).

The result types live entirely in C++ and are bound by nanobind:

    - ``oineus.DiagramMatching``  (= ``hera::WassersteinMatching``)
    - ``oineus.BottleneckMatching`` (subclass; adds ``longest`` view)
    - ``oineus.EssentialMatches``      — grouped view (m.essential)
    - ``oineus.LongestEdges``          — bottleneck longest-edge namespace
    - ``oineus.EssentialLongestEdges`` — per-family longest edges
    - ``oineus.FiniteLongestEdge`` / ``oineus.EssentialLongestEdge``
    - ``oineus.InfKind`` enum (the four essential families)

This module's job is just (a) coercing user-friendly diagram inputs into
``(n, 2)`` numpy arrays and (b) calling the new detailed Hera entry points.
"""

from __future__ import annotations

import typing

import numpy as np

from . import _oineus
from ._dtype import REAL_DTYPE, as_real_numpy

# Re-export the C++ types under their Python names; importers can use
# either ``oineus.X`` or ``oineus.matching.X``.
DiagramMatching         = _oineus.DiagramMatching
BottleneckMatching      = _oineus.BottleneckMatching
EssentialMatches        = _oineus.EssentialMatches
EssentialLongestEdges   = _oineus.EssentialLongestEdges
LongestEdges            = _oineus.LongestEdges
FiniteLongestEdge       = _oineus.FiniteLongestEdge
EssentialLongestEdge    = _oineus.EssentialLongestEdge
InfKind                 = _oineus.InfKind


def point_to_diagonal(dgm, indices=None):
    """Get diagonal projection coordinates for points in a persistence diagram.

    The diagonal projection of a point ``(b, d)`` is ``((b+d)/2, (b+d)/2)``.

    Parameters
    ----------
    dgm : array-like or list of DiagramPoint
        Persistence diagram as ``(n_points, 2)`` array or list of
        ``DiagramPoint`` objects.
    indices : array-like or None, default=None
        Indices of points to project. If None, projects all points.

    Returns
    -------
    ndarray or list
        If dgm is numpy array: ``(n, 2)`` array. If dgm is list: list of
        ``(x, x)`` tuples.
    """
    if isinstance(dgm, np.ndarray):
        if indices is None:
            b = dgm[:, 0]
            d = dgm[:, 1]
        else:
            indices = np.asarray(indices)
            b = dgm[indices, 0]
            d = dgm[indices, 1]
        diag_coords = (b + d) / 2.0
        return np.column_stack([diag_coords, diag_coords])

    if isinstance(dgm, list):
        if indices is None:
            indices = range(len(dgm))
        result = []
        for i in indices:
            pt = dgm[i]
            if hasattr(pt, "birth") and hasattr(pt, "death"):
                b, d = pt.birth, pt.death
            else:
                b, d = float(pt[0]), float(pt[1])
            diag_coord = (b + d) / 2.0
            result.append((diag_coord, diag_coord))
        return result

    if indices is None:
        indices = range(len(dgm))
    result = []
    for i in indices:
        pt = dgm[i]
        if hasattr(pt, "birth") and hasattr(pt, "death"):
            b, d = pt.birth, pt.death
        else:
            b, d = float(pt[0]), float(pt[1])
        diag_coord = (b + d) / 2.0
        result.append((diag_coord, diag_coord))
    return result


def _to_matching_array(dgm):
    """Coerce a single-dimension diagram input to a ``(n, 2)`` ndarray.

    Accepts: numpy ``(n, 2)`` arrays and ``list[DiagramPoint]`` (e.g.
    the output of ``Diagrams.in_dimension(d, as_numpy=False)``).
    Multi-dimensional ``oineus.Diagrams`` is rejected — the caller must
    extract a single dimension first via ``dgm.in_dimension(d)``.
    """
    # Local import to avoid circular dependency with oineus/__init__.py.
    from . import _check_numpy_diagram_shape

    if hasattr(dgm, "in_dimension"):  # oineus.Diagrams
        raise TypeError(
            "Pass a single-dimension diagram: use `dgm.in_dimension(d)` "
            "instead of passing the multi-dimensional Diagrams object.")
    if isinstance(dgm, list):
        if len(dgm) == 0:
            return np.zeros((0, 2), dtype=REAL_DTYPE)
        if hasattr(dgm[0], "birth") and hasattr(dgm[0], "death"):
            return np.array([(p.birth, p.death) for p in dgm], dtype=REAL_DTYPE)
    return as_real_numpy(_check_numpy_diagram_shape(dgm))


def wasserstein_matching(
    dgm_1,
    dgm_2,
    q: float = 2.0,
    delta: float = 0.01,
    internal_p: float = np.inf,
    ignore_inf_points: bool = True,
) -> DiagramMatching:
    """Compute the optimal q-Wasserstein matching between two persistence diagrams.

    Parameters
    ----------
    dgm_1, dgm_2 : array-like or list[DiagramPoint]
        Single-dimension persistence diagrams: a ``(n_points, 2)`` numpy
        array of (birth, death), or a ``list[DiagramPoint]``. To pass an
        ``oineus.Diagrams`` object, extract the dimension first via
        ``dgm.in_dimension(d)``.
    q : float, default 2.0
        Wasserstein exponent. ``distance == cost ** (1/q)``.
    delta : float, default 0.01
        Relative error parameter for Hera. Must be strictly positive
        (the auction has no exact mode).
    internal_p : float, default np.inf
        Ground metric in the (birth, death) plane. ``np.inf`` selects the
        L_infinity norm.
    ignore_inf_points : bool, default True
        If True, essential (infinite-coordinate) points are dropped. If False,
        each of the four essential families must have equal cardinality on
        both sides; otherwise ValueError is raised.

    Returns
    -------
    DiagramMatching
        Matching object with attributes ``finite_to_finite`` (ndarray
        ``(n, 2)``), ``a_to_diagonal``, ``b_to_diagonal`` (1-D ndarrays),
        ``essential`` (an :class:`EssentialMatches` grouped view) and the
        scalar ``distance``, ``cost`` fields.

    Raises
    ------
    ValueError
        If essential cardinalities differ in any of the four families when
        ``ignore_inf_points=False``.

    Examples
    --------
    >>> import numpy as np
    >>> import oineus
    >>> dgm_a = np.array([[0.0, 1.0], [0.5, 2.0]])
    >>> dgm_b = np.array([[0.1, 0.9], [0.6, 1.8]])
    >>> m = oineus.wasserstein_matching(dgm_a, dgm_b, q=2.0)
    >>> m.finite_to_finite.tolist()
    [[0, 0], [1, 1]]

    >>> dgm_a = np.array([[0.0, 1.0], [0.5, np.inf]])
    >>> dgm_b = np.array([[0.1, 0.9], [0.6, np.inf]])
    >>> m = oineus.wasserstein_matching(dgm_a, dgm_b, ignore_inf_points=False)
    >>> m.essential.inf_death.tolist()
    [[1, 1]]
    >>> m.essential[oineus.InfKind.INF_DEATH].tolist()
    [[1, 1]]
    """
    dgm_1 = _to_matching_array(dgm_1)
    dgm_2 = _to_matching_array(dgm_2)
    internal_p_hera = -1.0 if np.isinf(internal_p) else internal_p
    return _oineus.wasserstein_matching_detailed(
        dgm_1, dgm_2,
        wasserstein_q=q,
        wasserstein_delta=delta,
        internal_p=internal_p_hera,
        ignore_inf_points=ignore_inf_points,
    )


def bottleneck_matching(
    dgm_1,
    dgm_2,
    *,
    delta: float = 0.01,
    ignore_inf_points: bool = True,
) -> BottleneckMatching:
    """Compute the optimal bottleneck matching between two persistence diagrams.

    Parameters mirror :func:`wasserstein_matching` (without ``q`` and
    ``internal_p``). ``delta=0.0`` runs Hera's exact algorithm. Essential-family
    cardinality mismatch (when ``ignore_inf_points=False``) raises ValueError.

    Examples
    --------
    >>> import numpy as np, oineus
    >>> m = oineus.bottleneck_matching(np.array([[0.0, 5.0]]),
    ...                                np.array([[1.0, 6.0]]), delta=0.0)
    >>> m.distance
    1.0
    >>> len(m.longest.finite)
    1
    """
    dgm_1 = _to_matching_array(dgm_1)
    dgm_2 = _to_matching_array(dgm_2)
    return _oineus.bottleneck_matching_detailed(
        dgm_1, dgm_2,
        delta=delta,
        ignore_inf_points=ignore_inf_points,
    )
