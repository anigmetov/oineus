"""Sliced Wasserstein distance between persistence diagrams (numpy, non-differentiable).

This is the plain-numpy counterpart of ``oineus.diff.sliced_wasserstein_distance``.
The sliced Wasserstein distance approximates the Wasserstein distance by
averaging one-dimensional optimal-transport costs of the diagrams projected
onto random directions on the half-circle, with each finite point augmented by
the diagonal projection of the points of the opposite diagram.

Hera (the C++ backend behind ``oineus.wasserstein_distance``) has no sliced
variant, so this fills that gap for users who want a fast surrogate metric
without pulling in torch.

Inputs are single-dimension diagrams in the same forms accepted by
``oineus.wasserstein_distance`` (an ``(n, 2)`` numpy array or a
``list[DiagramPoint]``); pass ``dgm.in_dimension(d)`` to extract one dimension
from a multi-dimensional ``oineus.Diagrams`` object.
"""

import operator

import numpy as np

_ESSENTIAL_NAMES = ("(finite, +inf)", "(finite, -inf)", "(+inf, finite)", "(-inf, finite)")


def _coerce_pair(dgm_1, dgm_2):
    # Reuse the package-level coercion (rejects multi-dim Diagrams, accepts
    # numpy arrays and lists of DiagramPoint). Imported lazily to avoid an
    # import cycle with the package __init__.
    from . import _prepare_distance_args
    a, b = _prepare_distance_args(dgm_1, dgm_2)
    return np.asarray(a, dtype=np.float64).reshape(-1, 2), np.asarray(b, dtype=np.float64).reshape(-1, 2)


def _project_to_diagonal(points):
    if points.shape[0] == 0:
        return points
    t = (points[:, 0] + points[:, 1]) / 2.0
    return np.stack([t, t], axis=1)


def _split_finite_essential(dgm):
    """Split a diagram into finite points and the finite coordinates of the four
    essential families. Points with both coordinates infinite are discarded."""
    empty = np.empty((0,), dtype=np.float64)
    if dgm.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), (empty, empty, empty, empty)

    births, deaths = dgm[:, 0], dgm[:, 1]
    is_finite = np.isfinite(births) & np.isfinite(deaths)
    finite = dgm[is_finite]

    both_inf = (~np.isfinite(births)) & (~np.isfinite(deaths))
    is_ess = (~is_finite) & (~both_inf)
    ess = dgm[is_ess]
    if ess.shape[0] == 0:
        return finite, (empty, empty, empty, empty)

    eb, ed = ess[:, 0], ess[:, 1]
    coords = (
        eb[np.isfinite(eb) & np.isposinf(ed)],
        eb[np.isfinite(eb) & np.isneginf(ed)],
        ed[np.isposinf(eb) & np.isfinite(ed)],
        ed[np.isneginf(eb) & np.isfinite(ed)],
    )
    return finite, coords


def _match_essential_1d(ess1, ess2, q=1.0):
    if ess1.shape[0] != ess2.shape[0]:
        raise ValueError(f"Essential point cardinalities must match: {ess1.shape[0]} vs {ess2.shape[0]}")
    if ess1.shape[0] == 0:
        return 0.0
    return float(np.sum(np.abs(np.sort(ess1) - np.sort(ess2)) ** q))


def _sample_unit_directions(n_directions, rng):
    angles = rng.random(n_directions) * np.pi
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _validate_n_directions(n_directions):
    try:
        n = operator.index(n_directions)
    except TypeError as exc:
        raise TypeError("n_directions must be an integer") from exc
    if n <= 0:
        raise ValueError("n_directions must be positive")
    return n


def _prepare_directions(n_directions, seed, directions):
    if directions is None:
        return _sample_unit_directions(
            _validate_n_directions(n_directions),
            np.random.default_rng(seed),
        )

    U = np.asarray(directions, dtype=np.float64).reshape(-1, 2)
    if U.shape[0] == 0:
        raise ValueError("directions must contain at least one direction")
    return U


def _slice_costs_standard(fin1, fin2, U):
    """Per-direction sliced cost, diagonal points participating symmetrically."""
    n1, n2 = fin1.shape[0], fin2.shape[0]
    if n1 == 0 and n2 == 0:
        return np.zeros(U.shape[0], dtype=np.float64)

    # errstate guards against spurious matmul FP warnings when numpy runs in an
    # MKL/OpenMP-perturbed environment (e.g. after `import torch`)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        proj1 = fin1 @ U.T
        proj2 = fin2 @ U.T
        proj1_diag = _project_to_diagonal(fin2) @ U.T
        proj2_diag = _project_to_diagonal(fin1) @ U.T

        L1 = np.concatenate([proj1, proj1_diag], axis=0)
        L2 = np.concatenate([proj2, proj2_diag], axis=0)
        L1s = np.sort(L1, axis=0)
        L2s = np.sort(L2, axis=0)
        return np.sum(np.abs(L1s - L2s), axis=0)


def _slice_costs_corrected(fin1, fin2, U):
    """Per-direction diagonal-corrected sliced cost: a point matched to a
    diagonal projection is charged against its own diagonal projection, and two
    diagonal projections cost nothing."""
    n1, n2 = fin1.shape[0], fin2.shape[0]
    if n1 == 0 and n2 == 0:
        return np.zeros(U.shape[0], dtype=np.float64)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if n1 == 0:
            return np.sum(np.abs(fin2 @ U.T - _project_to_diagonal(fin2) @ U.T), axis=0)
        if n2 == 0:
            return np.sum(np.abs(fin1 @ U.T - _project_to_diagonal(fin1) @ U.T), axis=0)

        proj1 = fin1 @ U.T
        proj2 = fin2 @ U.T
        proj1_self_diag = _project_to_diagonal(fin1) @ U.T
        proj2_self_diag = _project_to_diagonal(fin2) @ U.T

        L1 = np.concatenate([proj1, _project_to_diagonal(fin2) @ U.T], axis=0)
        L2 = np.concatenate([proj2, _project_to_diagonal(fin1) @ U.T], axis=0)

        idx1 = np.argsort(L1, axis=0)
        idx2 = np.argsort(L2, axis=0)
        L1s = np.take_along_axis(L1, idx1, axis=0)
        L2s = np.take_along_axis(L2, idx2, axis=0)

        is_diag1 = idx1 >= n1
        is_diag2 = idx2 >= n2
        costs = np.abs(L1s - L2s)

        real_idx1 = np.clip(idx1, 0, n1 - 1)
        case3 = np.abs(np.take_along_axis(proj1, real_idx1, axis=0)
                       - np.take_along_axis(proj1_self_diag, real_idx1, axis=0))
        costs = np.where((~is_diag1) & is_diag2, case3, costs)

        real_idx2 = np.clip(idx2, 0, n2 - 1)
        case2 = np.abs(np.take_along_axis(proj2, real_idx2, axis=0)
                       - np.take_along_axis(proj2_self_diag, real_idx2, axis=0))
        costs = np.where(is_diag1 & (~is_diag2), case2, costs)

        costs = np.where(is_diag1 & is_diag2, 0.0, costs)
        return np.sum(costs, axis=0)


def _sliced_wasserstein(dgm_1, dgm_2, slice_fn, n_directions, ignore_inf_points, seed, directions):
    fin1, fin2 = _coerce_pair(dgm_1, dgm_2)
    fin1, ess1 = _split_finite_essential(fin1)
    fin2, ess2 = _split_finite_essential(fin2)

    total = 0.0
    if not ignore_inf_points:
        for c1, c2, name in zip(ess1, ess2, _ESSENTIAL_NAMES):
            if c1.shape[0] != c2.shape[0]:
                raise ValueError(
                    f"Essential point cardinalities must match. "
                    f"Got {c1.shape[0]} and {c2.shape[0]} points with {name}.")
            total += _match_essential_1d(c1, c2)

    if fin1.shape[0] == 0 and fin2.shape[0] == 0:
        return float(total)

    U = _prepare_directions(n_directions, seed, directions)

    total += float(slice_fn(fin1, fin2, U).mean())
    return float(total)


def sliced_wasserstein_distance(dgm_1, dgm_2, n_directions: int = 100,
                                ignore_inf_points: bool = False, seed=None, directions=None):
    """Sliced Wasserstein distance between two single-dimension diagrams.

    Standard form: diagonal projections of the opposite diagram are added to
    each side and participate symmetrically in the rank matching.

    Args:
        dgm_1, dgm_2: Single-dimension diagrams ((n, 2) numpy array or
            list[DiagramPoint]).
        n_directions: Number of random projection directions (ignored if
            ``directions`` is given).
        ignore_inf_points: If True, drop essential points and only match the
            finite part.
        seed: Seed for the random directions (forwarded to
            ``numpy.random.default_rng``); use it for reproducible results.
        directions: Optional explicit ``(n_directions, 2)`` array of directions;
            overrides ``n_directions``/``seed``. Useful for deterministic runs
            or comparing two distances under identical directions.

    Returns:
        The sliced Wasserstein distance as a Python float.
    """
    return _sliced_wasserstein(dgm_1, dgm_2, _slice_costs_standard,
                               n_directions, ignore_inf_points, seed, directions)


def sliced_wasserstein_distance_diag_corrected(dgm_1, dgm_2, n_directions: int = 100,
                                               ignore_inf_points: bool = False, seed=None, directions=None):
    """Diagonal-corrected sliced Wasserstein distance.

    Makes the sliced distance behave like true Wasserstein at the diagonal. The
    1D rank-matching can pair an off-diagonal point p with the diagonal
    projection of a *different* point p'; true Wasserstein never does this (such
    skew edges can always be straightened to ``p <-> diag(p)`` without raising
    the cost). The correction re-charges those matches:

    - A point matched to a diagonal slot is charged ``|proj(p) -
      proj(diag(p))|``, its distance to *its own* diagonal projection -- not to
      whichever point's diagonal stand-in the sort aligned it with.
    - A match between two diagonal stand-ins costs zero.

    Arguments are as in :func:`sliced_wasserstein_distance`.
    """
    return _sliced_wasserstein(dgm_1, dgm_2, _slice_costs_corrected,
                               n_directions, ignore_inf_points, seed, directions)
