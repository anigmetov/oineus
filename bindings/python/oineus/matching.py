"""Matchings between persistence diagrams (Wasserstein and bottleneck).

The public result objects follow a grouped-views layout:

    m = oineus.wasserstein_matching(dgm_a, dgm_b)
    m.finite_to_finite              # ndarray (n, 2)
    m.a_to_diagonal                 # ndarray (n,)
    m.b_to_diagonal                 # ndarray (n,)
    m.essential.inf_death           # ndarray (n, 2), (finite, +inf) family
    m.essential.neg_inf_death       # (finite, -inf)
    m.essential.inf_birth           # (+inf, finite)
    m.essential.neg_inf_birth       # (-inf, finite)
    m.essential.items()             # ("inf_death", arr), ("neg_inf_death", arr), ...
    m.essential[InfKind.INF_DEATH]  # enum-indexed access also works

    bm = oineus.bottleneck_matching(dgm_a, dgm_b)
    bm.longest.finite               # list[FiniteLongestEdge]
    bm.longest.essential.inf_death  # list[EssentialLongestEdge]
    bm.longest.essential.items()    # iterate all four families

Indices in every array are positions in the original input diagrams (before
any internal split into finite / essential).

Efficiency note: the per-family essential matching loop and a few list
comprehensions in this module are linear in the number of essential points
of one family. Bulk vectorization (and pushing more work into C++) is the
next step; for million-point diagrams the dominant cost today is still in
Hera and the C++ matching, which remain unchanged.
"""

from __future__ import annotations

import dataclasses
import enum
import typing

import numpy as np

from . import _oineus
from ._dtype import REAL_DTYPE, as_real_numpy


# ---------------------------------------------------------------------------
# Essential-family enum and ordered name list
# ---------------------------------------------------------------------------

class InfKind(enum.Enum):
    """The four families of essential (infinite-coordinate) diagram points."""

    INF_DEATH      = "inf_death"      # (finite, +inf)
    NEG_INF_DEATH  = "neg_inf_death"  # (finite, -inf)
    INF_BIRTH      = "inf_birth"      # (+inf, finite)
    NEG_INF_BIRTH  = "neg_inf_birth"  # (-inf, finite)

    def __str__(self) -> str:
        return self.value


# Canonical iteration order for all grouped views. Mirrors InfKind member order.
_ESSENTIAL_NAMES: typing.Tuple[str, ...] = tuple(k.value for k in InfKind)


def _normalize_essential_key(key) -> str:
    """Coerce InfKind / str inputs into the canonical attribute name."""
    if isinstance(key, InfKind):
        return key.value
    if isinstance(key, str) and key in _ESSENTIAL_NAMES:
        return key
    raise KeyError(
        f"Unknown essential family {key!r}; expected one of {_ESSENTIAL_NAMES} "
        f"or an InfKind member."
    )


# ---------------------------------------------------------------------------
# Grouped views
# ---------------------------------------------------------------------------

class _GroupedView:
    """Mixin: dict-like access over the four canonical essential families."""

    __slots__ = ()

    def __getitem__(self, key):
        return getattr(self, _normalize_essential_key(key))

    def __iter__(self):
        return iter(_ESSENTIAL_NAMES)

    def __len__(self) -> int:
        return len(_ESSENTIAL_NAMES)

    def __contains__(self, key) -> bool:
        try:
            _normalize_essential_key(key)
        except KeyError:
            return False
        return True

    def keys(self):
        return iter(_ESSENTIAL_NAMES)

    def values(self):
        return (getattr(self, n) for n in _ESSENTIAL_NAMES)

    def items(self):
        return ((n, getattr(self, n)) for n in _ESSENTIAL_NAMES)


class EssentialMatches(_GroupedView):
    """Index-pair matches between essential points, grouped by family.

    Each attribute is an ``ndarray`` of shape ``(n, 2)`` with dtype int64
    holding ``(idx_a, idx_b)`` rows. Empty families are ``(0, 2)`` arrays.
    """

    __slots__ = _ESSENTIAL_NAMES

    def __init__(self):
        empty = np.zeros((0, 2), dtype=np.int64)
        for n in _ESSENTIAL_NAMES:
            setattr(self, n, empty)

    def __repr__(self) -> str:
        sizes = ", ".join(f"{n}={len(getattr(self, n))}" for n in _ESSENTIAL_NAMES)
        return f"EssentialMatches({sizes})"


class EssentialLongestEdges(_GroupedView):
    """Tied-longest edges within each essential family (bottleneck only).

    Each attribute is a ``list[EssentialLongestEdge]``; empty families
    are empty lists.
    """

    __slots__ = _ESSENTIAL_NAMES

    def __init__(self):
        for n in _ESSENTIAL_NAMES:
            setattr(self, n, [])

    def __repr__(self) -> str:
        sizes = ", ".join(f"{n}={len(getattr(self, n))}" for n in _ESSENTIAL_NAMES)
        return f"EssentialLongestEdges({sizes})"


class LongestEdges:
    """Bottleneck longest-edge data, split into finite and essential parts."""

    __slots__ = ("finite", "essential")

    def __init__(self):
        self.finite: typing.List[FiniteLongestEdge] = []
        self.essential = EssentialLongestEdges()

    def __repr__(self) -> str:
        ess_total = sum(len(v) for v in self.essential.values())
        return f"LongestEdges(finite={len(self.finite)}, essential={ess_total})"


# ---------------------------------------------------------------------------
# Edge records
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FiniteLongestEdge:
    """A longest edge in the finite part of a bottleneck matching.

    Exactly one of ``idx_a`` / ``idx_b`` is None when the corresponding
    endpoint is a diagonal projection of the other side.
    """

    length: float
    idx_a: typing.Optional[int]
    idx_b: typing.Optional[int]
    point_a: typing.Tuple[float, float]
    point_b: typing.Tuple[float, float]


@dataclasses.dataclass
class EssentialLongestEdge:
    """A longest edge inside one essential family.

    Essential-family edges pair two real diagram points by sorting their
    single finite coordinate; both ``idx_a`` and ``idx_b`` are set and
    a single scalar coordinate per side is recorded. The family is the
    attribute name on ``BottleneckMatching.longest.essential``.
    """

    length: float
    idx_a: int
    idx_b: int
    coord_a: float
    coord_b: float


# ---------------------------------------------------------------------------
# Top-level matching classes
# ---------------------------------------------------------------------------

class DiagramMatching:
    """Optimal Wasserstein matching between two persistence diagrams.

    Attributes
    ----------
    finite_to_finite : ndarray, shape (n, 2), int64
        Pairs of indices ``(idx_a, idx_b)`` for finite points matched to
        each other.
    a_to_diagonal : ndarray, shape (n,), int64
        Indices in ``dgm_a`` of finite points matched to the diagonal.
    b_to_diagonal : ndarray, shape (n,), int64
        Indices in ``dgm_b`` of finite points matched to the diagonal.
    essential : EssentialMatches
        Grouped view of essential-point matches, with attributes
        ``inf_death``, ``neg_inf_death``, ``inf_birth``, ``neg_inf_birth``.
    cost : float
        Total Wasserstein cost (distance ** q).
    distance : float
        Wasserstein distance.
    """

    def __init__(self):
        self.finite_to_finite: np.ndarray = np.zeros((0, 2), dtype=np.int64)
        self.a_to_diagonal: np.ndarray = np.zeros((0,), dtype=np.int64)
        self.b_to_diagonal: np.ndarray = np.zeros((0,), dtype=np.int64)
        self.essential: EssentialMatches = EssentialMatches()
        self.cost: float = 0.0
        self.distance: float = 0.0

    def __repr__(self) -> str:
        ess_total = sum(len(v) for v in self.essential.values())
        return (f"{type(self).__name__}("
                f"finite_to_finite={len(self.finite_to_finite)}, "
                f"a_to_diagonal={len(self.a_to_diagonal)}, "
                f"b_to_diagonal={len(self.b_to_diagonal)}, "
                f"essential={ess_total}, "
                f"distance={self.distance:.4f})")


class BottleneckMatching(DiagramMatching):
    """Optimal bottleneck matching, with longest-edge data.

    Inherits the ``finite_to_finite`` / ``a_to_diagonal`` / ``b_to_diagonal``
    / ``essential`` / ``cost`` / ``distance`` attributes from
    :class:`DiagramMatching`, and adds:

    Attributes
    ----------
    longest : LongestEdges
        - ``longest.finite`` : list of :class:`FiniteLongestEdge` tied for
          the maximum length in the finite part of the matching, regardless
          of whether the global bottleneck is attained there.
        - ``longest.essential`` : :class:`EssentialLongestEdges` grouped view
          of tied-longest edges within each essential family.
    """

    def __init__(self):
        super().__init__()
        self.longest: LongestEdges = LongestEdges()

    def __repr__(self) -> str:
        ess_total = sum(len(v) for v in self.essential.values())
        ess_long = sum(len(v) for v in self.longest.essential.values())
        return (f"BottleneckMatching("
                f"finite_to_finite={len(self.finite_to_finite)}, "
                f"a_to_diagonal={len(self.a_to_diagonal)}, "
                f"b_to_diagonal={len(self.b_to_diagonal)}, "
                f"essential={ess_total}, "
                f"finite_longest={len(self.longest.finite)}, "
                f"essential_longest={ess_long}, "
                f"distance={self.distance:.4f})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _split_with_indices(dgm: np.ndarray):
    """Vectorized split of a diagram into finite + four essential families.

    Returns ``(finite_pts, essential_coords, finite_indices, essential_indices)``:
    - ``finite_pts``: ``(n_fin, 2)`` array of finite off-diagonal points
      in the working real dtype.
    - ``essential_coords``: dict ``name -> 1D array`` of the single finite
      coordinate, keyed by the canonical names in ``_ESSENTIAL_NAMES``.
    - ``finite_indices``: 1D int64 indices into the input diagram for the
      finite rows in ``finite_pts``.
    - ``essential_indices``: dict ``name -> 1D int64 array`` of original
      input indices for each essential family.
    """
    n = dgm.shape[0]
    if n == 0:
        empty_f = np.empty((0,), dtype=REAL_DTYPE)
        empty_i = np.empty((0,), dtype=np.int64)
        return (
            np.zeros((0, 2), dtype=REAL_DTYPE),
            {k: empty_f for k in _ESSENTIAL_NAMES},
            empty_i,
            {k: empty_i for k in _ESSENTIAL_NAMES},
        )

    b = dgm[:, 0]
    d = dgm[:, 1]
    finite_b = np.isfinite(b)
    finite_d = np.isfinite(d)

    # Skip diagonal points (b == d) when collecting "finite" off-diagonal pts.
    finite_mask = finite_b & finite_d & (b != d)
    finite_indices = np.flatnonzero(finite_mask).astype(np.int64, copy=False)
    finite_pts = np.ascontiguousarray(dgm[finite_mask].astype(REAL_DTYPE, copy=False))

    masks = {
        "inf_death":     finite_b & np.isposinf(d),
        "neg_inf_death": finite_b & np.isneginf(d),
        "inf_birth":     np.isposinf(b) & finite_d,
        "neg_inf_birth": np.isneginf(b) & finite_d,
    }
    coord_source = {
        "inf_death":     b,
        "neg_inf_death": b,
        "inf_birth":     d,
        "neg_inf_birth": d,
    }

    essential_coords = {
        n_: coord_source[n_][m].astype(REAL_DTYPE, copy=False)
        for n_, m in masks.items()
    }
    essential_indices = {
        n_: np.flatnonzero(m).astype(np.int64, copy=False) for n_, m in masks.items()
    }
    return finite_pts, essential_coords, finite_indices, essential_indices


def _to_matching_array(dgm, dim):
    """Coerce a diagram input (numpy / list / oineus.Diagrams) to ``(n, 2)`` array."""
    # Local import to avoid circular dependency with oineus/__init__.py.
    from . import _check_numpy_diagram_shape

    if hasattr(dgm, "in_dimension"):  # oineus.Diagrams
        if dim is None:
            raise ValueError("When passing oineus.Diagrams, specify dim=...")
        return dgm.in_dimension(dim, as_numpy=True)
    return as_real_numpy(_check_numpy_diagram_shape(dgm))


def _match_essential_family(coords_1, coords_2, indices_1, indices_2,
                            *, with_longest: bool):
    """Match one essential family by sorting its single finite coordinate.

    Returns ``(pairs, longest_list)``:
    - ``pairs``: ``(k, 2)`` int64 array of original-index pairs.
    - ``longest_list``: list of :class:`EssentialLongestEdge` for the
      tied-longest edges, or ``[]`` when ``with_longest`` is False.
    Raises ``ValueError`` on cardinality mismatch.
    """
    n1, n2 = len(coords_1), len(coords_2)
    if n1 != n2:
        raise ValueError(
            f"Essential point cardinalities must match. "
            f"Got {n1} and {n2} points."
        )
    if n1 == 0:
        return np.zeros((0, 2), dtype=np.int64), []

    order_1 = np.argsort(coords_1)
    order_2 = np.argsort(coords_2)

    pairs = np.column_stack([
        np.asarray(indices_1, dtype=np.int64)[order_1],
        np.asarray(indices_2, dtype=np.int64)[order_2],
    ])

    if not with_longest:
        return pairs, []

    sorted_1 = np.asarray(coords_1)[order_1]
    sorted_2 = np.asarray(coords_2)[order_2]
    lengths = np.abs(sorted_1 - sorted_2)
    max_len = float(lengths.max())
    tied = np.flatnonzero(lengths == max_len)

    longest = [
        EssentialLongestEdge(
            length=float(lengths[k]),
            idx_a=int(pairs[k, 0]),
            idx_b=int(pairs[k, 1]),
            coord_a=float(sorted_1[k]),
            coord_b=float(sorted_2[k]),
        )
        for k in tied
    ]
    return pairs, longest


def _populate_essential(out_matches: EssentialMatches,
                        ess_coords_1, ess_coords_2,
                        ess_idx_1, ess_idx_2,
                        *,
                        out_longest: typing.Optional[EssentialLongestEdges] = None):
    """Fill an ``EssentialMatches`` (and optionally an ``EssentialLongestEdges``)."""
    for name in _ESSENTIAL_NAMES:
        pairs, longest = _match_essential_family(
            ess_coords_1[name], ess_coords_2[name],
            ess_idx_1[name], ess_idx_2[name],
            with_longest=out_longest is not None,
        )
        setattr(out_matches, name, pairs)
        if out_longest is not None:
            setattr(out_longest, name, longest)


def _translate_finite_pairs(cpp_a_to_b, cpp_b_from_a, finite_indices_1, finite_indices_2):
    """Translate parallel C++ index arrays into an ``(n, 2)`` int64 pair array."""
    a_to_b = np.asarray(cpp_a_to_b, dtype=np.int64)
    b_from_a = np.asarray(cpp_b_from_a, dtype=np.int64)
    if a_to_b.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack([finite_indices_1[a_to_b], finite_indices_2[b_from_a]])


def _translate_diag_indices(cpp_indices, finite_indices):
    """Translate one C++ to-diagonal index list into original-diagram indices."""
    arr = np.asarray(cpp_indices, dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return finite_indices[arr]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def wasserstein_matching(
    dgm_1,
    dgm_2,
    q: float = 2.0,
    delta: float = 0.01,
    internal_p: float = np.inf,
    wasserstein_q: typing.Optional[float] = None,
    ignore_inf_points: bool = True,
    dim: typing.Optional[int] = None,
) -> DiagramMatching:
    """Compute the optimal Wasserstein matching between two persistence diagrams."""
    from . import wasserstein_distance

    dgm_1 = _to_matching_array(dgm_1, dim)
    dgm_2 = _to_matching_array(dgm_2, dim)

    if wasserstein_q is not None:
        q = wasserstein_q

    finite_1, ess_coords_1, fi1, ess_idx_1 = _split_with_indices(dgm_1)
    finite_2, ess_coords_2, fi2, ess_idx_2 = _split_with_indices(dgm_2)

    result = DiagramMatching()

    if not ignore_inf_points:
        _populate_essential(result.essential,
                            ess_coords_1, ess_coords_2,
                            ess_idx_1, ess_idx_2)

    if len(finite_1) > 0 or len(finite_2) > 0:
        internal_p_hera = -1.0 if np.isinf(internal_p) else internal_p
        cpp_matching = _oineus.wasserstein_matching_finite(
            finite_1, finite_2,
            wasserstein_q=q,
            wasserstein_delta=delta,
            internal_p=internal_p_hera,
        )
        result.finite_to_finite = _translate_finite_pairs(
            cpp_matching.a_to_b, cpp_matching.b_from_a, fi1, fi2)
        result.a_to_diagonal = _translate_diag_indices(cpp_matching.a_to_diag, fi1)
        result.b_to_diagonal = _translate_diag_indices(cpp_matching.b_to_diag, fi2)

    result.distance = wasserstein_distance(
        dgm_1, dgm_2, q=q, delta=delta, internal_p=internal_p, check_for_zero=False,
    )
    result.cost = result.distance ** q
    return result


def bottleneck_matching(
    dgm_1,
    dgm_2,
    *,
    delta: float = 0.01,
    ignore_inf_points: bool = True,
    dim: typing.Optional[int] = None,
) -> BottleneckMatching:
    """Compute the optimal bottleneck matching between two persistence diagrams.

    Parameters mirror :func:`wasserstein_matching` (without ``q`` and
    ``internal_p``). ``delta`` is Hera's relative-error parameter; ``delta=0.0``
    routes to the exact algorithm. Essential-family cardinality mismatch (when
    ``ignore_inf_points=False``) raises ValueError.

    Note
    ----
    Hera's bottleneck is hardcoded to the L-infinity plane metric; there is no
    ``internal_p`` parameter here.
    """
    dgm_1 = _to_matching_array(dgm_1, dim)
    dgm_2 = _to_matching_array(dgm_2, dim)

    finite_1, ess_coords_1, fi1, ess_idx_1 = _split_with_indices(dgm_1)
    finite_2, ess_coords_2, fi2, ess_idx_2 = _split_with_indices(dgm_2)

    result = BottleneckMatching()

    if not ignore_inf_points:
        _populate_essential(result.essential,
                            ess_coords_1, ess_coords_2,
                            ess_idx_1, ess_idx_2,
                            out_longest=result.longest.essential)

    d_finite = 0.0
    if len(finite_1) > 0 or len(finite_2) > 0:
        cpp_result = _oineus.bottleneck_matching_finite(
            finite_1, finite_2, delta=delta,
        )
        d_finite = float(cpp_result.distance)

        result.finite_to_finite = _translate_finite_pairs(
            cpp_result.a_to_b, cpp_result.b_from_a, fi1, fi2)
        result.a_to_diagonal = _translate_diag_indices(cpp_result.a_to_diag, fi1)
        result.b_to_diagonal = _translate_diag_indices(cpp_result.b_to_diag, fi2)

        # Translate BottleneckLongestEdge -> FiniteLongestEdge with original indices.
        # NOTE (efficiency): per-edge Python loop. Tied longest edges are usually
        # few in number (often a handful), so this is not a real hot path.
        for le in cpp_result.longest_edges:
            orig_a = int(fi1[le.idx_a]) if le.idx_a >= 0 else None
            orig_b = int(fi2[le.idx_b]) if le.idx_b >= 0 else None
            result.longest.finite.append(
                FiniteLongestEdge(
                    length=float(le.length),
                    idx_a=orig_a,
                    idx_b=orig_b,
                    point_a=(float(le.a_x), float(le.a_y)),
                    point_b=(float(le.b_x), float(le.b_y)),
                )
            )

    # Essential contribution to the overall bottleneck distance.
    d_essential = 0.0
    for edges in result.longest.essential.values():
        for e in edges:
            if e.length > d_essential:
                d_essential = e.length

    result.distance = max(d_finite, d_essential)
    result.cost = result.distance  # bottleneck has no q-exponent
    return result
