"""Example: persistence-diagram matchings (Wasserstein and bottleneck).

Five short scenarios that demonstrate ``oineus.wasserstein_matching``,
``oineus.bottleneck_matching`` and ``oineus.plot_matching``.

Section 1   Wasserstein matching of 5 vs. 4 finite points.
            Three obvious cross matches; the remainder match to the
            diagonal. Manually recompute the cost from the matching
            indices and compare against ``oineus.wasserstein_distance``.

Section 2   Bottleneck matching where Hera lazily matches most points
            to the diagonal because the bottleneck is attained by a
            single far-from-diagonal pair.

Section 3   Bottleneck matching with three tied longest edges, to
            illustrate ``BottleneckMatching.longest.finite`` carrying
            multiple entries.

Section 4   Diagrams with one essential ``(finite, +inf)`` point each;
            shows how to read essential matches via attribute access on
            ``m.essential.inf_death``.

Section 5   Sanity check: the matching functions accept native Oineus
            ``Diagrams`` objects (with ``dim=...``) and also lists of
            ``DiagramPoint``, in addition to numpy arrays.

Run as a script (``python example_matching.py``) or convert to a
notebook for the docs.
"""

import numpy as np
import matplotlib.pyplot as plt

import oineus


def matching_cost(matching, dgm_a, dgm_b, *, q, internal_p=np.inf):
    """Recompute the Wasserstein cost from a matching object.

    Provided for illustration; in real code just use
    ``matching.distance ** q`` (or call ``oineus.wasserstein_distance``).
    """
    cost = 0.0

    def _norm(diffs):
        if np.isinf(internal_p):
            return np.max(np.abs(diffs), axis=1)
        return np.linalg.norm(diffs, ord=internal_p, axis=1)

    if matching.finite_to_finite.size:
        ia = matching.finite_to_finite[:, 0]
        ib = matching.finite_to_finite[:, 1]
        cost += float(np.sum(_norm(dgm_a[ia] - dgm_b[ib]) ** q))

    for indices, dgm in ((matching.a_to_diagonal, dgm_a),
                         (matching.b_to_diagonal, dgm_b)):
        if indices.size:
            pts = dgm[indices]
            mid = 0.5 * (pts[:, 0] + pts[:, 1])
            cost += float(np.sum(_norm(pts - np.column_stack([mid, mid])) ** q))

    return cost


def _banner(title):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# Section 1 — Wasserstein matching, 5 vs. 4 finite points
# ---------------------------------------------------------------------------
# Three obvious cross matches (rows 0,1,2 of dgm_a vs. rows 0,1,2 of dgm_b);
# the cross offset is large enough (~0.3) that the matching edges are clearly
# visible in the plot. Rows 3,4 of dgm_a and row 3 of dgm_b are
# medium-persistence points placed far from any partner — they will match
# to their own diagonal projections.

dgm_a = np.array([
    [0.0, 2.0],
    [3.0, 5.0],
    [6.0, 8.0],
    [4.0, 4.5],
    [7.0, 7.4],
])
dgm_b = np.array([
    [ 0.3, 1.7],
    [ 2.7, 5.3],
    [ 5.7, 8.3],
    [10.0, 10.5],
])

q = 2.0
delta = 0.01  # auction is inherently approximate; delta=0 is rejected for Wasserstein.
m = oineus.wasserstein_matching(dgm_a, dgm_b, q=q, delta=delta)

_banner(f"Section 1: Wasserstein matching, q = {q}")
print(repr(m))
print()
print("finite_to_finite:")
for ia, ib in m.finite_to_finite:
    print(f"  a[{ia}]={dgm_a[ia]}  <->  b[{ib}]={dgm_b[ib]}")
print(f"a_to_diagonal: {m.a_to_diagonal.tolist()}  "
      f"(points: {[dgm_a[i].tolist() for i in m.a_to_diagonal]})")
print(f"b_to_diagonal: {m.b_to_diagonal.tolist()}  "
      f"(points: {[dgm_b[i].tolist() for i in m.b_to_diagonal]})")
print()

manual = matching_cost(m, dgm_a, dgm_b, q=q, internal_p=np.inf)
direct = oineus.wasserstein_distance(dgm_a, dgm_b, q=q, delta=delta) ** q
print(f"cost recomputed from matching:        {manual:.6f}")
print(f"cost stored on the matching object:   {m.cost:.6f}")
print(f"oineus.wasserstein_distance ** q:     {direct:.6f}")

fig, ax = plt.subplots(figsize=(5, 5))
oineus.plot_matching(
    dgm_a, dgm_b, m, ax=ax,
    plot_diagonal_projections=True,
    title="Wasserstein matching (5 vs 4 finite points)",
)


# ---------------------------------------------------------------------------
# Section 2 — Bottleneck matching with lazy diagonal matches
# ---------------------------------------------------------------------------
# A single high-persistence pair (rows 0 of each diagram) drives the
# bottleneck: the L_infinity gap between (0.0, 12.0) and (1.0, 11.0) is 1.0.
# Every other point has persistence 0.5 and is placed far from every other
# small point, so any cross-pairing would exceed 1.0 — strictly worse than
# matching to the diagonal at cost 0.25. Hera therefore matches all small
# points to the diagonal.

dgm_a2 = np.array([
    [0.0, 12.0],
    [3.0,  3.5],
    [8.0,  8.5],
])
dgm_b2 = np.array([
    [ 1.0, 11.0],
    [ 5.0,  5.5],
    [10.0, 10.5],
    [14.0, 14.5],
])

m2 = oineus.bottleneck_matching(dgm_a2, dgm_b2, delta=0.0)

_banner("Section 2: bottleneck matching with lazy diagonal matches")
print(repr(m2))
print(f"\nbottleneck distance: {m2.distance:.4f}")
print("\nfinite_to_finite (the one cross edge):")
for ia, ib in m2.finite_to_finite:
    print(f"  a[{ia}]={dgm_a2[ia]} <-> b[{ib}]={dgm_b2[ib]}")
print(f"a_to_diagonal: {m2.a_to_diagonal.tolist()}")
print(f"b_to_diagonal: {m2.b_to_diagonal.tolist()}")
print(f"\nlongest.finite has {len(m2.longest.finite)} edge(s):")
for e in m2.longest.finite:
    print(f"  length={e.length:.4f}  a-index={e.idx_a}  b-index={e.idx_b}")
    print(f"    point_a={e.point_a}  point_b={e.point_b}")

fig, ax = plt.subplots(figsize=(5, 5))
oineus.plot_matching(
    dgm_a2, dgm_b2, m2, ax=ax,
    plot_a_to_diagonal=True, plot_b_to_diagonal=True,
    title="Bottleneck matching (lazy diagonal matches)",
)


# ---------------------------------------------------------------------------
# Section 3 — Bottleneck matching with three tied longest edges
# ---------------------------------------------------------------------------
# Three identically-offset twin pairs: every cross edge has L_infinity
# length exactly 1.0. Persistence is 5 in each diagram, so any diagonal
# projection costs 2.5 — strictly worse. Hera matches all three across,
# and `longest.finite` carries three tied entries.

dgm_a3 = np.array([
    [ 0.0,  5.0],
    [10.0, 15.0],
    [20.0, 25.0],
])
dgm_b3 = np.array([
    [ 1.0,  6.0],
    [11.0, 16.0],
    [21.0, 26.0],
])

m3 = oineus.bottleneck_matching(dgm_a3, dgm_b3, delta=0.0)

_banner("Section 3: bottleneck matching with three tied longest edges")
print(repr(m3))
print(f"\nbottleneck distance: {m3.distance:.4f}")
print(f"longest.finite has {len(m3.longest.finite)} tied edge(s):")
for e in m3.longest.finite:
    print(f"  length={e.length:.4f}  a={e.idx_a} <-> b={e.idx_b}")

fig, ax = plt.subplots(figsize=(5, 5))
oineus.plot_matching(
    dgm_a3, dgm_b3, m3, ax=ax,
    title="Bottleneck matching (three tied longest edges)",
)


# ---------------------------------------------------------------------------
# Section 4 — Essential points: the (finite, +inf) family
# ---------------------------------------------------------------------------
# By far the most common essential family in real data is (finite, +inf):
# a homology class that is born at some finite filtration value and never
# dies (e.g. the connected component of a point cloud). Both diagrams must
# carry the same number of (finite, +inf) points, otherwise the matching
# raises ValueError. Pass ignore_inf_points=False to actually match them;
# essential matches show up under m.essential.inf_death (attribute access).

dgm_a4 = np.array([
    [0.0, 2.0],
    [3.0, 5.0],
    [4.0, np.inf],
])
dgm_b4 = np.array([
    [ 0.3, 1.7],
    [ 2.7, 5.3],
    [ 4.5, np.inf],
])

m4 = oineus.wasserstein_matching(
    dgm_a4, dgm_b4, q=2.0, delta=0.01, ignore_inf_points=False,
)

_banner("Section 4: matching with (finite, +inf) essential points")
print(repr(m4))
print()
print("finite_to_finite:")
for ia, ib in m4.finite_to_finite:
    print(f"  a[{ia}]={dgm_a4[ia]}  <->  b[{ib}]={dgm_b4[ib]}")
print()
print("essential.inf_death  (the (finite, +inf) family):")
for ia, ib in m4.essential.inf_death:
    print(f"  a[{ia}]={dgm_a4[ia]}  <->  b[{ib}]={dgm_b4[ib]}  "
          f"(matched by sorting the finite birth coordinate)")

fig, ax = plt.subplots(figsize=(5, 5))
oineus.plot_matching(
    dgm_a4, dgm_b4, m4, ax=ax,
    plot_essential=True,
    title="Wasserstein matching with one (finite, +inf) essential point",
)


# ---------------------------------------------------------------------------
# Section 5 — Native Oineus diagram inputs (no numpy conversion needed)
# ---------------------------------------------------------------------------
# The matching functions accept single-dimension diagrams as numpy arrays
# (used in sections 1-4) or as list[DiagramPoint] (the result of
# Diagrams.in_dimension(d, as_numpy=False)). To pass a multi-dim
# oineus.Diagrams, extract a single dimension first via in_dimension(d).

rng = np.random.default_rng(0)
points_a = rng.random((20, 2))
points_b = points_a + 0.02

diagrams_a = oineus.compute_diagrams_vr(points_a, max_dim=1)
diagrams_b = oineus.compute_diagrams_vr(points_b, max_dim=1)

# Pass list[DiagramPoint] (in_dimension with as_numpy=False).
list_a = diagrams_a.in_dimension(0, as_numpy=False)
list_b = diagrams_b.in_dimension(0, as_numpy=False)
oineus.wasserstein_matching(list_a, list_b, q=2.0, delta=0.01)
oineus.bottleneck_matching(list_a, list_b, delta=0.0)

plt.show()
