#!/usr/bin/env python3
"""
Correctness tests for the in-order (VRE) Vietoris-Rips construction.

oineus.vr_filtration() now uses VRE unconditionally; here we cross-check it
against the brute-force ``_oineus._get_vr_filtration_naive`` reference (a
straight enumeration of all subsets, max_dim <= 3) on small inputs. The
C++ test in tests_reduction.cpp also compares VRE against the legacy
Bron-Kerbosch construction.
"""

import pytest
import numpy as np
import oineus as oin
from oineus import _oineus


def _cell_set(fil, ndigits: int = 12):
    """Canonical (uid, value) set for a filtration; tolerant to FP rounding."""
    return {(c.uid, round(c.value, ndigits)) for c in fil.cells()}


def _vr_naive(pts, max_dim, max_diameter):
    """Reference brute-force VR (max_dim <= 3 enforced by the C++ side)."""
    return _oineus._get_vr_filtration_naive(
        pts, max_dim=max_dim, max_diameter=max_diameter, n_threads=1)


@pytest.mark.parametrize("n_points,dim,max_dim", [
    (10, 2, 1),
    (10, 2, 2),
    (10, 3, 3),
    (15, 2, 2),
    (15, 3, 3),
    (20, 2, 2),
])
def test_vr_matches_naive_points(n_points, dim, max_dim):
    """vr_filtration matches the brute-force reference on point clouds."""
    rng = np.random.default_rng(42)
    pts = rng.random((n_points, dim)).astype(np.float64)
    # Use a permissive but explicit threshold so both paths see the same value.
    thr = float(np.linalg.norm(pts[:, None] - pts[None, :], axis=-1).max()) + 1e-6
    fil = oin.vr_filtration(pts, max_dim=max_dim, max_diameter=thr)
    fil_naive = _vr_naive(pts, max_dim=max_dim, max_diameter=thr)
    assert fil.size() == fil_naive.size()
    assert _cell_set(fil) == _cell_set(fil_naive)


@pytest.mark.parametrize("n_points,dim,max_dim,thr", [
    (15, 2, 2, 0.5),
    (20, 3, 2, 0.4),
    (10, 2, 3, 0.6),
])
def test_vr_matches_naive_thresholded(n_points, dim, max_dim, thr):
    """Match under a non-trivial diameter cutoff."""
    rng = np.random.default_rng(123)
    pts = rng.random((n_points, dim)).astype(np.float64)
    fil = oin.vr_filtration(pts, max_dim=max_dim, max_diameter=thr)
    fil_naive = _vr_naive(pts, max_dim=max_dim, max_diameter=thr)
    assert fil.size() == fil_naive.size()
    assert _cell_set(fil) == _cell_set(fil_naive)


@pytest.mark.parametrize("n_points,dim,max_dim", [
    (15, 2, 2),
    (20, 3, 3),
])
def test_vr_critical_edges_have_correct_length(n_points, dim, max_dim):
    """Each returned critical edge has length equal to its simplex's value."""
    rng = np.random.default_rng(7)
    pts = rng.random((n_points, dim)).astype(np.float64)
    fil, edges = oin.vr_filtration(pts, max_dim=max_dim,
                                   with_critical_edges=True)
    cells = list(fil.cells())
    assert len(cells) == len(edges)
    for c, e in zip(cells, edges):
        if c.dim == 0:
            continue
        u, v = e
        d = float(np.linalg.norm(pts[u] - pts[v]))
        np.testing.assert_allclose(d, c.value, rtol=1e-9, atol=1e-12)


def test_vr_pwdists_preserves_user_distances():
    """Dist-matrix path: filtration values must equal the user's distances
    bit-for-bit. Earlier code went through sqrt(d*d), which loses ~1 ULP
    for normalized inputs and overflows/underflows at the FP extremes."""
    rng = np.random.default_rng(99)
    pts = rng.random((12, 3)).astype(np.float64)
    pwd = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)

    fil = oin.vr_filtration(pwd, from_pwdists=True, max_dim=2,
                            max_diameter=10.0)
    for c in fil.cells():
        if c.dim == 0:
            continue
        verts = list(c.vertices)
        # The simplex's filtration value must be the maximum pwd[i, j] over
        # its vertex pairs, *equal to the input value*, not sqrt(d*d).
        true_max = max(pwd[u, v] for u in verts for v in verts if u < v)
        assert c.value == true_max, (
            f"value mismatch on simplex {verts}: "
            f"stored {c.value!r}, expected {true_max!r}")


def test_vr_pwdists_matches_points():
    """Pairwise-distance and point-cloud paths produce the same filtration on
    the same data."""
    rng = np.random.default_rng(11)
    pts = rng.random((20, 3)).astype(np.float64)
    pwd = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)

    fil_pts = oin.vr_filtration(pts, max_dim=2, max_diameter=0.7)
    fil_pwd = oin.vr_filtration(pwd, from_pwdists=True, max_dim=2,
                                max_diameter=0.7)
    assert fil_pts.size() == fil_pwd.size()
    assert _cell_set(fil_pts) == _cell_set(fil_pwd)


def test_vr_ties_unit_square():
    """Corner case: corners of a unit square -- many ties at distance 1, plus
    two diagonals at sqrt(2). Result must contain the full 2-skeleton plus
    the tetrahedron."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                   dtype=np.float64)
    fil = oin.vr_filtration(pts, max_dim=3, max_diameter=10.0)
    fil_naive = _vr_naive(pts, max_dim=3, max_diameter=10.0)
    # full VR on 4 points: 4 + 6 + 4 + 1 = 15
    assert fil.size() == 15
    assert fil_naive.size() == 15
    assert _cell_set(fil) == _cell_set(fil_naive)


def test_vr_edge_cases():
    """Single point, max_dim = 0, and a too-small max_diameter."""
    # Single point: max_distance() helper assumes >= 2 points, so pass
    # max_diameter explicitly.
    pts = np.array([[0.0, 0.0]], dtype=np.float64)
    fil = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0)
    assert fil.size() == 1

    # max_dim = 0: only vertices come out, regardless of distances.
    pts = np.random.default_rng(0).random((5, 2)).astype(np.float64)
    fil = oin.vr_filtration(pts, max_dim=0)
    assert fil.size() == 5

    # Tiny threshold: no edges qualify, so only vertices.
    fil = oin.vr_filtration(pts, max_dim=2, max_diameter=1e-12)
    assert fil.size() == 5


def test_vr_max_dim_one_only_edges():
    """max_dim=1 returns vertices + edges only -- exercises the layer-1
    code path without invoking the cofacet generation."""
    rng = np.random.default_rng(5)
    pts = rng.random((20, 2)).astype(np.float64)
    fil = oin.vr_filtration(pts, max_dim=1, max_diameter=10.0)
    fil_naive = _vr_naive(pts, max_dim=1, max_diameter=10.0)
    assert fil.size() == fil_naive.size()
    assert _cell_set(fil) == _cell_set(fil_naive)
    for c in fil.cells():
        assert c.dim in (0, 1)


def test_vr_persistence_diagram_matches_naive():
    """End-to-end: persistence diagrams from VRE match those from the naive
    reference. This exercises the presorted Filtration ctor's index
    bookkeeping (id_to_sorted_id_, uid_to_sorted_id, dim_first_/dim_last_)
    against the original sort-based ctor used by the naive path."""
    rng = np.random.default_rng(31)
    pts = rng.random((25, 2)).astype(np.float64)
    thr = 0.5

    fil_vre = oin.vr_filtration(pts, max_dim=2, max_diameter=thr)
    fil_naive = _vr_naive(pts, max_dim=2, max_diameter=thr)

    dcmp_vre = oin.Decomposition(fil_vre, True)
    dcmp_vre.reduce(oin.ReductionParams())
    dgm_vre = dcmp_vre.diagram(fil=fil_vre, include_inf_points=False)

    dcmp_naive = oin.Decomposition(fil_naive, True)
    dcmp_naive.reduce(oin.ReductionParams())
    dgm_naive = dcmp_naive.diagram(fil=fil_naive, include_inf_points=False)

    for d in (0, 1):
        pts_vre = sorted(tuple(p) for p in dgm_vre.in_dimension(d))
        pts_naive = sorted(tuple(p) for p in dgm_naive.in_dimension(d))
        assert pts_vre == pts_naive, (
            f"diagram mismatch in dimension {d}: "
            f"VRE has {len(pts_vre)} points, naive has {len(pts_naive)}")


# ----------------------------------------------------------------------------
# Heavier test: opt-in via -m slow. Bound: n=80 in 2D, max_dim=2 (naive
# permits this), no cutoff. Worst-case simplex count
# = C(80,1)+C(80,2)+C(80,3) = 80 + 3160 + 82160 = 85400, ~10 MB at
# ~120 B/simplex. Both VRE and naive produce the same count; well below
# 1 GB.
# ----------------------------------------------------------------------------
@pytest.mark.slow
def test_vr_matches_naive_medium():
    rng = np.random.default_rng(2024)
    pts = rng.random((80, 2)).astype(np.float64)
    fil = oin.vr_filtration(pts, max_dim=2, max_diameter=10.0)
    fil_naive = _vr_naive(pts, max_dim=2, max_diameter=10.0)
    assert fil.size() == fil_naive.size()
    assert _cell_set(fil) == _cell_set(fil_naive)
