#!/usr/bin/env python3
"""
Correctness tests for the in-order (VRE) Vietoris-Rips construction.

Compares the VRE algorithm against the existing Bron-Kerbosch (BK) construction
on small inputs (sub-second runtime, well under 50 MB memory) plus one slightly
heavier case marked @pytest.mark.slow.

The point-based BK and VRE paths must produce IDENTICAL filtrations (same set
of (uid, value) pairs). The pairwise-distance path is checked using explicit
max_diameter values, because the existing BK distance-matrix path has a
pre-existing bug (vietoris_rips.h:266, :351 compare raw distance against
max_diameter * max_diameter -- this is independent of VRE and tracked as a
follow-up). VRE pwdists is cross-checked against VRE points instead.
"""

import pytest
import numpy as np
import oineus as oin


def _cell_set(fil, ndigits: int = 12):
    """Canonical (uid, value) set for a filtration; tolerant to FP rounding."""
    return {(c.uid, round(c.value, ndigits)) for c in fil.cells()}


@pytest.mark.parametrize("n_points,dim,max_dim", [
    (10, 2, 1),
    (10, 2, 2),
    (10, 3, 3),
    (30, 2, 2),
    (30, 3, 3),
    (50, 2, 2),
])
def test_vre_matches_bk_points(n_points, dim, max_dim):
    """VRE and BK produce identical filtrations on point clouds."""
    rng = np.random.default_rng(42)
    pts = rng.random((n_points, dim)).astype(np.float64)
    fil_bk = oin.vr_filtration(pts, max_dim=max_dim, algorithm="bron-kerbosch")
    fil_vre = oin.vr_filtration(pts, max_dim=max_dim, algorithm="inorder")
    assert fil_bk.size() == fil_vre.size()
    assert _cell_set(fil_bk) == _cell_set(fil_vre)


@pytest.mark.parametrize("n_points,dim,max_dim,thr", [
    (15, 2, 2, 0.5),
    (20, 3, 2, 0.4),
    (10, 2, 3, 0.6),
])
def test_vre_matches_bk_points_thresholded(n_points, dim, max_dim, thr):
    """VRE and BK match under a non-trivial diameter cutoff."""
    rng = np.random.default_rng(123)
    pts = rng.random((n_points, dim)).astype(np.float64)
    fil_bk = oin.vr_filtration(pts, max_dim=max_dim, max_diameter=thr,
                               algorithm="bron-kerbosch")
    fil_vre = oin.vr_filtration(pts, max_dim=max_dim, max_diameter=thr,
                                algorithm="inorder")
    assert fil_bk.size() == fil_vre.size()
    assert _cell_set(fil_bk) == _cell_set(fil_vre)


@pytest.mark.parametrize("n_points,dim,max_dim", [
    (15, 2, 2),
    (20, 3, 3),
])
def test_vre_critical_edges_have_correct_length(n_points, dim, max_dim):
    """Each returned critical edge has length equal to its simplex's value."""
    rng = np.random.default_rng(7)
    pts = rng.random((n_points, dim)).astype(np.float64)
    fil, edges = oin.vr_filtration(pts, max_dim=max_dim,
                                   with_critical_edges=True,
                                   algorithm="inorder")
    cells = list(fil.cells())
    assert len(cells) == len(edges)
    for c, e in zip(cells, edges):
        if c.dim == 0:
            continue
        u, v = e
        d = float(np.linalg.norm(pts[u] - pts[v]))
        np.testing.assert_allclose(d, c.value, rtol=1e-9, atol=1e-12)


def test_vre_pwdists_matches_points():
    """VRE on a pairwise distance matrix matches VRE on points (the same data
    in two forms). We use an explicit threshold to bypass the BK pwdists bug
    flagged in the plan; the comparison here is VRE-pwd vs VRE-pts, which is
    the right invariant."""
    rng = np.random.default_rng(11)
    pts = rng.random((20, 3)).astype(np.float64)
    pwd = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)

    fil_pts = oin.vr_filtration(pts, max_dim=2, max_diameter=0.7,
                                algorithm="inorder")
    fil_pwd = oin.vr_filtration(pwd, from_pwdists=True, max_dim=2,
                                max_diameter=0.7, algorithm="inorder")
    assert fil_pts.size() == fil_pwd.size()
    assert _cell_set(fil_pts) == _cell_set(fil_pwd)


def test_vre_ties_unit_square():
    """Corner case: corners of a unit square -- many ties at distance 1, plus
    two diagonals at sqrt(2). VRE and BK must agree exactly."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                   dtype=np.float64)
    fil_bk = oin.vr_filtration(pts, max_dim=3, algorithm="bron-kerbosch")
    fil_vre = oin.vr_filtration(pts, max_dim=3, algorithm="inorder")
    # full VR on 4 points: 4 + 6 + 4 + 1 = 15
    assert fil_bk.size() == 15
    assert fil_vre.size() == 15
    assert _cell_set(fil_bk) == _cell_set(fil_vre)


def test_vre_edge_cases():
    """Single point, max_dim = 0, and a too-small max_diameter."""
    # Single point: max_distance() helper assumes >= 2 points, so pass
    # max_diameter explicitly.
    pts = np.array([[0.0, 0.0]], dtype=np.float64)
    fil = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0,
                            algorithm="inorder")
    assert fil.size() == 1

    # max_dim = 0: only vertices come out, regardless of distances.
    pts = np.random.default_rng(0).random((5, 2)).astype(np.float64)
    fil = oin.vr_filtration(pts, max_dim=0, algorithm="inorder")
    assert fil.size() == 5

    # Tiny threshold: no edges qualify, so only vertices.
    fil = oin.vr_filtration(pts, max_dim=2, max_diameter=1e-12,
                            algorithm="inorder")
    assert fil.size() == 5


def test_vre_max_dim_one_only_edges():
    """max_dim=1 should return vertices + edges only -- exercises the layer-1
    code path without entering Cases I/II/III."""
    rng = np.random.default_rng(5)
    pts = rng.random((20, 2)).astype(np.float64)
    fil_bk = oin.vr_filtration(pts, max_dim=1, algorithm="bron-kerbosch")
    fil_vre = oin.vr_filtration(pts, max_dim=1, algorithm="inorder")
    assert fil_bk.size() == fil_vre.size()
    assert _cell_set(fil_bk) == _cell_set(fil_vre)
    # all simplices have dim 0 or 1
    for c in fil_vre.cells():
        assert c.dim in (0, 1)


# ----------------------------------------------------------------------------
# Heavier test: opt-in via -m slow. Bound: n=80 in 2D, max_dim=2, no cutoff.
# Worst-case simplex count = C(80,1) + C(80,2) + C(80,3) = 80 + 3160 + 82160
# = 85400, ~10 MB at ~120 B/simplex. Well below 1 GB.
# ----------------------------------------------------------------------------
@pytest.mark.slow
def test_vre_matches_bk_medium():
    rng = np.random.default_rng(2024)
    pts = rng.random((80, 2)).astype(np.float64)
    fil_bk = oin.vr_filtration(pts, max_dim=2, algorithm="bron-kerbosch")
    fil_vre = oin.vr_filtration(pts, max_dim=2, algorithm="inorder")
    assert fil_bk.size() == fil_vre.size()
    assert _cell_set(fil_bk) == _cell_set(fil_vre)
