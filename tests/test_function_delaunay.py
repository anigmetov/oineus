"""Tests for the two-value function-Delaunay bifiltration.

Validation strategy:
- collapse theorem (Alonso-Kerber-Lam-Lesnick): the slice diagrams must
  equal the alpha diagrams of the corresponding sublevel point sets;
- the KICR diagrams of L -> K must equal those of the independently built
  full-Cech pair Cech(A) subseteq Cech(X) (same union-of-balls inclusion);
- structural invariants: subcomplex by uid with equal values, downward
  closure, determinism, insertion-order independence, bfs == setdiff.
"""

from itertools import combinations

import numpy as np
import pytest

import oineus as oin

pytest.importorskip("scipy")


def diagrams_of(fil):
    dcmp = oin.Decomposition(fil, False)
    dcmp.reduce(oin.ReductionParams())
    return dcmp.diagram(fil=fil, include_inf_points=True)


def multiset_close(a, b, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if len(a) == 0:
        return True
    return oin.bottleneck_distance(a, b, delta=0.0) <= tol


def cells_of(fil):
    return {tuple(sorted(s.vertices)): s.value for s in fil.cells()}


def random_two_species(n, d, seed, frac_a=0.5):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, d))
    labels = (rng.random(n) >= frac_a).astype(int)
    # need at least one atom of each species
    labels[0], labels[1] = 0, 1
    return pts, labels


@pytest.mark.parametrize("d,n,seed", [(2, 50, 0), (3, 45, 1)])
def test_collapse_theorem_slice_diagrams(d, n, seed):
    pytest.importorskip("diode")
    pts, labels = random_two_species(n, d, seed)
    K, L = oin.function_delaunay_bifiltration(pts, labels, seed=0)
    dg_K, dg_L = diagrams_of(K), diagrams_of(L)
    dg_all = oin.compute_diagrams_alpha(pts)
    dg_a = oin.compute_diagrams_alpha(pts[labels == 0])
    for q in range(d):
        assert multiset_close(dg_K.in_dimension(q), dg_all.in_dimension(q)), (d, q, "K")
        assert multiset_close(dg_L.in_dimension(q), dg_a.in_dimension(q)), (d, q, "L")


@pytest.mark.parametrize("d,n,seed", [(2, 40, 2), (3, 30, 3)])
def test_subcomplex_by_uid_with_equal_values(d, n, seed):
    pts, labels = random_two_species(n, d, seed)
    K, L = oin.function_delaunay_bifiltration(pts, labels, seed=0)
    k_cells = cells_of(K)
    l_cells = cells_of(L)
    is_a = labels == 0
    for simplex, val in l_cells.items():
        assert all(is_a[v] for v in simplex)
        assert simplex in k_cells
        assert k_cells[simplex] == val
    # L is exactly the all-A part of K
    n_all_a = sum(1 for s in k_cells if all(is_a[v] for v in s))
    assert len(l_cells) == n_all_a


@pytest.mark.parametrize("d,n,seed", [(2, 16, 4), (3, 14, 5)])
def test_kicr_matches_full_cech_pair(d, n, seed):
    """End-to-end oracle: kernel/image/cokernel diagrams of the
    bifiltration pair equal those of the independently built full-Cech
    pair Cech(A) subseteq Cech(X) -- both model the same inclusion of
    union-of-balls."""
    pts, labels = random_two_species(n, d, seed)
    idx_a = np.flatnonzero(labels == 0)

    K_fd, L_fd = oin.function_delaunay_bifiltration(pts, labels, seed=0)

    R = oin.max_distance(pts)
    K_cech = oin.cech_filtration(pts, max_radius=R)
    L_cech = oin.cech_filtration(pts[idx_a], max_radius=R, vertex_ids=idx_a)

    params = oin.KICRParams()
    kicr_fd = oin.compute_kernel_image_cokernel_reduction(K_fd, L_fd, params)
    kicr_cech = oin.compute_kernel_image_cokernel_reduction(K_cech, L_cech, params)

    for q in range(d):
        for family in ("kernel_diagrams", "image_diagrams", "cokernel_diagrams"):
            a = getattr(kicr_fd, family)().in_dimension(q)
            b = getattr(kicr_cech, family)().in_dimension(q)
            assert multiset_close(a, b), (d, q, family, np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("d,n,seed", [(2, 35, 6), (3, 25, 7)])
def test_closure_and_monotone(d, n, seed):
    pts, labels = random_two_species(n, d, seed)
    K, _ = oin.function_delaunay_bifiltration(pts, labels, seed=0)
    values = cells_of(K)
    for simplex, val in values.items():
        if len(simplex) == 1:
            continue
        for facet in combinations(simplex, len(simplex) - 1):
            assert facet in values
            assert values[facet] <= val * (1 + 1e-12)


def test_determinism_and_order_independence():
    pts, labels = random_two_species(40, 2, 8)
    K1, L1 = oin.function_delaunay_bifiltration(pts, labels, seed=0)
    K2, L2 = oin.function_delaunay_bifiltration(pts, labels, seed=0)
    assert cells_of(K1) == cells_of(K2)
    assert cells_of(L1) == cells_of(L2)

    # different intra-group insertion order -> generally a different I(X),
    # but the same persistence (each slice is weakly equivalent to a fixed
    # union of balls) and the same KICR diagrams
    K3, L3 = oin.function_delaunay_bifiltration(pts, labels, seed=12345)
    params = oin.KICRParams()
    kicr_a = oin.compute_kernel_image_cokernel_reduction(K1, L1, params)
    kicr_b = oin.compute_kernel_image_cokernel_reduction(K3, L3, params)
    for q in range(2):
        for family in ("kernel_diagrams", "image_diagrams", "cokernel_diagrams"):
            a = getattr(kicr_a, family)().in_dimension(q)
            b = getattr(kicr_b, family)().in_dimension(q)
            assert multiset_close(a, b), (q, family)


@pytest.mark.parametrize("d", [2, 3])
def test_bfs_equals_setdiff(d):
    pts, labels = random_two_species(20, d, 9)
    K1, L1 = oin.function_delaunay_bifiltration(pts, labels, seed=0, conflict_method="bfs")
    K2, L2 = oin.function_delaunay_bifiltration(pts, labels, seed=0, conflict_method="setdiff")
    assert cells_of(K1) == cells_of(K2)
    assert cells_of(L1) == cells_of(L2)


def test_jittered_lattice_runs():
    # NaCl-like 3D lattice: heavily cospherical, needs jitter
    side = 4
    g = np.arange(side)
    xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
    labels = (pts.sum(axis=1) % 2).astype(int)
    K, L = oin.function_delaunay_bifiltration(pts, labels, jitter=0.01, seed=0)
    assert L.size() < K.size()
    params = oin.KICRParams()
    kicr = oin.compute_kernel_image_cokernel_reduction(K, L, params)
    for q in range(3):
        assert kicr.image_diagrams().in_dimension(q) is not None
    mb = oin.mixup_barcodes_of_filtrations(K, L, max_dim=2)
    assert np.isfinite(mb.total_mixup(0))


def test_mixup_runs_on_bifiltration():
    pts, labels = random_two_species(50, 2, 10)
    K, L = oin.function_delaunay_bifiltration(pts, labels, seed=0)
    mb = oin.mixup_barcodes_of_filtrations(K, L, max_dim=1)
    for q in (0, 1):
        assert np.isfinite(mb.total_mixup(q))


def test_input_validation():
    rng = np.random.default_rng(11)
    pts = rng.random((10, 2))
    with pytest.raises(ValueError):
        oin.function_delaunay_bifiltration(pts, np.zeros(10, dtype=int))  # one label value
    with pytest.raises(ValueError):
        oin.function_delaunay_bifiltration(pts, np.arange(10) % 3)  # three label values
    with pytest.raises(ValueError):
        oin.function_delaunay_bifiltration(pts, np.arange(9) % 2)  # wrong length
    with pytest.raises(ValueError):
        oin.function_delaunay_bifiltration(rng.random((10, 4)), np.arange(10) % 2)  # d=4
    with pytest.raises(ValueError):
        oin.function_delaunay_bifiltration(pts, np.arange(10) % 2, max_dim=3)  # max_dim > d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
