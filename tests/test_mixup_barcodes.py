"""Tests for oineus.mixup_barcodes (Wagner-Arustamyan-Wheeler-Bubenik mixup barcodes).

Oracles, strongest first:

1. A literal transcription of Algorithm 1 of the paper (arXiv:2402.15058):
   take the boundary matrix of K in filtration order, reorder its rows so
   that the cells of L come first, zero out the K-L columns to get BL,
   REDUCE both, and read the triples (b, d', d) off the pivots. It shares
   no code with the implementation (which goes through the C++ KICR
   reduction and uid-based matching), so it pins the *definition*,
   including the index-level matching on tie-heavy inputs.

2. The authors' reference implementation (github.com/hubwag/Mixup-SoCG26,
   hacked-Ripser image persistence): tests/mixup_ref_socg.npz stores a
   50-point distance matrix (10 points of A, 40 of B) shipped in that
   repository together with the standard and image barcodes their pipeline
   computed for it. We recompute the mixup barcode from the raw distance
   matrix and compare bar by bar (dim 0 per birth vertex, dim 1 directly).
   Their files are in radius units (ripser diameters / 2) and tag dim-0
   births with negative per-vertex values; the npz is already converted to
   oineus conventions (diameters, vertex indices recovered from the tags).

3. Hand-traced tiny configurations, fully worked in the comments.

Plus the structural identities stated in the paper (sub-barcodes vs the
standard and image diagrams, total mixup = total persistence - total image
persistence, the subsampling lemma) on random inputs, and edge cases.
"""

import math
import pickle
from pathlib import Path

import numpy as np
import pytest

import oineus as oin

INF = math.inf
SQ2 = math.sqrt(2.0)
SQ05 = math.sqrt(0.5)


# ---------------------------------------------------------------------------
# Oracle 1: literal Algorithm 1 of the paper
# ---------------------------------------------------------------------------

def z2_reduce(columns):
    """Standard left-to-right Z/2 column reduction; columns are sets of rows."""
    R = [set(c) for c in columns]
    low_to_col = {}
    for j in range(len(R)):
        while R[j]:
            lo = max(R[j])
            if lo in low_to_col:
                R[j] ^= R[low_to_col[lo]]
            else:
                low_to_col[lo] = j
                break
    return R


def naive_mixup(K, L, max_dim):
    """Algorithm 1 of the paper, verbatim, on oineus filtrations K, L (L in K).

    Returns dict dim -> (finite (n, 3) value triples, finite (n, 3) index
    triples in K sorted ids, essential (m, 3) value triples), mirroring the
    conventions of the implementation: zero-persistence bars of L dropped,
    a zero-length image sub-bar records the birth cell as its own image
    death cell.
    """
    m = K.size()
    val = [K.cell_value_by_sorted_id(i) for i in range(m)]
    dim_of = [K.cell(i).dim for i in range(m)]
    in_L = [False] * m
    for i in range(L.size()):
        in_L[K.sorted_id_by_uid(L.cell(i).uid)] = True

    # rows of BK reordered: L cells first, then K-L cells, both in K order
    new_to_old = sorted(range(m), key=lambda i: (0 if in_L[i] else 1, i))
    old_to_new = [0] * m
    for r, o in enumerate(new_to_old):
        old_to_new[o] = r

    DK = K.boundary_matrix()
    BK = [{old_to_new[x] for x in col} for col in DK]
    # BL: zero out the entries involving the cells of K-L (their columns;
    # their rows are never hit by an L column since L is a subcomplex)
    BL = [set(c) if in_L[j] else set() for j, c in enumerate(BK)]

    RK = z2_reduce(BK)
    RL = z2_reduce(BL)
    lowK = {max(c): j for j, c in enumerate(RK) if c}
    lowL = {max(c): j for j, c in enumerate(RL) if c}

    out = {d: ([], [], []) for d in range(max_dim + 1)}
    for sigma in range(m):
        if not in_L[sigma] or RL[sigma] or dim_of[sigma] > max_dim:
            continue  # only positive (in L) cells of L in tracked degrees
        d = dim_of[sigma]
        b = val[sigma]
        tau = lowL.get(old_to_new[sigma])
        tau_p = lowK.get(old_to_new[sigma])
        death = val[tau] if tau is not None else INF
        death_p = val[tau_p] if tau_p is not None else INF
        assert death_p <= death
        if tau is None:
            out[d][2].append((b, death_p, INF))
            continue
        if b == death:
            continue  # zero-persistence bar of L: dropped
        # mirror the implementation's index convention for empty image sub-bars
        idx_p = tau_p if death_p > b else sigma
        out[d][0].append((b, min(death_p, death), death))
        out[d][1].append((sigma, idx_p, tau))
    return {d: (np.array(f).reshape(-1, 3), np.array(fi, dtype=np.int64).reshape(-1, 3),
                np.array(e).reshape(-1, 3))
            for d, (f, fi, e) in out.items()}


def sorted_rows(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a.reshape(0, a.shape[-1] if a.ndim == 2 else 0)
    a = a.reshape(len(a), -1)
    return a[np.lexsort(a.T[::-1])]


def assert_rows_equal(a, b, atol=1e-12):
    a, b = sorted_rows(a), sorted_rows(b)
    assert a.shape == b.shape, (a, b)
    assert np.allclose(a, b, atol=atol, equal_nan=True), (a, b)


def make_K_L(A, B, max_dim, max_diameter=None):
    if max_diameter is None:
        max_diameter = float(oin.max_distance(A))
    L = oin.vr_filtration(A, max_dim=max_dim + 1, max_diameter=max_diameter, packed=False)
    K = oin.vr_filtration(np.vstack([A, B]), max_dim=max_dim + 1,
                          max_diameter=max_diameter, packed=False)
    return K, L


# ---------------------------------------------------------------------------
# hand-traced examples
# ---------------------------------------------------------------------------

def test_two_points_on_line():
    # A = {0, 2} on a line, B = {1} in between.
    #
    # L = VR(A): vertices v0, v2 at 0; edge (v0, v2) at value 2.
    # dgm_0(L): essential bar [0, inf) for v0; bar [0, 2) for v2.
    # K = VR(A u B): vertex order v0, v2, vB (A points first); edges
    # (v0, vB) and (v2, vB) at value 1, (v0, v2) at value 2.
    # Image of H_0(L) -> H_0(K): v2's class merges with v0's older class
    # at value 1 (through vB), so its image bar is [0, 1).
    # Mixup triple in degree 0: (0, 1, 2); mixup = 1, percentage = 1/2.
    A = np.array([[0.0], [2.0]])
    B = np.array([[1.0]])
    mb = oin.mixup_barcodes(A, B, max_dim=0)
    assert_rows_equal(mb.in_dimension(0), [[0.0, 1.0, 2.0]])
    assert_rows_equal(mb.essential_in_dimension(0), [[0.0, INF, INF]])
    assert mb.total_mixup(0) == pytest.approx(1.0)
    assert mb.total_persistence(0) == pytest.approx(2.0)
    assert mb.total_mixup_percentage(0) == pytest.approx(0.5)
    assert mb.mean_mixup_percentage(0) == pytest.approx(0.5)


def test_square_with_center():
    # A = unit square corners, B = center point.
    #
    # L = VR(A): 4 side edges at 1, 2 diagonals at sqrt(2), triangles at
    # sqrt(2). dgm_0(L): three bars [0, 1) (side edges merge the corners)
    # + essential. dgm_1(L): the square cycle is born at the 4th side edge
    # (value 1) and dies at sqrt(2) when triangles fill it: bar [1, sqrt(2)).
    #
    # K adds the center c: edges corner-c at sqrt(1/2) < 1, triangles
    # (corner, corner, c) at value 1 (their longest edge is the side).
    # Degree 0: each of the three younger corner classes now merges with an
    # older one already at sqrt(1/2), through c: image bars [0, sqrt(1/2)),
    # so triples (0, sqrt(1/2), 1) x3.
    # Degree 1: the square cycle equals the sum of the boundaries of the four
    # (corner, corner, c) triangles, all present at value 1 = its birth: the
    # image class dies on arrival, image sub-bar empty, triple
    # (1, 1, sqrt(2)) with mixup percentage 1 -- the center point kills the
    # hole of A instantly.
    A = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.5, 0.5]])
    mb = oin.mixup_barcodes(A, B, max_dim=1)

    assert_rows_equal(mb.in_dimension(0), [[0.0, SQ05, 1.0]] * 3)
    assert_rows_equal(mb.in_dimension(1), [[1.0, 1.0, SQ2]])
    assert_rows_equal(mb.essential_in_dimension(0), [[0.0, INF, INF]])
    assert mb.essential_in_dimension(1).shape == (0, 3)

    assert mb.total_mixup(0) == pytest.approx(3 * (1 - SQ05))
    assert mb.total_mixup_percentage(0) == pytest.approx(3 * (1 - SQ05))
    assert mb.mean_mixup_percentage(0) == pytest.approx(1 - SQ05)
    assert mb.total_mixup(1) == pytest.approx(SQ2 - 1)
    assert mb.total_mixup_percentage(1) == pytest.approx(1.0)
    assert mb.mean_mixup_percentage(1) == pytest.approx(1.0)

    # the degenerate image sub-bar records the birth cell as its image death
    idx1 = mb.index_triples_in_dimension(1)
    assert idx1.shape == (1, 3) and idx1[0, 0] == idx1[0, 1]


# ---------------------------------------------------------------------------
# Oracle 1: Algorithm 1 on random and tie-heavy inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,n_A,n_B,max_dim", [
    (1, 12, 6, 1),
    (2, 10, 5, 1),
    (3, 9, 5, 2),
    (4, 8, 8, 2),
])
def test_matches_algorithm1_random(seed, n_A, n_B, max_dim):
    rng = np.random.default_rng(seed)
    A = rng.random((n_A, 3))
    B = rng.random((n_B, 3))
    K, L = make_K_L(A, B, max_dim)

    mb = oin.mixup_barcodes_of_filtrations(K, L, max_dim=max_dim)
    ref = naive_mixup(K, L, max_dim)
    for dim in range(max_dim + 1):
        fin, fin_idx, ess = ref[dim]
        assert_rows_equal(mb.in_dimension(dim), fin)
        assert_rows_equal(mb.index_triples_in_dimension(dim), fin_idx)
        assert_rows_equal(mb.essential_in_dimension(dim), ess)

    # the facade builds the same filtrations
    mb2 = oin.mixup_barcodes(A, B, max_dim=max_dim)
    for dim in range(max_dim + 1):
        assert_rows_equal(mb2.in_dimension(dim), mb.in_dimension(dim))
        assert_rows_equal(mb2.essential_in_dimension(dim), mb.essential_in_dimension(dim))


def test_matches_algorithm1_ties():
    # integer grid points: masses of equal pairwise distances stress the
    # tie-breaking consistency between the orders of L and K
    A = np.array([[x, y] for x in range(3) for y in range(3)], dtype=float)
    B = np.array([[0.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    K, L = make_K_L(A, B, 1)
    mb = oin.mixup_barcodes_of_filtrations(K, L, max_dim=1)
    ref = naive_mixup(K, L, 1)
    for dim in (0, 1):
        assert_rows_equal(mb.in_dimension(dim), ref[dim][0])
        assert_rows_equal(mb.index_triples_in_dimension(dim), ref[dim][1])
        assert_rows_equal(mb.essential_in_dimension(dim), ref[dim][2])


# ---------------------------------------------------------------------------
# Oracle 2: the authors' reference implementation (SoCG repo data)
# ---------------------------------------------------------------------------

def test_reference_implementation_cross_check():
    # Data provenance: analysis/data of github.com/hubwag/Mixup-SoCG26 -- a
    # 50-point run of the authors' hacked-Ripser pipeline (A = points 0..9,
    # B = points 10..49). The npz holds the true pairwise distance matrix
    # and their standard/image barcodes converted to oineus conventions:
    # dom0/im0 are (birth vertex, death diameter) rows of the finite degree-0
    # bars of VR(A) and of the image of VR(A) -> VR(A u B) (their per-vertex
    # negative birth tags decoded); dom1/im1 are the degree-1 bars in
    # diameters. Their pipeline includes the B vertices in the subfiltration
    # as isolated points, but since all A vertices precede all B vertices,
    # the elder rule gives the same image deaths as with L = VR(A); the
    # B-born image bars are simply absent here.
    data = np.load(Path(__file__).with_name("mixup_ref_socg.npz"))
    D, n_A, thr = data["dist"], int(data["n_A"]), float(data["threshold"])

    L = oin.vr_filtration(D[:n_A, :n_A], from_pwdists=True, max_dim=2,
                          max_diameter=thr, packed=False)
    K = oin.vr_filtration(D, from_pwdists=True, max_dim=2,
                          max_diameter=thr, packed=False)
    mb = oin.mixup_barcodes_of_filtrations(K, L, max_dim=1)

    # degree 0, matched per birth vertex: since all vertices of K are born at
    # value 0 in id order (A points first), the K sorted id of a vertex is its
    # point index, so column 0 of index_triples is the birth vertex
    idx0 = mb.index_triples_in_dimension(0)
    val0 = mb.in_dimension(0)
    assert len(val0) == n_A - 1
    mine_dom = {int(i): d for i, d in zip(idx0[:, 0], val0[:, 2])}
    mine_im = {int(i): d for i, d in zip(idx0[:, 0], val0[:, 1])}
    ref_dom = {int(v): d for v, d in data["dom0"]}
    ref_im = {int(v): d for v, d in data["im0"]}
    assert mine_dom.keys() == ref_dom.keys() == mine_im.keys() == ref_im.keys()
    for v in ref_dom:
        assert mine_dom[v] == pytest.approx(ref_dom[v], abs=1e-9)
        assert mine_im[v] == pytest.approx(ref_im[v], abs=1e-9)

    # degree 1: a single bar
    (b, d), (bi, di) = data["dom1"][0], data["im1"][0]
    assert b == pytest.approx(bi, abs=1e-9)
    assert_rows_equal(mb.in_dimension(1), [[b, di, d]], atol=1e-9)

    # statistics recomputed with the formulas of the authors' pipeline_stats
    # (get_persistence_stats): mixup = d - d', pers = d - b (dim-0 births 0)
    mix0 = np.array([ref_dom[v] - ref_im[v] for v in ref_dom])
    pers0 = np.array([ref_dom[v] for v in ref_dom])
    assert mb.total_mixup(0) == pytest.approx(mix0.sum(), abs=1e-9)
    assert mb.total_persistence(0) == pytest.approx(pers0.sum(), abs=1e-9)
    assert mb.total_mixup_percentage(0) == pytest.approx((mix0 / pers0).sum(), abs=1e-9)
    assert mb.mean_mixup_percentage(0) == pytest.approx((mix0 / pers0).mean(), abs=1e-9)
    assert mb.total_mixup(1) == pytest.approx(d - di, abs=1e-9)
    assert mb.mean_mixup_percentage(1) == pytest.approx((d - di) / (d - b), abs=1e-9)


# ---------------------------------------------------------------------------
# structural identities from the paper, on random inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_paper_identities(seed):
    rng = np.random.default_rng(seed)
    A = rng.random((11, 2))
    B = rng.random((6, 2))
    max_dim = 1
    K, L = make_K_L(A, B, max_dim)
    mb = oin.mixup_barcodes_of_filtrations(K, L, max_dim=max_dim)

    # independent computations of dgm(L) and of the image diagram
    dcmp = oin.Decomposition(L, dualize=False, n_threads=1)
    dcmp.reduce(oin.ReductionParams())
    dgms_L = dcmp.diagram(L)
    params = oin.KICRParams()
    params.kernel = params.cokernel = params.codomain = False
    kicr = oin.compute_kernel_image_cokernel_reduction(K, L, params)

    for dim in range(max_dim + 1):
        t = mb.in_dimension(dim)
        # b <= d' <= d
        assert (t[:, 0] <= t[:, 1]).all() and (t[:, 1] <= t[:, 2]).all()

        # the mixup barcode contains the (finite) persistence bars of L
        dgm = np.asarray(dgms_L.in_dimension(dim), dtype=float).reshape(-1, 2)
        finite = dgm[np.isfinite(dgm[:, 1])]
        assert_rows_equal(mb.persistence_barcode(dim), finite)

        # its image sub-bars correspond to the image persistence barcode
        # (positive-length sub-bars = the finite image diagram points)
        im = np.asarray(kicr.image_diagrams().in_dimension(dim), dtype=float).reshape(-1, 2)
        im_finite = im[np.isfinite(im[:, 1])]
        sub = mb.image_sub_barcode(dim)
        assert_rows_equal(sub[sub[:, 1] > sub[:, 0]], im_finite)

        # total mixup = total persistence of L - total image persistence
        assert mb.total_mixup(dim) == pytest.approx(
            (finite[:, 1] - finite[:, 0]).sum() - (im_finite[:, 1] - im_finite[:, 0]).sum())

        # totals decompose bar by bar
        assert mb.total_mixup(dim) == pytest.approx(
            mb.total_persistence(dim) - (sub[:, 1] - sub[:, 0]).sum())


def test_subsampling_property():
    # Lemma (Subsampling Property): total-mixup(A, B') <= total-mixup(A, B)
    # for B' a subset of B, in every degree; same for the total percentage
    rng = np.random.default_rng(5)
    A = rng.random((10, 2))
    B = rng.random((8, 2))
    full = oin.mixup_barcodes(A, B, max_dim=1)
    for k in range(len(B) + 1):
        sub = oin.mixup_barcodes(A, B[:k], max_dim=1)
        for dim in (0, 1):
            assert sub.total_mixup(dim) <= full.total_mixup(dim) + 1e-12
            assert sub.total_mixup_percentage(dim) <= full.total_mixup_percentage(dim) + 1e-12


def test_b_empty_means_zero_mixup():
    rng = np.random.default_rng(6)
    A = rng.random((9, 3))
    for B in (None, [], np.empty((0, 3)), np.empty((0, 0))):
        mb = oin.mixup_barcodes(A, B, max_dim=1)
        for dim in (0, 1):
            t = mb.in_dimension(dim)
            assert np.array_equal(t[:, 1], t[:, 2])  # d' == d: no premature death
            assert mb.total_mixup(dim) == 0.0
            assert mb.mean_mixup_percentage(dim) == 0.0
        assert len(mb.in_dimension(0)) == len(A) - 1


# ---------------------------------------------------------------------------
# edge cases, result-object behavior
# ---------------------------------------------------------------------------

def test_empty_A():
    mb = oin.mixup_barcodes(np.empty((0, 2)), np.random.default_rng(0).random((4, 2)), max_dim=1)
    for dim in (0, 1):
        assert mb.in_dimension(dim).shape == (0, 3)
        assert mb.essential_in_dimension(dim).shape == (0, 3)
        assert mb.total_mixup(dim) == 0.0
        assert mb.mean_mixup_percentage(dim) == 0.0


def test_single_point_A():
    mb = oin.mixup_barcodes(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), max_dim=1)
    assert mb.in_dimension(0).shape == (0, 3)
    assert_rows_equal(mb.essential_in_dimension(0), [[0.0, INF, INF]])
    assert mb.total_mixup(0) == 0.0


def test_empty_higher_dimension():
    # 4 generic points in the plane have no degree-2 homology; simplices up
    # to dimension 3 are enumerated but the degree-2 mixup barcode is empty
    rng = np.random.default_rng(8)
    mb = oin.mixup_barcodes(rng.random((4, 2)), rng.random((2, 2)), max_dim=2)
    assert mb.in_dimension(2).shape == (0, 3)
    assert mb.total_mixup(2) == 0.0
    assert mb.mean_mixup_percentage(2) == 0.0


def test_truncation_essential_bars():
    # A = {0, 10}, B = {5} on a line, truncated at diameter 6: the edge of L
    # never appears, so both classes of L are essential, but the younger one
    # still dies in the image at 5 through the bridge point of B: its
    # essential triple has a finite image death.
    A = np.array([[0.0], [10.0]])
    B = np.array([[5.0]])
    mb = oin.mixup_barcodes(A, B, max_dim=0, max_diameter=6.0)
    assert mb.in_dimension(0).shape == (0, 3)
    assert_rows_equal(mb.essential_in_dimension(0), [[0.0, 5.0, INF], [0.0, INF, INF]])


def test_truncated_essential_with_zero_image_bar():
    # A = unit square, B = center, truncated between 1 and sqrt(2): the
    # square cycle of L is born at 1 but its death at sqrt(2) is cut off,
    # so it is essential in L -- while in K the center triangles fill it at
    # exactly its birth value 1, a zero-persistence image pair. The
    # essential triple must report the true premature death (= the birth),
    # not +inf.
    A = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.5, 0.5]])
    mb = oin.mixup_barcodes(A, B, max_dim=1, max_diameter=1.2)
    assert mb.in_dimension(1).shape == (0, 3)
    assert_rows_equal(mb.essential_in_dimension(1), [[1.0, 1.0, INF]])


def test_negative_filtration_values():
    # mixup_barcodes_of_filtrations accepts general sublevel filtrations;
    # negative values must not trip the sign-sensitive tolerance in the
    # image-death sanity check. K == L, so every image death equals the
    # domain death exactly (zero mixup).
    cells = [[0, [0], -3.0], [1, [1], -3.0], [2, [0, 1], -2.0]]
    K = oin.list_to_filtration(cells)
    mb = oin.mixup_barcodes_of_filtrations(K, K, max_dim=0)
    assert_rows_equal(mb.in_dimension(0), [[-3.0, -2.0, -2.0]])
    assert mb.total_mixup(0) == 0.0


def test_input_validation():
    A = np.zeros((3, 2))
    with pytest.raises(ValueError):
        oin.mixup_barcodes(A, np.zeros((2, 3)))  # ambient dimension mismatch
    with pytest.raises(ValueError):
        oin.mixup_barcodes(np.zeros(3), None)  # not 2D
    with pytest.raises(ValueError):
        oin.mixup_barcodes(A, None, max_dim=-1)
    mb = oin.mixup_barcodes(A[:2] + np.arange(2), np.ones((1, 2)), max_dim=0)
    with pytest.raises(KeyError):
        mb.in_dimension(5)


def test_pickle_repr_and_dict_access():
    rng = np.random.default_rng(9)
    A, B = rng.random((7, 2)), rng.random((3, 2))
    mb = oin.mixup_barcodes(A, B, max_dim=1)
    r = repr(mb)
    assert "MixupBarcodes" in r and "total_mixup" in r
    mb2 = pickle.loads(pickle.dumps(mb))
    assert mb2.keys() == mb.keys() == [0, 1]
    assert 0 in mb2 and 5 not in mb2
    for dim in (0, 1):
        assert np.array_equal(mb2[dim], mb[dim])
        assert np.array_equal(mb2.index_triples_in_dimension(dim),
                              mb.index_triples_in_dimension(dim))
        assert mb2.total_mixup(dim) == mb.total_mixup(dim)
    # accessors return copies
    mb[0][:] = -1.0
    assert (mb[0] >= 0).all()


def test_sub_barcode_views_consistent():
    rng = np.random.default_rng(13)
    mb = oin.mixup_barcodes(rng.random((8, 2)), rng.random((4, 2)), max_dim=1)
    for dim in (0, 1):
        t = mb.in_dimension(dim)
        assert np.array_equal(mb.persistence_barcode(dim), t[:, [0, 2]])
        assert np.array_equal(mb.image_sub_barcode(dim), t[:, [0, 1]])
        assert np.array_equal(mb.mixup_sub_barcode(dim), t[:, [1, 2]])
        p = mb.mixup_percentages(dim)
        assert (p >= 0).all() and (p <= 1).all()


def test_n_threads_agree():
    rng = np.random.default_rng(14)
    A, B = rng.random((12, 2)), rng.random((6, 2))
    mb1 = oin.mixup_barcodes(A, B, max_dim=1, n_threads=1)
    mb4 = oin.mixup_barcodes(A, B, max_dim=1, n_threads=4)
    for dim in (0, 1):
        assert_rows_equal(mb1.in_dimension(dim), mb4.in_dimension(dim))
        assert_rows_equal(mb1.essential_in_dimension(dim), mb4.essential_in_dimension(dim))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
