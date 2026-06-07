"""Tests for decomposition-manipulation methods (vineyards, moves, move
schedules, Luo-Nelson warm-start updates) and the DecompositionManipStats
struct. Correctness is checked at the index-pairing level against a from-scratch
reduction of the (permuted/edited) boundary matrix."""
import pickle

import numpy as np
import pytest

import oineus as oin


def pairing(r_data):
    """Index persistence pairing { (low(R[c]), c) } of a reduced R."""
    return frozenset((max(col), c) for c, col in enumerate(r_data) if col)


def reduce_params():
    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = 1
    p.clearing_opt = False
    return p


def from_scratch_pairing(boundary):
    d = oin.Decomposition(boundary, len(boundary), False, True)
    d.reduce(reduce_params())
    return pairing(d.r_data)


def grid_decomposition(side=6, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((side, side))
    fil = oin.freudenthal_filtration(data, max_dim=2)
    dcmp = oin.Decomposition(fil, False)
    dcmp.reduce(reduce_params())
    return dcmp


def within_dim_permutation(dcmp, seed):
    n = len(dcmp.d_data)
    df, dl = list(dcmp.dim_first), list(dcmp.dim_last)
    new_to_old = list(range(n))
    rng = np.random.default_rng(seed)
    for d in range(len(df)):
        block = list(range(df[d], dl[d] + 1))
        perm = block.copy()
        rng.shuffle(perm)
        for i, pos in enumerate(block):
            new_to_old[pos] = perm[i]
    return new_to_old


@pytest.mark.parametrize("method", ["transpose_to", "apply_move_schedule", "update_with_permutation"])
def test_reorder_matches_full_recompute(method):
    base = grid_decomposition(seed=1)
    new_to_old = within_dim_permutation(base, seed=2)
    for _ in range(3):
        dcmp = grid_decomposition(seed=1)
        stats = oin.DecompositionManipStats()
        getattr(dcmp, method)(new_to_old, stats)
        assert pairing(dcmp.r_data) == from_scratch_pairing(dcmp.d_data)


def test_update_with_permutation_sanity():
    # Alg 2 re-triangularizes V, so the full sanity_check must pass.
    dcmp = grid_decomposition(seed=3)
    new_to_old = within_dim_permutation(dcmp, seed=4)
    dcmp.update_with_permutation(new_to_old)
    assert dcmp.is_reduced_consistent()
    assert dcmp.sanity_check()


def test_update_with_edits_delete_and_insert():
    base = grid_decomposition(seed=5)
    D0 = [list(col) for col in base.d_data]
    n_old = len(D0)
    df, dl = list(base.dim_first), list(base.dim_last)
    top_dim = len(df) - 1
    e0 = df[1]  # first edge index

    def cell_dim(o):
        return next(d for d in range(len(df)) if df[d] <= o <= dl[d])

    # delete the last (top-dimensional) cell, append a new top cell (3 edges):
    # keeps the order dimension-blocked
    new_to_old = list(range(n_old - 1)) + [-1]
    new_boundary = [list(col) for col in D0[:n_old - 1]]
    new_boundary.append([e0, e0 + 1, e0 + 2])

    dims = [cell_dim(o) for o in range(n_old - 1)] + [top_dim]
    ndf = [dims.index(d) for d in range(top_dim + 1)]
    ndl = [max(k for k, dd in enumerate(dims) if dd == d) for d in range(top_dim + 1)]

    dcmp = grid_decomposition(seed=5)
    stats = oin.DecompositionManipStats()
    dcmp.update_with_edits(new_to_old, new_boundary, ndf, ndl, stats)

    assert len(dcmp.r_data) == n_old
    assert dcmp.sanity_check()
    assert [list(c) for c in dcmp.d_data] == new_boundary
    assert list(dcmp.dim_first) == ndf and list(dcmp.dim_last) == ndl
    assert pairing(dcmp.r_data) == from_scratch_pairing(dcmp.d_data)


def test_transpose_single():
    dcmp = grid_decomposition(seed=6)
    df, dl = list(dcmp.dim_first), list(dcmp.dim_last)
    # a same-dimension adjacent pair inside the edge block
    i = df[1]
    dcmp.transpose(i)
    assert dcmp.sanity_check()
    assert pairing(dcmp.r_data) == from_scratch_pairing(dcmp.d_data)


def test_stats_pickle_and_counters():
    dcmp = grid_decomposition(seed=7)
    new_to_old = within_dim_permutation(dcmp, seed=8)
    stats = oin.DecompositionManipStats()
    dcmp.transpose_to(new_to_old, stats)
    assert stats.n_transpositions > 0
    assert stats.n_column_additions() == stats.n_column_additions_r + stats.n_column_additions_v
    assert stats.elapsed_transpose >= 0.0

    blob = pickle.dumps(stats)
    s2 = pickle.loads(blob)
    assert s2.n_transpositions == stats.n_transpositions
    assert s2.n_column_additions_r == stats.n_column_additions_r
    assert s2.n_column_additions_v == stats.n_column_additions_v

    stats.reset()
    assert stats.n_transpositions == 0
    assert stats.n_column_additions() == 0


def _coh_dgm_keys(dcmp, fil, max_dim=2):
    out = []
    for d in range(max_dim + 1):
        a = dcmp.diagram(fil, include_inf_points=True).in_dimension(d)
        a = a[np.isfinite(a).all(axis=1)] if len(a) else a
        out.append(np.round(np.sort(a, axis=0), 6) if len(a) else np.zeros((0, 2)))
    return out


def _keys_equal(a, b):
    return len(a) == len(b) and all(x.shape == y.shape and np.allclose(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("method", ["transpose_to", "apply_move_schedule", "update_with_permutation"])
def test_cohomology_reorder_matches_full_recompute(method):
    # Cohomology (dualize=True) is supported. The matrix is the antitransposed
    # boundary in reversed filtration order, so a filtration reorder is handed to
    # the matrix-space methods as the reversal-conjugate permutation. The warm
    # cohomology diagram must match the from-scratch dualize reduction (and the
    # homology diagram).
    rng = np.random.default_rng(11)
    d0 = np.ascontiguousarray(rng.random((8, 8)))
    d1 = d0.copy()
    for _ in range(5):
        r, c = rng.integers(0, 8, 2)
        d1[r, c] = rng.random()
    f0 = oin.freudenthal_filtration(d0, max_dim=2)
    f1 = oin.freudenthal_filtration(np.ascontiguousarray(d1), max_dim=2)
    n = f0.size()
    o0 = {c.uid: c.sorted_id for c in f0.cells()}
    nto_fil = [o0[c.uid] for c in f1.cells()]
    nto_mat = [(n - 1) - nto_fil[(n - 1) - k] for k in range(n)]   # reversal-conjugate

    scratch = oin.Decomposition(f1, True)
    scratch.reduce(reduce_params())
    key = _coh_dgm_keys(scratch, f1)

    dcmp = oin.Decomposition(f0, True)
    dcmp.reduce(reduce_params())
    getattr(dcmp, method)(nto_mat)
    assert _keys_equal(_coh_dgm_keys(dcmp, f1), key)


def test_update_with_edits_dualize_raises():
    # Edits (insert/delete) under the reversed cohomology layout are not handled.
    f = oin.freudenthal_filtration(np.random.default_rng(9).random((6, 6)), max_dim=2)
    dcmp_coh = oin.Decomposition(f, True)
    dcmp_coh.reduce(reduce_params())
    n = len(dcmp_coh.d_data)
    with pytest.raises(Exception):
        dcmp_coh.update_with_edits(list(range(n)), [list(c) for c in dcmp_coh.d_data],
                                   list(dcmp_coh.dim_first), list(dcmp_coh.dim_last))
