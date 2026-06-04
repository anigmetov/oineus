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

    # delete the last (top-dimensional, coface-free) cell, append a new edge
    new_to_old = list(range(n_old - 1)) + [-1]
    new_boundary = [list(col) for col in D0[:n_old - 1]]
    new_boundary.append([0, 1])  # new edge on existing vertices 0,1

    dcmp = grid_decomposition(seed=5)
    stats = oin.DecompositionManipStats()
    dcmp.update_with_edits(new_to_old, new_boundary, stats)

    assert len(dcmp.r_data) == n_old
    assert dcmp.sanity_check()
    assert [list(c) for c in dcmp.d_data] == new_boundary
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


def test_dualize_raises():
    # manipulation methods are homology-only
    dcmp = grid_decomposition(seed=9)
    dcmp_coh = oin.Decomposition(oin.freudenthal_filtration(
        np.random.default_rng(9).random((6, 6)), max_dim=2), True)
    dcmp_coh.reduce(reduce_params())
    with pytest.raises(Exception):
        dcmp_coh.transpose(0)
