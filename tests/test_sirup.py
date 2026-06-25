"""Tests for SiRUP simplex removal (Decomposition.remove_simplices,
Filtration.star_closure / is_up_closed / without_cells, and the high-level
oin.remove_simplices convenience).

Correctness is checked against a from-scratch reduction of the survivor
filtration: the updated diagram must match in every dimension, and for the
clearing-off case the updated R = D V must hold exactly (is_reduced_consistent).
"""
import numpy as np
import pytest

import oineus as oin


def reduce_params(clearing):
    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = 1
    p.clearing_opt = clearing
    return p


def diagram_dict(dgm, max_dim=4):
    """Per-dimension multiset of (birth, death) points, as sorted rounded tuples."""
    out = {}
    for d in range(max_dim):
        try:
            arr = dgm.in_dimension(d)
        except IndexError:
            arr = np.empty((0, 2))
        pts = sorted((round(float(b), 7), round(float(de), 7)) for b, de in arr)
        out[d] = pts
    return out


def reduced(fil, clearing):
    dcmp = oin.Decomposition(fil, False)
    dcmp.reduce(reduce_params(clearing))
    return dcmp


def freudenthal_fil(side=6, seed=0):
    rng = np.random.default_rng(seed)
    return oin.freudenthal_filtration(rng.random((side, side)), max_dim=2)


def vr_fil(n_pts=15, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.random((n_pts, 2))
    return oin.vr_filtration(pts, max_dim=2, max_diameter=2.0)


def cube_fil(side=6, seed=0):
    rng = np.random.default_rng(seed)
    return oin.cube_filtration(rng.random((side, side)), max_dim=2)


def make_fil(kind, seed):
    if kind == "freudenthal":
        return freudenthal_fil(seed=seed)
    if kind == "vr":
        return vr_fil(seed=seed)
    if kind == "cube":
        return cube_fil(seed=seed)
    raise ValueError(kind)


def check_removal(fil, seeds, clearing):
    """SiRUP via oin.remove_simplices vs a from-scratch reduction of survivors."""
    L = list(fil.star_closure(seeds))
    assert fil.is_up_closed(L)

    dcmp = reduced(fil, clearing)
    stats = oin.DecompositionManipStats()
    new_fil = oin.remove_simplices(fil, dcmp, seeds, stats=stats)

    assert len(new_fil) == len(fil) - len(L)
    got = diagram_dict(dcmp.diagram(new_fil, include_inf_points=True))

    ref_dcmp = reduced(new_fil, clearing)
    want = diagram_dict(ref_dcmp.diagram(new_fil, include_inf_points=True))
    assert got == want

    assert stats.n_column_additions_r >= 0 and stats.n_column_additions_v >= 0
    if not clearing:
        assert dcmp.is_reduced_consistent()
    return L


@pytest.mark.parametrize("kind", ["freudenthal", "vr", "cube"])
@pytest.mark.parametrize("clearing", [False, True])
def test_random_star_removals(kind, clearing):
    for seed in range(6):
        fil = make_fil(kind, seed)
        n = len(fil)
        rng = np.random.default_rng(1000 + seed)
        seeds = sorted(set(int(x) for x in rng.integers(0, n, size=3)))
        check_removal(fil, seeds, clearing)


@pytest.mark.parametrize("clearing", [False, True])
def test_remove_single_cells(clearing):
    fil = freudenthal_fil(side=5, seed=3)
    n = len(fil)
    for c in range(0, n, max(1, n // 12)):
        check_removal(fil, [c], clearing)


def test_explicit_L_and_close_star_false():
    """The explicit-set path (close_star=False) on an already-closed L matches."""
    fil = freudenthal_fil(seed=2)
    L = list(fil.star_closure([len(fil) // 2]))

    dcmp = reduced(fil, False)
    new_fil = oin.remove_simplices(fil, dcmp, L, close_star=False)
    got = diagram_dict(dcmp.diagram(new_fil, include_inf_points=True))

    ref = reduced(new_fil, False)
    want = diagram_dict(ref.diagram(new_fil, include_inf_points=True))
    assert got == want
    assert dcmp.is_reduced_consistent()


def test_low_level_remove_simplices_matches_convenience():
    fil = freudenthal_fil(seed=4)
    L = list(fil.star_closure([7, 30]))

    dcmp = reduced(fil, False)
    dcmp.remove_simplices(L)            # low-level, mutates dcmp in place
    new_fil = fil.without_cells(L)

    ref = reduced(new_fil, False)
    assert diagram_dict(dcmp.diagram(new_fil)) == diagram_dict(ref.diagram(new_fil))


def test_repeated_successive_removals():
    fil = freudenthal_fil(seed=5)
    dcmp = reduced(fil, False)
    rng = np.random.default_rng(11)
    for _ in range(4):
        if len(fil) <= 3:
            break
        seed_cell = int(rng.integers(0, len(fil)))
        fil = oin.remove_simplices(fil, dcmp, [seed_cell])
        ref = reduced(fil, False)
        assert diagram_dict(dcmp.diagram(fil)) == diagram_dict(ref.diagram(fil))
        assert dcmp.is_reduced_consistent()


def test_star_closure_of_vertex_is_all_cofaces():
    # tetra boundary: star of vertex 0 = every simplex containing vertex 0
    simplices = [
        oin.Simplex([0], 0.0), oin.Simplex([1], 0.1), oin.Simplex([2], 0.2), oin.Simplex([3], 0.3),
        oin.Simplex([0, 1], 0.4), oin.Simplex([0, 2], 0.5), oin.Simplex([0, 3], 0.6),
        oin.Simplex([1, 2], 0.7), oin.Simplex([1, 3], 0.8), oin.Simplex([2, 3], 0.9),
        oin.Simplex([0, 1, 2], 1.0), oin.Simplex([0, 1, 3], 1.1),
        oin.Simplex([0, 2, 3], 1.2), oin.Simplex([1, 2, 3], 1.3),
        oin.Simplex([0, 1, 2, 3], 1.4),
    ]
    fil = oin.Filtration(simplices, negate=False, n_threads=1)
    v0_sid = fil.sorted_id_by_id(0)
    star = set(fil.star_closure([v0_sid]))
    expected = {fil.sorted_id_by_id(i) for i, s in enumerate(simplices)
                if 0 in list(s.vertices)}
    assert star == expected
    assert fil.is_up_closed(list(star))


def test_is_up_closed_rejects_open_set():
    fil = freudenthal_fil(seed=1)
    # an edge that has at least one triangle coface is not up-closed by itself
    edge = next(c for c in range(len(fil)) if fil[c].dim == 1
                and len(fil.star_closure([c])) > 1)
    assert not fil.is_up_closed([edge])


def test_remove_simplices_rejects_non_closed_set():
    fil = freudenthal_fil(seed=1)
    edge = next(c for c in range(len(fil)) if fil[c].dim == 1
                and len(fil.star_closure([c])) > 1)
    dcmp = reduced(fil, False)
    with pytest.raises(Exception):
        dcmp.remove_simplices([edge])


def test_negate_filtration_essential_points():
    # negate=True exercises infinity()/neg_infinity() signs in the survivor
    # filtration; without_cells must inherit negate_ so essential points match.
    rng = np.random.default_rng(2)
    fil = oin.freudenthal_filtration(rng.random((6, 6)), negate=True, max_dim=2)
    n = len(fil)
    seeds = sorted(set(int(x) for x in np.random.default_rng(9).integers(0, n, size=3)))
    dcmp = reduced(fil, False)
    new_fil = oin.remove_simplices(fil, dcmp, seeds)
    got = diagram_dict(dcmp.diagram(new_fil, include_inf_points=True))
    ref = reduced(new_fil, False)
    want = diagram_dict(ref.diagram(new_fil, include_inf_points=True))
    assert got == want


def test_remove_simplices_rejects_cohomology():
    fil = freudenthal_fil(seed=1)
    dcmp = oin.Decomposition(fil, True)        # dualize = cohomology
    dcmp.reduce(reduce_params(False))
    with pytest.raises(Exception):
        dcmp.remove_simplices([0])
