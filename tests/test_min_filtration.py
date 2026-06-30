"""Coverage for the non-differentiable oin.min_filtration over every cell encoding.

min_filtration / min_filtration(..., with_indices=True) were folded over all slim/packed
cell types (cube / Freudenthal / bit-packed) but had no committed coverage at the bare
(non-diff) API level -- only the diff wrapper and a fat size-mismatch test existed. Here
we check, for fat / slim-Freudenthal / packed-VR / cube filtrations of the same complex:
  - the result has the same encoding and the filtration-min value per cell (by uid);
  - negate is respected (filtration-min is numeric max when negate=True);
  - mismatched size and mismatched negate are rejected with a RuntimeError, for both the
    plain and the with_indices entry points (the assert->throw lives in the shared C++
    template, so exercising each encoding guards against a per-type binding regression).
"""
import math

import numpy as np
import pytest

import oineus as oin

dion = None  # not needed


def _value_by_uid(fil):
    return {fil.cell(i).uid: fil.cell_value_by_sorted_id(i) for i in range(fil.size())}


def _check_min(A, B, negate):
    mn = oin.min_filtration(A, B)
    assert type(mn).__name__ == type(A).__name__
    assert mn.size() == A.size()
    va, vb = _value_by_uid(A), _value_by_uid(B)
    combine = max if negate else min
    for i in range(mn.size()):
        uid = mn.cell(i).uid
        exp = combine(va[uid], vb[uid])
        assert mn.cell_value_by_sorted_id(i) == pytest.approx(exp), f"cell {i}"
    # with_indices must return a filtration equal in cell values to the plain path
    mn2, _, _ = oin.min_filtration(A, B, with_indices=True)
    assert type(mn2).__name__ == type(A).__name__
    v1, v2 = _value_by_uid(mn), _value_by_uid(mn2)
    assert v1.keys() == v2.keys()
    for uid in v1:
        assert v1[uid] == pytest.approx(v2[uid])


@pytest.mark.parametrize("negate", [False, True])
def test_min_filtration_freudenthal_slim(negate):
    shape = (6, 7)
    a = np.random.default_rng(0).random(shape)
    b = np.random.default_rng(1).random(shape)
    A = oin.freudenthal_filtration(a, max_dim=2, negate=negate)   # default slim
    B = oin.freudenthal_filtration(b, max_dim=2, negate=negate)
    assert type(A).__name__ == "_FreudenthalFiltration_2D"
    _check_min(A, B, negate)


@pytest.mark.parametrize("negate", [False, True])
def test_min_filtration_cube(negate):
    shape = (5, 6)
    a = np.random.default_rng(2).random(shape)
    b = np.random.default_rng(3).random(shape)
    A = oin.cube_filtration(a, max_dim=2, negate=negate)
    B = oin.cube_filtration(b, max_dim=2, negate=negate)
    assert type(A).__name__.startswith("_CubeFiltration")
    _check_min(A, B, negate)


def test_min_filtration_packed_vr():
    # max_diameter huge -> complete 2-skeleton on both clouds, so identical combinatorics
    pts1 = np.ascontiguousarray(np.random.default_rng(4).random((6, 3)))
    pts2 = np.ascontiguousarray(np.random.default_rng(5).random((6, 3)))
    A = oin.vr_filtration(pts1, max_dim=2, max_diameter=1e9)       # default packed
    B = oin.vr_filtration(pts2, max_dim=2, max_diameter=1e9)
    assert type(A).__name__ == "_PackedSimplexFiltration_64"
    _check_min(A, B, negate=False)


def test_min_filtration_fat():
    pts1 = np.ascontiguousarray(np.random.default_rng(6).random((6, 3)))
    pts2 = np.ascontiguousarray(np.random.default_rng(7).random((6, 3)))
    A = oin.vr_filtration(pts1, max_dim=2, max_diameter=1e9, packed=False)
    B = oin.vr_filtration(pts2, max_dim=2, max_diameter=1e9, packed=False)
    assert type(A).__name__ == "_Filtration"
    _check_min(A, B, negate=False)


@pytest.mark.parametrize("with_indices", [False, True])
def test_min_filtration_rejects_mismatched_negate_slim(with_indices):
    a = np.random.default_rng(8).random((5, 5))
    A = oin.freudenthal_filtration(a, max_dim=2, negate=False)
    B = oin.freudenthal_filtration(a, max_dim=2, negate=True)
    with pytest.raises(RuntimeError):
        oin.min_filtration(A, B, with_indices=with_indices)


@pytest.mark.parametrize("with_indices", [False, True])
def test_min_filtration_rejects_mismatched_sizes_packed(with_indices):
    pa = np.ascontiguousarray(np.random.default_rng(9).random((7, 3)))
    A = oin.vr_filtration(pa, max_dim=2, max_diameter=1.0)
    B = A.without_cells([A.size() - 1])     # one fewer cell, same encoding
    assert A.size() != B.size()
    with pytest.raises(RuntimeError):
        oin.min_filtration(A, B, with_indices=with_indices)


def test_min_filtration_with_indices_aligned_to_result_order():
    # Regression: min_filtration_with_indices built the source-index permutations in the
    # (value, dim) presort order used internally, but the Filtration constructor re-sorts the
    # cells into (dim, value, id) order. The returned indices must index the SAME cell as the
    # returned filtration at each sorted position. The two orders diverge only when, IN THE
    # RESULT (min) filtration, a lower-dim cell out-values a higher-dim cell. That is the
    # configuration this input is built for: after min(), vertex [2] = min(10, 12) = 10 still
    # exceeds edge [0,1] = min(2, 5) = 2, so the presort order (value-first) puts the edge
    # before the vertex while the final order (dim-first) puts the vertex first -- the buggy
    # permutation misaligns at exactly those positions. (An earlier version of this test used an
    # input whose result had NO such inversion, so the presort and final orders coincided and
    # the buggy code passed it; the values below restore a genuine guard.)
    cells = [oin.Simplex([0], 0.0), oin.Simplex([1], 0.0), oin.Simplex([2], 10.0),
             oin.Simplex([0, 1], 2.0), oin.Simplex([0, 2], 11.0), oin.Simplex([1, 2], 11.0)]
    A = oin.Filtration(cells, False, 1)
    cells2 = [oin.Simplex([0], 1.0), oin.Simplex([1], 0.0), oin.Simplex([2], 12.0),
              oin.Simplex([0, 1], 5.0), oin.Simplex([0, 2], 11.0), oin.Simplex([1, 2], 13.0)]
    B = oin.Filtration(cells2, False, 1)

    # guard the guard: the result must actually contain the dim/value inversion this test
    # relies on, else it would silently degrade into a non-guard again (as the earlier input
    # did). The buggy permutation only diverges from the correct one when some dim-0 cell
    # out-values some dim-1 cell in the result.
    res = oin.min_filtration(A, B)
    dim0_max = max(res.cell_value_by_sorted_id(k) for k in range(res.size()) if res.cell(k).dim == 0)
    dim1_min = min(res.cell_value_by_sorted_id(k) for k in range(res.size()) if res.cell(k).dim == 1)
    assert dim0_max > dim1_min, "input no longer triggers the dim/value inversion it guards"

    fil, i1, i2 = oin.min_filtration(A, B, with_indices=True)
    assert fil.size() == A.size() == len(i1) == len(i2)
    va, vb = _value_by_uid(A), _value_by_uid(B)
    for k in range(fil.size()):
        uid = fil.cell(k).uid
        # index k must point to the SAME cell (by uid) in both source filtrations
        assert A.cell(i1[k]).uid == uid
        assert B.cell(i2[k]).uid == uid
        # and the result value is the per-cell filtration-min of the two sources
        assert fil.cell_value_by_sorted_id(k) == pytest.approx(min(va[uid], vb[uid]))
