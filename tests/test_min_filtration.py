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
