"""Oracle tests for the apparent-pairs (decorated-matrix) optimization.

params.apparent_opt leaves the apparent (Bauer) columns out of the working
matrix and resolves them on demand. It must be a pure optimization: the diagram
(finite + essential + zero-persistence, every dimension, homology and
cohomology) must be identical to the unoptimized reduction, and a matrix access
must reconstruct a valid R = D V.

Activation requires the fused compute_v path (params.compute_v=True,
params.n_threads > 1) on a complete cubical complex; oin.reduce(...) is that
entry point.
"""
import gc

import numpy as np
import pytest
import oineus as oin


def _reduce(a, dualize, apparent, values_on="vertices", n_threads=4):
    fil = oin.cube_filtration(a, n_threads=n_threads, values_on=values_on)
    p = oin.ReductionParams()
    p.n_threads = n_threads
    p.compute_v = True
    p.apparent_opt = apparent
    dcmp = oin.reduce(fil, p, dualize)
    return fil, dcmp


def _dgms(dcmp, fil, ndim):
    return [np.asarray(dcmp.diagram(fil).in_dimension(d)).reshape(-1, 2)
            for d in range(ndim)]


def _zero_pers(dcmp, fil, ndim):
    return [np.asarray(dcmp.zero_pers_diagram(fil).in_dimension(d)).reshape(-1, 2)
            for d in range(ndim)]


def _canon(pts):
    # sort rows so the comparison is order-independent; inf survives the sort
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return pts
    return pts[np.lexsort((pts[:, 1], pts[:, 0]))]


def _assert_dgms_equal(d_ref, d_test, ctx):
    assert len(d_ref) == len(d_test), ctx
    for dim, (r, t) in enumerate(zip(d_ref, d_test)):
        r, t = _canon(r), _canon(t)
        assert r.shape == t.shape, f"{ctx} dim {dim}: shape {r.shape} vs {t.shape}"
        if r.size:
            assert np.allclose(r, t, atol=1e-9, equal_nan=True), f"{ctx} dim {dim} values differ"


SHAPES_2D = [(8, 8), (13, 9), (32, 32)]
SHAPES_3D = [(5, 6, 4), (8, 7, 6)]


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("values_on", ["vertices", "cells"])
def test_apparent_matches_plain_2d(shape, dualize, values_on):
    a = np.random.default_rng(abs(hash((shape, dualize, values_on))) % 2**31)
    a = a.standard_normal(shape).astype(np.float64)
    _, ref = _reduce(a, dualize, apparent=False, values_on=values_on)
    fil, test = _reduce(a, dualize, apparent=True, values_on=values_on)
    ctx = f"2d shape={shape} dualize={dualize} values_on={values_on}"
    _assert_dgms_equal(_dgms(ref, fil, len(shape)), _dgms(test, fil, len(shape)), ctx)
    _assert_dgms_equal(_zero_pers(ref, fil, len(shape)), _zero_pers(test, fil, len(shape)), ctx + " [zero-pers]")


@pytest.mark.parametrize("shape", SHAPES_3D)
@pytest.mark.parametrize("dualize", [False, True])
def test_apparent_matches_plain_3d(shape, dualize):
    a = np.random.default_rng(abs(hash((shape, dualize))) % 2**31)
    a = a.standard_normal(shape).astype(np.float64)
    _, ref = _reduce(a, dualize, apparent=False)
    fil, test = _reduce(a, dualize, apparent=True)
    ctx = f"3d shape={shape} dualize={dualize}"
    _assert_dgms_equal(_dgms(ref, fil, len(shape)), _dgms(test, fil, len(shape)), ctx)
    _assert_dgms_equal(_zero_pers(ref, fil, len(shape)), _zero_pers(test, fil, len(shape)), ctx + " [zero-pers]")


def test_apparent_ties_constant_and_plateau():
    # heavy ties (constant / plateau regions) stress the youngest-facet /
    # oldest-cofacet tie-breaking that apparent detection relies on
    for shape, builder in [((10, 10), np.zeros), ((8, 8, 4), np.ones)]:
        a = builder(shape, dtype=np.float64)
        a[tuple(s // 2 for s in shape)] = -1.0  # one well, rest constant
        for dualize in [False, True]:
            _, ref = _reduce(a, dualize, apparent=False)
            fil, test = _reduce(a, dualize, apparent=True)
            ctx = f"ties shape={shape} dualize={dualize}"
            _assert_dgms_equal(_dgms(ref, fil, len(shape)), _dgms(test, fil, len(shape)), ctx)
            _assert_dgms_equal(_zero_pers(ref, fil, len(shape)), _zero_pers(test, fil, len(shape)), ctx + " [zero-pers]")


@pytest.mark.parametrize("dualize", [False, True])
def test_apparent_materialize_is_valid_decomposition(dualize):
    # a matrix access must reconstruct a valid R = D V from the lean form, and
    # leave the diagram unchanged
    a = np.random.default_rng(11).standard_normal((9, 7, 5)).astype(np.float64)
    fil, dcmp = _reduce(a, dualize, apparent=True)
    before = _dgms(dcmp, fil, a.ndim)

    D = fil.coboundary_matrix(n_threads=1) if dualize else fil.boundary_matrix(n_threads=1)
    R = dcmp.r_as_csc()   # triggers materialize_from_working_
    V = dcmp.v_as_csc()
    assert dcmp.sanity_check(D), f"R = D V failed after materialize (dualize={dualize})"
    assert R.nnz > 0 and V.nnz > 0

    after = _dgms(dcmp, fil, a.ndim)
    _assert_dgms_equal(before, after, f"diagram stability across materialize (dualize={dualize})")


@pytest.mark.parametrize("dualize", [False, True])
def test_apparent_resolver_survives_filtration_gc(dualize):
    # the lean form's resolver closes over the filtration; oin.reduce keeps it
    # alive (keep_alive<0,1>), so a deferred matrix access after the caller drops
    # its own reference must NOT be a use-after-free
    a = np.random.default_rng(17).standard_normal((9, 8)).astype(np.float64)
    fil = oin.cube_filtration(a, n_threads=4, values_on="vertices")
    D = fil.coboundary_matrix(n_threads=1) if dualize else fil.boundary_matrix(n_threads=1)
    p = oin.ReductionParams()
    p.n_threads = 4
    p.compute_v = True
    p.apparent_opt = True
    dcmp = oin.reduce(fil, p, dualize)
    del fil
    gc.collect()
    # triggers materialize_from_working_ -> resolver -> filtration access
    assert dcmp.sanity_check(D)
    assert dcmp.r_as_csc().nnz > 0


@pytest.mark.parametrize("dualize", [False, True])
def test_apparent_clone_is_self_contained(dualize):
    # clone() materializes the source; the copy must be a valid decomposition
    a = np.random.default_rng(13).standard_normal((10, 10)).astype(np.float64)
    fil, dcmp = _reduce(a, dualize, apparent=True)
    D = fil.coboundary_matrix(n_threads=1) if dualize else fil.boundary_matrix(n_threads=1)
    clone = dcmp.clone()
    assert clone.sanity_check(D)
    _assert_dgms_equal(_dgms(dcmp, fil, a.ndim), _dgms(clone, fil, a.ndim), f"clone diagram dualize={dualize}")
