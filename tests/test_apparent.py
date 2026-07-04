"""Oracle tests for the apparent-pairs (decorated-matrix) optimization.

params.use_apparent_pairs leaves the apparent (Bauer) columns out of the working
matrix and resolves them on demand. It must be a pure optimization: the diagram
(finite + essential + zero-persistence, every dimension, homology and
cohomology) must be identical to the unoptimized reduction, and a matrix access
must reconstruct a valid R = D V.

Activation requires the fused compute_v path (params.compute_v=True,
params.n_threads > 1) on a complete cubical or slim Freudenthal grid complex;
oin.reduce(...) is that entry point.
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
    p.use_apparent_pairs = apparent
    dcmp = oin.reduce(fil, p, dualize)
    return fil, dcmp


def _reduce_fr(a, dualize, apparent, n_threads=4, max_dim=None):
    # slim Freudenthal (the default builder for non-wrap grids up to 4D)
    if max_dim is None:
        max_dim = a.ndim
    fil = oin.freudenthal_filtration(a, max_dim=max_dim, n_threads=n_threads)
    p = oin.ReductionParams()
    p.n_threads = n_threads
    p.compute_v = True
    p.use_apparent_pairs = apparent
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
    p.use_apparent_pairs = True
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


# --- slim Freudenthal (kind == Freudenthal now takes the apparent path too) ---

FR_CASES = [
    # (shape, seed, max_dim); max_dim=None means full dimension
    ((8, 8), 101, None),
    ((13, 9), 102, None),
    ((5, 6, 4), 103, None),
    ((8, 7, 6), 104, None),
    ((5, 5, 5), 105, 2),   # truncated: 3D grid, cells only up to dim 2
]


@pytest.mark.parametrize("shape,seed,max_dim", FR_CASES)
@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("n_threads", [1, 4])
def test_apparent_fr_matches_plain(shape, seed, max_dim, dualize, n_threads):
    # n_threads=1 cannot fuse, so use_apparent_pairs must be a silent no-op there
    a = np.random.default_rng(seed).standard_normal(shape).astype(np.float64)
    _, ref = _reduce_fr(a, dualize, apparent=False, n_threads=n_threads, max_dim=max_dim)
    fil, test = _reduce_fr(a, dualize, apparent=True, n_threads=n_threads, max_dim=max_dim)
    ctx = f"fr shape={shape} max_dim={max_dim} dualize={dualize} n_threads={n_threads}"
    _assert_dgms_equal(_dgms(ref, fil, len(shape)), _dgms(test, fil, len(shape)), ctx)
    _assert_dgms_equal(_zero_pers(ref, fil, len(shape)), _zero_pers(test, fil, len(shape)), ctx + " [zero-pers]")


@pytest.mark.parametrize("dualize", [False, True])
def test_apparent_fr_ties_plateau(dualize):
    # constant field with one well: heavy value ties stress the youngest-facet /
    # oldest-cofacet sorted-order tie-breaking on the Kuhn triangulation
    for shape in [(10, 10), (6, 6, 4)]:
        a = np.zeros(shape, dtype=np.float64)
        a[tuple(s // 2 for s in shape)] = -1.0
        _, ref = _reduce_fr(a, dualize, apparent=False)
        fil, test = _reduce_fr(a, dualize, apparent=True)
        ctx = f"fr ties shape={shape} dualize={dualize}"
        _assert_dgms_equal(_dgms(ref, fil, len(shape)), _dgms(test, fil, len(shape)), ctx)
        _assert_dgms_equal(_zero_pers(ref, fil, len(shape)), _zero_pers(test, fil, len(shape)), ctx + " [zero-pers]")


@pytest.mark.parametrize("dualize", [False, True])
def test_apparent_fr_materialize_is_valid_decomposition(dualize):
    # the on-demand resolver must regenerate the Freudenthal apparent columns
    # into a valid at-rest R = D V, leaving the diagram unchanged
    a = np.random.default_rng(23).standard_normal((6, 5, 4)).astype(np.float64)
    fil, dcmp = _reduce_fr(a, dualize, apparent=True)
    before = _dgms(dcmp, fil, a.ndim)

    D = fil.coboundary_matrix(n_threads=1) if dualize else fil.boundary_matrix(n_threads=1)
    R = dcmp.r_as_csc()   # triggers materialize_from_working_
    V = dcmp.v_as_csc()
    assert dcmp.sanity_check(D), f"fr R = D V failed after materialize (dualize={dualize})"
    assert R.nnz > 0 and V.nnz > 0

    after = _dgms(dcmp, fil, a.ndim)
    _assert_dgms_equal(before, after, f"fr diagram stability across materialize (dualize={dualize})")


@pytest.mark.parametrize("dualize", [False, True])
def test_apparent_activation_telemetry(dualize):
    # guard against a silent gate regression: with the flag ON on the fused
    # multi-threaded path, both the cube and the slim Freudenthal grid
    # filtrations must actually take the apparent path
    a = np.random.default_rng(7).standard_normal((12, 11)).astype(np.float64)
    for reducer in (_reduce, _reduce_fr):
        name = reducer.__name__
        fil, on = reducer(a, dualize, apparent=True)
        assert on.n_apparent_pairs() > 0, f"{name}: apparent path fell back silently"
        assert on.n_apparent_pairs() < fil.size()
        _, off = reducer(a, dualize, apparent=False)
        assert off.n_apparent_pairs() == 0, f"{name}: OFF path reports apparent pairs"
        # serial reduction cannot fuse, so the flag is a no-op there
        _, serial = reducer(a, dualize, apparent=True, n_threads=1)
        assert serial.n_apparent_pairs() == 0, f"{name}: serial path took apparent"
        # a materializing access consumes the lean state and resets the counter
        assert on.r_as_csc().nnz > 0
        assert on.n_apparent_pairs() == 0, f"{name}: counter survived materialize"
