"""Oracle tests for the apparent-pairs (decorated-matrix) optimization.

params.use_apparent_pairs leaves the apparent (Bauer) columns out of the working
matrix and resolves them on demand. It must be a pure optimization: the diagram
(finite + essential + zero-persistence, every dimension, homology and
cohomology) must be identical to the unoptimized reduction, and a matrix access
must reconstruct a valid R = D V.

Activation requires the fused parallel path (params.n_threads > 1, both
compute_v=True and the R-only compute_v=False variant) on a complete cubical or
slim Freudenthal grid complex; oin.reduce(...) is that entry point.

The flag is tri-state: True/False force it on/off, None (the default) is auto
and resolves to ON only on the fused R-only cubical homology path.
"""
import gc
import pickle
import zlib

import numpy as np
import pytest
import oineus as oin


def _reduce(a, dualize, apparent, values_on="vertices", n_threads=4, negate=False, compute_v=True):
    fil = oin.cube_filtration(a, n_threads=n_threads, values_on=values_on, negate=negate)
    p = oin.ReductionParams()
    p.n_threads = n_threads
    p.compute_v = compute_v
    p.use_apparent_pairs = apparent
    dcmp = oin.reduce(fil, p, dualize)
    return fil, dcmp


def _reduce_fr(a, dualize, apparent, n_threads=4, max_dim=None, negate=False, compute_v=True):
    # slim Freudenthal (the default builder for non-wrap grids up to 4D)
    if max_dim is None:
        max_dim = a.ndim
    fil = oin.freudenthal_filtration(a, max_dim=max_dim, n_threads=n_threads, negate=negate)
    p = oin.ReductionParams()
    p.n_threads = n_threads
    p.compute_v = compute_v
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
@pytest.mark.parametrize("negate", [False, True])
def test_apparent_matches_plain_3d(shape, dualize, negate):
    # negate=True is the upper-star filtration: sorted order (and hence the
    # apparent pairing) is built on descending values
    a = np.random.default_rng(abs(hash((shape, dualize))) % 2**31)
    a = a.standard_normal(shape).astype(np.float64)
    _, ref = _reduce(a, dualize, apparent=False, negate=negate)
    fil, test = _reduce(a, dualize, apparent=True, negate=negate)
    ctx = f"3d shape={shape} dualize={dualize} negate={negate}"
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
    # (shape, seed, max_dim, negate); max_dim=None means full dimension,
    # negate=True is the upper-star filtration
    ((8, 8), 101, None, False),
    ((13, 9), 102, None, False),
    ((5, 6, 4), 103, None, False),
    ((8, 7, 6), 104, None, False),
    ((5, 5, 5), 105, 2, False),   # truncated: 3D grid, cells only up to dim 2
    ((9, 8), 107, None, True),
    ((6, 5, 4), 108, None, True),
]


@pytest.mark.parametrize("shape,seed,max_dim,negate", FR_CASES)
@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("n_threads", [1, 4])
def test_apparent_fr_matches_plain(shape, seed, max_dim, negate, dualize, n_threads):
    # n_threads=1 cannot fuse, so use_apparent_pairs must be a silent no-op there
    a = np.random.default_rng(seed).standard_normal(shape).astype(np.float64)
    _, ref = _reduce_fr(a, dualize, apparent=False, n_threads=n_threads, max_dim=max_dim, negate=negate)
    fil, test = _reduce_fr(a, dualize, apparent=True, n_threads=n_threads, max_dim=max_dim, negate=negate)
    ctx = f"fr shape={shape} max_dim={max_dim} negate={negate} dualize={dualize} n_threads={n_threads}"
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


# --- R-only (compute_v=False) fused path: the pivots-only post-state must
# --- carry the same diagram, with the apparent columns never materialized ---

@pytest.mark.parametrize("kind", ["cube", "fr"])
@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("n_threads", [2, 8])
@pytest.mark.parametrize("negate", [False, True])
def test_apparent_r_only_matches_plain(kind, dualize, n_threads, negate):
    reducer = _reduce if kind == "cube" else _reduce_fr
    # crc32, not hash(): string hashing is salted per process, and a failure
    # must be replayable with the same grid
    seed = zlib.crc32(repr((kind, dualize, n_threads, negate)).encode())
    a = np.random.default_rng(seed).standard_normal((8, 7, 6)).astype(np.float64)
    _, ref = reducer(a, dualize, apparent=False, n_threads=n_threads, negate=negate, compute_v=False)
    fil, test = reducer(a, dualize, apparent=True, n_threads=n_threads, negate=negate, compute_v=False)
    assert test.n_apparent_pairs() > 0, f"{kind}: R-only apparent path fell back silently"
    assert ref.n_apparent_pairs() == 0
    ctx = f"r-only kind={kind} dualize={dualize} n_threads={n_threads} negate={negate}"
    _assert_dgms_equal(_dgms(ref, fil, a.ndim), _dgms(test, fil, a.ndim), ctx)
    _assert_dgms_equal(_zero_pers(ref, fil, a.ndim), _zero_pers(test, fil, a.ndim), ctx + " [zero-pers]")


@pytest.mark.parametrize("kind", ["cube", "fr"])
@pytest.mark.parametrize("dualize", [False, True])
def test_apparent_r_only_serial_inert(kind, dualize):
    # serial reduction cannot fuse, so the flag stays a silent no-op at 1 thread
    reducer = _reduce if kind == "cube" else _reduce_fr
    a = np.random.default_rng(3).standard_normal((8, 7, 6)).astype(np.float64)
    fil, serial = reducer(a, dualize, apparent=True, n_threads=1, compute_v=False)
    assert serial.n_apparent_pairs() == 0, f"{kind}: serial R-only path took apparent"
    _, ref = reducer(a, dualize, apparent=False, n_threads=1, compute_v=False)
    ctx = f"r-only serial kind={kind} dualize={dualize}"
    _assert_dgms_equal(_dgms(ref, fil, a.ndim), _dgms(serial, fil, a.ndim), ctx)
    _assert_dgms_equal(_zero_pers(ref, fil, a.ndim), _zero_pers(serial, fil, a.ndim), ctx + " [zero-pers]")


# --- tri-state flag: None (the default) is Auto and resolves to ON only in
# --- the measured pure-win corner, the fused R-only cubical homology path ---

def _reduce_auto(kind, a, dualize=False, n_threads=8, compute_v=False):
    build = oin.cube_filtration if kind == "cube" else oin.freudenthal_filtration
    fil = build(a, n_threads=n_threads)
    # nothing set on the flag: default params, i.e. Auto
    dcmp = oin.reduce(fil, None, dualize, n_threads=n_threads, compute_v=compute_v)
    return fil, dcmp


def test_apparent_auto_activates_cube_hom_r_only():
    # default params must take the apparent path on the fused R-only cubical
    # homology reduction, and the diagram must match an explicit-Off run
    a = np.random.default_rng(2027).standard_normal((8, 7, 6)).astype(np.float64)
    fil, auto_d = _reduce_auto("cube", a)
    assert auto_d.n_apparent_pairs() > 0, "Auto did not activate in the pure-win corner"
    _, off = _reduce(a, dualize=False, apparent=False, n_threads=8, compute_v=False)
    assert off.n_apparent_pairs() == 0
    ctx = "auto cube hom r-only"
    _assert_dgms_equal(_dgms(off, fil, a.ndim), _dgms(auto_d, fil, a.ndim), ctx)
    _assert_dgms_equal(_zero_pers(off, fil, a.ndim), _zero_pers(auto_d, fil, a.ndim), ctx + " [zero-pers]")


@pytest.mark.parametrize("kind,dualize,compute_v", [
    ("cube", True, False),   # cohomology
    ("cube", False, True),   # RV path
    ("fr", False, False),    # Freudenthal homology R-only
])
def test_apparent_auto_stays_off_elsewhere(kind, dualize, compute_v):
    a = np.random.default_rng(2028).standard_normal((8, 7, 6)).astype(np.float64)
    _, dcmp = _reduce_auto(kind, a, dualize=dualize, compute_v=compute_v)
    assert dcmp.n_apparent_pairs() == 0, f"Auto activated outside its corner: {kind} dualize={dualize} compute_v={compute_v}"


def test_apparent_explicit_overrides_auto():
    a = np.random.default_rng(2029).standard_normal((8, 7, 6)).astype(np.float64)
    # explicit True on Freudenthal hom R-only: Auto would stay off, On activates
    _, on = _reduce_fr(a, dualize=False, apparent=True, n_threads=8, compute_v=False)
    assert on.n_apparent_pairs() > 0
    # explicit False on cube hom R-only: Auto would turn on, Off wins
    _, off = _reduce(a, dualize=False, apparent=False, n_threads=8, compute_v=False)
    assert off.n_apparent_pairs() == 0


def test_apparent_tristate_property_pickle_repr():
    p = oin.ReductionParams()
    assert p.use_apparent_pairs is None
    assert "use_apparent_pairs = auto" in repr(p)
    p.use_apparent_pairs = True
    assert p.use_apparent_pairs is True
    assert "use_apparent_pairs = on" in repr(p)
    p.use_apparent_pairs = False
    assert p.use_apparent_pairs is False
    assert "use_apparent_pairs = off" in repr(p)
    p.use_apparent_pairs = None
    assert p.use_apparent_pairs is None

    for val in (None, True, False):
        p.use_apparent_pairs = val
        back = pickle.loads(pickle.dumps(p))
        assert back == p
        assert back.use_apparent_pairs is val

    # the kwargs ctor accepts all three states and defaults to Auto
    assert oin.ReductionParams().use_apparent_pairs is None
    assert oin.ReductionParams(use_apparent_pairs=None).use_apparent_pairs is None
    assert oin.ReductionParams(use_apparent_pairs=True).use_apparent_pairs is True
    assert oin.ReductionParams(use_apparent_pairs=False).use_apparent_pairs is False


def test_apparent_tristate_reduce_kwargs():
    # the oin.reduce kwargs layer must route all three states, None included
    a = np.random.default_rng(2030).standard_normal((8, 8)).astype(np.float64)
    fil = oin.cube_filtration(a, n_threads=4)
    assert oin.reduce(fil, n_threads=4, compute_v=False, use_apparent_pairs=True).n_apparent_pairs() > 0
    assert oin.reduce(fil, n_threads=4, compute_v=False, use_apparent_pairs=False).n_apparent_pairs() == 0
    # cube hom R-only is Auto's pure-win corner, so None turns it on
    assert oin.reduce(fil, n_threads=4, compute_v=False, use_apparent_pairs=None).n_apparent_pairs() > 0


@pytest.mark.parametrize("dualize", [False, True])
def test_apparent_activation_telemetry(dualize):
    # guard against a silent gate regression: with the flag ON on the fused
    # multi-threaded path, both the cube and the slim Freudenthal grid
    # filtrations must actually take the apparent path, on 2D and 3D grids
    arrays = [np.random.default_rng(7).standard_normal((12, 11)).astype(np.float64),
              np.random.default_rng(8).standard_normal((8, 7, 6)).astype(np.float64)]
    for a in arrays:
        for reducer in (_reduce, _reduce_fr):
            name = f"{reducer.__name__} {a.shape}"
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
