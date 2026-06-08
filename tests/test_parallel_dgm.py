"""Parallel diagram extraction must match the serial oracle.

dcmp.diagram(fil, n_threads=t) is compared against dcmp.diagram_serial(fil)
as a per-dimension multiset (order-independent: lexsort then compare) of both
the (birth, death) values and the (birth_index, death_index) pairs. Coverage:
homology and cohomology (dualize), negate, include_inf_points on/off,
zero-persistence, the classic (r_data) and fused (pivots-only) states, and
thread counts {1, 2, 4, 8}, plus cross-thread determinism.

Inputs are sized so n_cols exceeds the parallel function's small-input serial
fallback threshold, i.e. the parallel path is actually exercised.
"""
import numpy as np
import pytest
import oineus as oin

N_THREADS = [1, 2, 4, 8]


def _grid_fil(n=40, seed=0, negate=False):
    rng = np.random.default_rng(seed)
    data = np.ascontiguousarray(rng.random((n, n)))
    return oin.freudenthal_filtration(data=data, negate=negate, max_dim=2)


def _vr_fil(n=48, seed=0, dim=2):
    rng = np.random.default_rng(seed)
    pts = np.ascontiguousarray(rng.random((n, 3)))
    return oin.vr_filtration(pts, max_dim=dim)


def _make_dcmp(fil, dualize, fused, reduce_threads=4):
    p = oin.ReductionParams()
    p.n_threads = reduce_threads
    if fused:
        return oin.reduce(fil, p, dualize)
    dcmp = oin.Decomposition(fil, dualize)
    dcmp.reduce(p)
    return dcmp


def _sort_rows(a):
    a = np.asarray(a)
    if a.ndim != 2:
        a = a.reshape(-1, 2)
    if a.shape[0] == 0:
        return a
    return a[np.lexsort(a.T)]


def _vals(dcmp, fil, n_threads, inf):
    d = dcmp.diagram(fil, include_inf_points=inf, n_threads=n_threads)
    return [d.in_dimension(k) for k in range(3)]


def _vals_serial(dcmp, fil, inf):
    d = dcmp.diagram_serial(fil, include_inf_points=inf)
    return [d.in_dimension(k) for k in range(3)]


def _idx(dcmp, fil, n_threads, inf):
    d = dcmp.diagram(fil, include_inf_points=inf, n_threads=n_threads)
    return [d.index_diagram_in_dimension(k) for k in range(3)]


def _idx_serial(dcmp, fil, inf):
    d = dcmp.diagram_serial(fil, include_inf_points=inf)
    return [d.index_diagram_in_dimension(k) for k in range(3)]


def _assert_vals_equal(a, b, msg=""):
    assert len(a) == len(b)
    for d in range(len(a)):
        x, y = _sort_rows(a[d]).astype(float), _sort_rows(b[d]).astype(float)
        assert x.shape == y.shape, f"{msg} dim {d}: shape {x.shape} != {y.shape}"
        if x.size:
            assert np.allclose(x, y, equal_nan=True), f"{msg} dim {d}: values differ"


def _assert_idx_equal(a, b, msg=""):
    assert len(a) == len(b)
    for d in range(len(a)):
        x, y = _sort_rows(a[d]), _sort_rows(b[d])
        assert x.shape == y.shape, f"{msg} dim {d}: idx shape {x.shape} != {y.shape}"
        if x.size:
            assert np.array_equal(x, y), f"{msg} dim {d}: index pairs differ"


@pytest.mark.parametrize("n_threads", N_THREADS)
@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize("inf", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_grid_values_match_serial(n_threads, dualize, negate, inf, fused):
    fil = _grid_fil(negate=negate)
    dcmp = _make_dcmp(fil, dualize, fused)
    _assert_vals_equal(_vals_serial(dcmp, fil, inf), _vals(dcmp, fil, n_threads, inf),
                       f"grid neg={negate} coh={dualize} inf={inf} fused={fused} nt={n_threads}")


@pytest.mark.parametrize("n_threads", N_THREADS)
@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_grid_index_pairs_match_serial(n_threads, dualize, fused):
    # Index pairs catch a swap/dim bug that happens to preserve values.
    fil = _grid_fil()
    dcmp = _make_dcmp(fil, dualize, fused)
    _assert_idx_equal(_idx_serial(dcmp, fil, True), _idx(dcmp, fil, n_threads, True),
                      f"grid-idx coh={dualize} fused={fused} nt={n_threads}")


@pytest.mark.parametrize("n_threads", N_THREADS)
@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_vr_match_serial(n_threads, dualize, fused):
    fil = _vr_fil()
    dcmp = _make_dcmp(fil, dualize, fused)
    _assert_vals_equal(_vals_serial(dcmp, fil, True), _vals(dcmp, fil, n_threads, True),
                       f"vr coh={dualize} fused={fused} nt={n_threads}")
    _assert_idx_equal(_idx_serial(dcmp, fil, True), _idx(dcmp, fil, n_threads, True),
                      f"vr-idx coh={dualize} fused={fused} nt={n_threads}")


@pytest.mark.parametrize("n_threads", [2, 4, 8])
@pytest.mark.parametrize("dualize", [False, True])
def test_zero_persistence_matches_serial(n_threads, dualize):
    fil = _grid_fil()
    dcmp = _make_dcmp(fil, dualize, fused=False)
    s = [dcmp.zero_pers_diagram(fil, n_threads=1).in_dimension(k) for k in range(3)]
    p = [dcmp.zero_pers_diagram(fil, n_threads=n_threads).in_dimension(k) for k in range(3)]
    _assert_vals_equal(s, p, f"zero-pers coh={dualize} nt={n_threads}")


@pytest.mark.parametrize("dualize", [False, True])
def test_inf_point_count_matches(dualize):
    # Per-dimension count of essential (death == inf) points must match serial,
    # under both homology and cohomology (the essential branch does not shift dim
    # or swap birth/death, unlike the finite branch).
    fil = _grid_fil()
    dcmp = _make_dcmp(fil, dualize, fused=False)
    s = _vals_serial(dcmp, fil, True)
    p = _vals(dcmp, fil, 8, True)
    for d in range(3):
        ns = int(np.sum(~np.isfinite(np.asarray(s[d]).reshape(-1, 2)[:, 1]))) if np.asarray(s[d]).size else 0
        npar = int(np.sum(~np.isfinite(np.asarray(p[d]).reshape(-1, 2)[:, 1]))) if np.asarray(p[d]).size else 0
        assert ns == npar, f"coh={dualize} dim {d}: inf count {ns} != {npar}"


@pytest.mark.parametrize("kind", ["grid", "vr"])
@pytest.mark.parametrize("dualize", [False, True])
def test_determinism_across_thread_counts(kind, dualize):
    fil = _grid_fil(seed=3) if kind == "grid" else _vr_fil(seed=3)
    dcmp = _make_dcmp(fil, dualize, fused=False)
    base_v = _vals(dcmp, fil, 1, True)
    base_i = _idx(dcmp, fil, 1, True)
    for nt in [2, 4, 8]:
        _assert_vals_equal(base_v, _vals(dcmp, fil, nt, True), f"det {kind} coh={dualize} nt={nt}")
        _assert_idx_equal(base_i, _idx(dcmp, fil, nt, True), f"det-idx {kind} coh={dualize} nt={nt}")


@pytest.mark.parametrize("dualize", [False, True])
def test_medium_grid_stress(dualize):
    # Larger grid at high thread count to exercise chunk boundaries.
    fil = _grid_fil(n=96, seed=7)
    dcmp = _make_dcmp(fil, dualize, fused=True)
    _assert_vals_equal(_vals_serial(dcmp, fil, True), _vals(dcmp, fil, 8, True),
                       f"stress coh={dualize}")
    _assert_idx_equal(_idx_serial(dcmp, fil, True), _idx(dcmp, fil, 8, True),
                      f"stress-idx coh={dualize}")
