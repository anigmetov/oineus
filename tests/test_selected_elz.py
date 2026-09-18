import gc
import os
import pickle

import numpy as np
import pytest

import oineus as oin


REPRS = [oin.ColumnRepr.Set, oin.ColumnRepr.Heap, oin.ColumnRepr.Full, oin.ColumnRepr.BitTree]


@pytest.fixture(scope="module")
def filtration():
    return oin.freudenthal_filtration(np.random.default_rng(7).random((10, 10)), n_threads=1)


def reduce(fil, dualize=False, fused=False, **kwargs):
    params = oin.ReductionParams(compute_v=True, **kwargs)
    if fused:
        return oin.reduce(fil, params, dualize)
    result = oin.Decomposition(fil, dualize=dualize)
    result.reduce(params)
    return result


def columns(dcmp, n):
    # Per-column reads preserve the retained working representation
    return [dcmp.r_col(j) for j in range(n)], [dcmp.v_col(j) for j in range(n)]


def assert_factorization(dcmp, boundary):
    residual = boundary @ dcmp.v_as_csc() - dcmp.r_as_csc()
    assert np.all(residual.data % 2 == 0)
    assert dcmp.factorization_valid


@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("clearing", [False, True])
@pytest.mark.parametrize("reduce_threads", [1, 4])
@pytest.mark.parametrize("restore_threads", [1, 2, 4])
@pytest.mark.parametrize("col_repr", REPRS)
def test_selected_exact_and_untouched(filtration, dualize, fused, clearing,
                                      reduce_threads, restore_threads, col_repr):
    n = filtration.size()
    oracle = reduce(filtration, dualize, n_threads=1, use_clearing=False)
    expected_r, expected_v = columns(oracle, n)
    params = oin.ReductionParams(compute_v=True, n_threads=reduce_threads, use_clearing=clearing)
    params.advanced.col_repr = col_repr
    if fused:
        dcmp = oin.reduce(filtration, params, dualize)
    else:
        dcmp = oin.Decomposition(filtration, dualize)
        dcmp.reduce(params)
    original_r, original_v = columns(dcmp, n)
    flags = [dcmp.negative_v_elz_in_dim(d) for d in range(3)]
    requested = list(range(2, n, 7))
    assert any(original_r[j] for j in requested)
    assert any(not original_r[j] for j in requested)
    request = requested[::-1] + requested[:4]
    assert dcmp.restore_elz_columns(request, n_threads=restore_threads) is None
    actual_r, actual_v = columns(dcmp, n)
    selected = set(requested)
    for j in range(n):
        assert actual_r[j] == (expected_r if j in selected else original_r)[j]
        assert actual_v[j] == (expected_v if j in selected else original_v)[j]
    assert flags == [dcmp.negative_v_elz_in_dim(d) for d in range(3)]
    assert dcmp.restore_elz_columns(request, n_threads=restore_threads) is None
    assert columns(dcmp, n) == (actual_r, actual_v)
    assert_factorization(dcmp, oracle.d_as_csc())
    # Subsequent overlapping and exhaustive requests can read previously restored sources
    dcmp.restore_elz_columns(list(range(n)), n_threads=restore_threads)
    assert columns(dcmp, n) == (expected_r, expected_v)
    assert_factorization(dcmp, oracle.d_as_csc())
    assert not dcmp.has_matrix_u()


@pytest.mark.parametrize("fused", [False, True])
def test_invalid_requests_and_empty_are_noops(filtration, fused):
    dcmp = reduce(filtration, fused=fused, n_threads=4, compute_u=True)
    before = dcmp.r_data, dcmp.v_data, dcmp.u_data_t
    for indices in ([0, -1], [2, filtration.size()], [-100]):
        with pytest.raises(IndexError, match="column index"):
            dcmp.restore_elz_columns(indices, n_threads=4)
        assert (dcmp.r_data, dcmp.v_data, dcmp.u_data_t) == before
        assert dcmp.has_full_matrix_u()
    for threads in (0, -1):
        with pytest.raises(ValueError, match="n_threads"):
            dcmp.restore_elz_columns([1], n_threads=threads)
    assert dcmp.restore_elz_columns([]) is None
    assert (dcmp.r_data, dcmp.v_data, dcmp.u_data_t) == before


def test_prerequisites_and_v_only_guard(filtration):
    dcmp = oin.Decomposition(filtration, False)
    with pytest.raises(RuntimeError, match="not reduced"):
        dcmp.restore_elz_columns([1])
    dcmp.reduce(oin.ReductionParams(compute_v=False, n_threads=4))
    with pytest.raises(RuntimeError, match="V matrix"):
        dcmp.restore_elz_columns([1])
    dcmp = reduce(filtration, n_threads=4)
    dcmp.restore_elz(v_only=True)
    assert not dcmp.factorization_valid
    before = dcmp.__getstate__()[:4]
    with pytest.raises(RuntimeError, match="R = D V"):
        dcmp.restore_elz_columns([1])
    for getter in (dcmp.r_col, dcmp.v_col):
        with pytest.raises(RuntimeError, match="R = D V"):
            getter(0)
    assert dcmp.__getstate__()[:4] == before


@pytest.mark.parametrize("partial_u", [False, True])
def test_changed_v_discards_u_and_dynamic_indices(filtration, partial_u):
    dcmp = reduce(filtration, n_threads=4, compute_u=True)
    oracle = reduce(filtration, n_threads=1, use_clearing=False)
    target = next(j for j in range(filtration.size()) if dcmp.v_col(j) != oracle.v_col(j))
    dcmp.make_dynamic()
    if partial_u:
        # A completed dimension enables a partial row solve; other dimensions remain noncanonical
        dcmp.restore_elz(2)
        dcmp.compute_partial_u_rows(filtration, rows=[dcmp.dim_first[2]],
                                    bounds=[1e9], dim=2, cmp="above")
        target = next(j for j in range(filtration.size()) if dcmp.v_col(j) != oracle.v_col(j))
        assert not dcmp.has_full_matrix_u()
    assert dcmp.has_matrix_u()
    dcmp.restore_elz_columns([target], n_threads=4)
    assert not dcmp.has_matrix_u()
    assert not dcmp.has_full_matrix_u()
    assert not dcmp.is_dynamic()
    assert dcmp.n_computed_u_rows == dcmp.n_valid_u_rows == 0
    with pytest.raises(IndexError):
        dcmp.u_row(target)
    assert_factorization(dcmp, oracle.d_as_csc())
    dcmp.make_dynamic()
    assert dcmp.is_dynamic()


def test_noop_preserves_complete_u_and_flags(filtration):
    dcmp = reduce(filtration, n_threads=1, use_clearing=False, compute_u=True)
    before = dcmp.u_data_t
    dcmp.make_dynamic()
    dcmp.restore_elz_columns(list(range(0, filtration.size(), 3)), n_threads=4)
    assert dcmp.u_data_t == before
    assert dcmp.has_full_matrix_u()
    assert dcmp.is_dynamic()
    assert all(dcmp.negative_v_elz_in_dim(d) for d in range(3))


@pytest.mark.parametrize("dualize", [False, True])
def test_subset_does_not_enable_dimension_u_solve(filtration, dualize):
    dcmp = reduce(filtration, dualize, n_threads=4)
    row = filtration.size() - 1 if dualize else 0
    dcmp.restore_elz_columns([row], n_threads=4)
    with pytest.raises(RuntimeError, match="ELZ"):
        dcmp.compute_partial_u_rows(filtration, rows=[row], bounds=[1e9],
                                    dim=0, cmp="below" if dualize else "above")
    restored = pickle.loads(pickle.dumps(dcmp))
    assert columns(restored, filtration.size()) == columns(dcmp, filtration.size())
    assert not restored.negative_v_elz_in_dim(0)


def test_column_snapshots_and_unavailable_matrices(filtration):
    dcmp = reduce(filtration, n_threads=4)
    n = filtration.size()
    r, v, d = dcmp.r_data, dcmp.v_data, dcmp.d_data
    for j in range(n):
        assert dcmp.r_col(j) == r[j]
        assert dcmp.v_col(j) == v[j]
        assert dcmp.d_col(j) == d[j]
    snapshots = dcmp.r_col(0), dcmp.v_col(0), dcmp.d_col(0)
    assert snapshots[0] == []  # A valid zero column is available
    snapshots[1].append(n + 1)
    assert dcmp.v_col(0) == v[0]
    dcmp.restore_elz_columns(list(range(n)), n_threads=4)
    assert snapshots[1] == v[0] + [n + 1]
    for getter in (dcmp.r_col, dcmp.v_col, dcmp.d_col):
        with pytest.raises(IndexError):
            getter(n)
        with pytest.raises((IndexError, TypeError)):
            getter(-1)
    del dcmp
    gc.collect()
    assert snapshots[2] == d[0]
    fused = reduce(filtration, fused=True, n_threads=4)
    with pytest.raises(RuntimeError, match="D is unavailable"):
        fused.d_col(0)
    lean = oin.reduce(filtration, oin.ReductionParams(n_threads=4, compute_v=False), False)
    with pytest.raises(RuntimeError, match="R is unavailable"):
        lean.r_col(0)
    with pytest.raises(RuntimeError, match="V matrix"):
        lean.v_col(0)


@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("threads", [1, 4])
def test_noncanonical_negative_sources_need_not_be_requested(filtration, dualize, threads):
    dcmp = reduce(filtration, dualize, n_threads=1, use_clearing=False)
    expected_r, expected_v = dcmp.r_data, dcmp.v_data
    r, v = [set(c) for c in expected_r], [set(c) for c in expected_v]
    changed_negative = []
    # Add earlier columns without changing any pivot. This constructs a valid
    # reduced factorization with noncanonical negative sources as well as targets.
    for j in range(len(r)):
        candidates = [k for k in range(j) if r[k] and r[j] and max(r[k]) < max(r[j])]
        if candidates:
            k = candidates[-1]
            r[j].symmetric_difference_update(r[k])
            v[j].symmetric_difference_update(v[k])
            changed_negative.append(j)
    assert len(changed_negative) > 20
    dcmp.r_data = [sorted(c) for c in r]
    dcmp.v_data = [sorted(c) for c in v]
    assert_factorization(dcmp, dcmp.d_as_csc())
    targets = changed_negative[1::4]
    before_r, before_v = dcmp.r_data, dcmp.v_data
    dcmp.restore_elz_columns(targets[::-1], n_threads=threads)
    for j in range(len(r)):
        assert dcmp.r_col(j) == (expected_r if j in targets else before_r)[j]
        assert dcmp.v_col(j) == (expected_v if j in targets else before_v)[j]
    assert_factorization(dcmp, dcmp.d_as_csc())


@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("builder", [oin.cube_filtration, oin.freudenthal_filtration])
def test_apparent_columns_restoration_and_access(dualize, builder):
    fil = builder(np.random.default_rng(17).random((12, 12)), n_threads=1)
    params = oin.ReductionParams(n_threads=4, compute_v=True, use_apparent_pairs=True)
    dcmp = oin.reduce(fil, params, dualize)
    assert dcmp.n_apparent_pairs() > 0
    count = dcmp.n_apparent_pairs()
    with pytest.raises(IndexError):
        dcmp.restore_elz_columns([0, fil.size()], n_threads=4)
    assert dcmp.n_apparent_pairs() == count
    dcmp.restore_elz_columns([], n_threads=4)
    assert dcmp.n_apparent_pairs() == count
    pairings = [dcmp.diagram(fil).index_diagram_in_dimension(d) for d in range(2)]
    oracle = reduce(fil, dualize, n_threads=1, use_clearing=False)
    selected = list(range(3, fil.size(), 7))
    dcmp.restore_elz_columns(selected, n_threads=4)
    for j in selected:
        assert dcmp.r_col(j) == oracle.r_col(j)
        assert dcmp.v_col(j) == oracle.v_col(j)
    for dim in range(2):
        np.testing.assert_array_equal(dcmp.diagram(fil).index_diagram_in_dimension(dim), pairings[dim])
    assert_factorization(dcmp, oracle.d_as_csc())


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGINT handler")
@pytest.mark.parametrize("threads", [1, 4])
def test_interrupt_during_call_releases_gil_and_preserves_columns(threads):
    import subprocess
    import sys
    import textwrap

    # Isolate SIGINT so a delayed signal cannot interrupt the pytest process
    script = textwrap.dedent("""
        import hashlib
        import os
        import signal
        import threading
        import time
        import numpy as np
        import oineus as oin

        fil = oin.freudenthal_filtration(np.random.default_rng(7).random((256, 256)), n_threads=1)
        dcmp = oin.reduce(fil, oin.ReductionParams(compute_v=True, n_threads=4), False)
        indices = list(range(fil.size()))
        def fingerprint():
            digest = hashlib.sha256()
            for j in indices:
                digest.update(repr((dcmp.r_col(j), dcmp.v_col(j))).encode())
            return digest.digest()
        before = fingerprint()
        timer = threading.Timer(0.01, lambda: os.kill(os.getpid(), signal.SIGINT))
        started = time.monotonic()
        caught = False
        timer.start()
        try:
            try:
                dcmp.restore_elz_columns(indices, n_threads=THREAD_COUNT)
            except KeyboardInterrupt:
                caught = True
        finally:
            timer.cancel()
            timer.join()
        assert caught, 'restoration finished before the interrupt'
        assert time.monotonic() - started < 3
        assert fingerprint() == before
        assert dcmp.factorization_valid
        assert not dcmp.has_matrix_u()
        dcmp.restore_elz_columns([10, 20, 30], n_threads=THREAD_COUNT)
    """).replace("THREAD_COUNT", str(threads))
    child = subprocess.run([sys.executable, "-c", script], env=os.environ.copy(),
                           text=True, capture_output=True, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
