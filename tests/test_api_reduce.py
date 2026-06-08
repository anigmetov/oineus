import numpy as np
import pytest
import oineus as oin


def _grid_fil(n=24, seed=0):
    rng = np.random.default_rng(seed)
    return oin.freudenthal_filtration(data=np.ascontiguousarray(rng.random((n, n))))


def _classic_dgms(fil, compute_v=True, n_threads=1):
    dcmp = oin.Decomposition(fil, False)
    p = oin.ReductionParams()
    p.compute_v = compute_v
    p.n_threads = n_threads
    dcmp.reduce(p)
    return [dcmp.diagram(fil).in_dimension(d) for d in range(3)]


def _sorted(dgm):
    return dgm[np.lexsort(dgm.T)] if len(dgm) else dgm


def _dgms_equal(a, b):
    for d in range(3):
        x, y = _sorted(a[d]), _sorted(b[d])
        if x.shape != y.shape or not np.allclose(x, y):
            return False
    return True


@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("n_threads", [1, 2, 4])
@pytest.mark.parametrize("compute_v", [False, True])
def test_reduce_diagrams_match_classic(dualize, n_threads, compute_v):
    # Fused oin.reduce must give the same diagrams as the classic
    # ctor+reduce path, across dualize / threads / compute_v.
    fil = _grid_fil()
    oracle = _classic_dgms(fil)

    p = oin.ReductionParams()
    p.compute_v = compute_v
    p.n_threads = n_threads
    dcmp = oin.reduce(fil, p, dualize)
    fused = [dcmp.diagram(fil).in_dimension(d) for d in range(3)]

    assert _dgms_equal(oracle, fused)


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_reduce_rv_is_valid_decomposition(n_threads):
    # With compute_v the fused path keeps R and V; D*V == R must hold
    # (sanity_check with the boundary supplied). Parallel V is not unique,
    # so we check validity, not equality to the serial V.
    fil = _grid_fil(seed=1)
    D = fil.boundary_matrix()

    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = n_threads
    dcmp = oin.reduce(fil, p)

    assert dcmp.sanity_check(D)
    assert len(dcmp.v_data) == fil.size()


def test_reduce_pivots_only_r_data_raises():
    # Fused parallel R-only frees the reduced columns (pivots-only state):
    # the diagram still works, but r_data / r_as_csc must raise a clear
    # error rather than silently returning an empty matrix.
    fil = _grid_fil(seed=2)
    p = oin.ReductionParams()
    p.compute_v = False
    p.n_threads = 4
    dcmp = oin.reduce(fil, p)

    # diagram works from pivots
    assert len(dcmp.diagram(fil).in_dimension(0)) > 0

    with pytest.raises(Exception):
        _ = dcmp.r_data
    with pytest.raises(Exception):
        _ = dcmp.r_as_csc()


def test_reduce_serial_r_only_keeps_r_data():
    # Fused serial reduces in place, so r_data holds the reduced R.
    fil = _grid_fil(seed=3)
    p = oin.ReductionParams()
    p.compute_v = False
    p.n_threads = 1
    dcmp = oin.reduce(fil, p)

    assert len(dcmp.r_data) == fil.size()
    assert dcmp.r_as_csc().shape == (fil.size(), fil.size())


def test_reduce_rv_parallel_exposes_r_and_v():
    # compute_v keeps both R and V even in parallel (lazy-materialized on access).
    fil = _grid_fil(seed=4)
    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = 4
    dcmp = oin.reduce(fil, p)

    assert len(dcmp.r_data) == fil.size()
    assert len(dcmp.v_data) == fil.size()


def test_reduce_sanity_check_without_d_returns_false():
    # The fused path does not hold D; the no-argument sanity_check must
    # fail cleanly (return False) instead of pretending to verify.
    fil = _grid_fil(seed=5)
    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = 2
    dcmp = oin.reduce(fil, p)

    assert dcmp.sanity_check() is False
    assert dcmp.sanity_check(fil.boundary_matrix()) is True


def test_reduce_timings_propagate():
    # oin.reduce takes params by reference, so the per-phase timings are
    # written back to the caller's object.
    fil = _grid_fil(n=48, seed=6)
    p = oin.ReductionParams()
    p.compute_v = False
    p.n_threads = 4
    oin.reduce(fil, p)

    assert p.timings.reduce >= 0.0
    assert p.elapsed >= 0.0


def test_reduce_keep_working_diagram_then_materialize():
    # Parallel RV keeps the working form: the diagram reads pivots (no
    # materialization), and a later v_data/r_data access materializes lazily,
    # giving a valid D*V == R decomposition.
    fil = _grid_fil(seed=7)
    D = fil.boundary_matrix()
    oracle = _classic_dgms(fil)

    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = 4
    dcmp = oin.reduce(fil, p)

    # diagram works straight from pivots, before any matrix access
    got = [dcmp.diagram(fil).in_dimension(d) for d in range(3)]
    assert _dgms_equal(got, oracle)

    # first matrix access materializes; the decomposition stays valid
    assert len(dcmp.v_data) == fil.size()
    assert dcmp.sanity_check(D)


def test_reduce_keep_working_clone():
    # clone() copies via the C++ copy ctor, which materializes the source's
    # working form first; both the clone and the original must be valid.
    fil = _grid_fil(seed=8)
    D = fil.boundary_matrix()

    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = 4
    dcmp = oin.reduce(fil, p)

    clone = dcmp.clone()
    assert clone.sanity_check(D)
    assert len(clone.r_data) == fil.size()
    # original is independently valid
    assert dcmp.sanity_check(D)


def test_reduce_keep_working_pickle_roundtrip():
    # Pickling a keep-working decomposition materializes it first, so the
    # round trip preserves diagrams and validity.
    import pickle
    fil = _grid_fil(seed=9)
    D = fil.boundary_matrix()
    oracle = _classic_dgms(fil)

    p = oin.ReductionParams()
    p.compute_v = True
    p.n_threads = 4
    dcmp = oin.reduce(fil, p)

    dcmp2 = pickle.loads(pickle.dumps(dcmp))
    got = [dcmp2.diagram(fil).in_dimension(d) for d in range(3)]
    assert _dgms_equal(got, oracle)
    assert dcmp2.sanity_check(D)
