import numpy as np
import pytest
import oineus as oin


def _grid_fil(n=24, seed=0):
    rng = np.random.default_rng(seed)
    return oin.freudenthal_filtration(data=np.ascontiguousarray(rng.random((n, n))))


def _classic_dgms(fil, compute_v=True, n_threads=1, ndims=3):
    dcmp = oin.Decomposition(fil, False)
    p = oin.ReductionParams()
    p.compute_v = compute_v
    p.n_threads = n_threads
    dcmp.reduce(p)
    return [dcmp.diagram(fil).in_dimension(d) for d in range(ndims)]


def _sorted(dgm):
    return dgm[np.lexsort(dgm.T)] if len(dgm) else dgm


def _dgms_equal(a, b, ndims=3):
    for d in range(ndims):
        x, y = _sorted(a[d]), _sorted(b[d])
        if x.shape != y.shape or not np.allclose(x, y):
            return False
    return True


def _finite(dgm):
    # rows with both coordinates finite (drop essential/infinite-death points)
    return dgm[np.isfinite(dgm).all(axis=1)] if len(dgm) else dgm.reshape(0, 2)


def _exact_points(dgms_obj, ndims):
    # full diagram points (birth, death, birth_index, death_index) per dimension, sorted,
    # so two extractions can be compared for EXACT identity (not just value-closeness)
    out = []
    for d in range(ndims):
        pts = dgms_obj.in_dimension(d, as_numpy=False)
        rows = sorted((p.birth, p.death, p.birth_index, p.death_index) for p in pts)
        out.append(rows)
    return out


@pytest.mark.parametrize("builder", ["freudenthal", "cube", "vr"])
@pytest.mark.parametrize("dualize", [False, True])
def test_diagram_pod_serial_parallel_invariant(builder, dualize):
    # The non-relative diagram extraction was de-templated onto a cell-type-erased
    # FiltrationValues view (one compiled function for all cell types). This pins the
    # invariant that matters: for one reduced decomposition, the serial path, the
    # taskflow-parallel path at several thread counts, and diagram_serial() all yield the
    # EXACT same diagram (same points AND same birth/death indices) -- across cell types.
    rng = np.random.default_rng(1)
    if builder == "freudenthal":
        fil = oin.freudenthal_filtration(data=np.ascontiguousarray(rng.random((20, 20))))
    elif builder == "cube":
        fil = oin.cube_filtration(np.ascontiguousarray(rng.random((16, 16))))
    else:
        fil = oin.vr_filtration(np.ascontiguousarray(rng.random((40, 3))), max_dim=2, max_diameter=0.6)
    ndims = fil.max_dim + 1

    dcmp = oin.Decomposition(fil, dualize)
    p = oin.ReductionParams()
    p.n_threads = 4
    dcmp.reduce(p)

    ref = _exact_points(dcmp.diagram(fil, include_inf_points=True, n_threads=1), ndims)
    for nt in (2, 4):
        assert _exact_points(dcmp.diagram(fil, include_inf_points=True, n_threads=nt), ndims) == ref
    assert _exact_points(dcmp.diagram_serial(fil, include_inf_points=True), ndims) == ref


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


def test_reduce_timings_on_decomposition():
    # the per-phase timings live on the decomposition, not on the params
    fil = _grid_fil(n=48, seed=6)
    p = oin.ReductionParams()
    p.compute_v = False
    p.n_threads = 4
    dcmp = oin.reduce(fil, p)

    assert dcmp.timings.reduce >= 0.0
    assert dcmp.timings.total >= dcmp.timings.reduce

    # the removed params outputs must fail loudly, pointing at dcmp.timings
    with pytest.raises(RuntimeError, match="dcmp.timings"):
        _ = p.timings
    with pytest.raises(RuntimeError, match="dcmp.timings"):
        _ = p.elapsed


def test_reduce_kwargs_equivalent_to_params():
    # kwargs are a convenience spelling of ReductionParams fields; both the
    # free reduce and the Decomposition method must accept them and produce
    # the same diagrams as the explicit-params call
    fil = _grid_fil(seed=9)

    p = oin.ReductionParams()
    p.n_threads = 2
    p.compute_v = True
    dcmp_params = oin.reduce(fil, p)
    dcmp_kwargs = oin.reduce(fil, n_threads=2, compute_v=True)

    for d in range(3):
        got = np.asarray(dcmp_kwargs.diagram(fil).in_dimension(d))
        want = np.asarray(dcmp_params.diagram(fil).in_dimension(d))
        assert np.array_equal(np.sort(got, axis=0), np.sort(want, axis=0))

    # same timings shape: every phase field is present and non-negative
    for t in (dcmp_params.timings, dcmp_kwargs.timings):
        for phase in ("prepare", "reduce", "bauer", "restore_elz", "copy_back", "copy_pivots"):
            assert getattr(t, phase) >= 0.0
        assert t.total >= 0.0

    # the method form: dcmp.reduce(n_threads=..., compute_u=...) must work
    dcmp_m = oin.Decomposition(fil, False)
    dcmp_m.reduce(n_threads=1, compute_u=True, compute_v=True)
    assert dcmp_m.has_matrix_u()
    assert dcmp_m.has_matrix_v()

    # kwargs must not mutate the caller's params object
    base = oin.ReductionParams()
    dcmp_o = oin.Decomposition(fil, False)
    dcmp_o.reduce(base, compute_v=True)
    assert base.compute_v is False
    assert dcmp_o.has_matrix_v()

    # old-name aliases work as kwargs too (set through the compat property)
    dcmp_a = oin.Decomposition(fil, False)
    dcmp_a.reduce(clearing_opt=False)
    assert dcmp_a.is_reduced


def test_reduce_unknown_kwarg_raises_typeerror():
    fil = _grid_fil(seed=10)
    with pytest.raises(TypeError, match="no_such_field"):
        oin.reduce(fil, no_such_field=1)
    dcmp = oin.Decomposition(fil, False)
    with pytest.raises(TypeError, match="no_such_field"):
        dcmp.reduce(no_such_field=1)


def test_reduce_advanced_kwargs_route_to_advanced():
    # the kwargs layer stays flat: chunk_size / col_repr / dims_to_restore_elz
    # given to reduce() must land in params.advanced
    fil = _grid_fil(seed=11)

    p = oin._apply_reduction_kwargs(None, dict(n_threads=2, chunk_size=64,
                                              col_repr=oin.ColumnRepr.Full,
                                              dims_to_restore_elz=[0, 1]))
    assert p.n_threads == 2
    assert p.advanced.chunk_size == 64
    assert p.advanced.col_repr == oin.ColumnRepr.Full
    assert list(p.advanced.dims_to_restore_elz) == [0, 1]

    # advanced kwargs must not mutate the caller's params object either
    base = oin.ReductionParams()
    p2 = oin._apply_reduction_kwargs(base, dict(chunk_size=base.advanced.chunk_size + 1))
    assert p2.advanced.chunk_size == base.advanced.chunk_size + 1
    assert base == oin.ReductionParams()

    # end to end: reduction with flat advanced kwargs matches the classic path
    oracle = _classic_dgms(fil)
    dcmp = oin.reduce(fil, n_threads=2, chunk_size=64, col_repr=oin.ColumnRepr.Full)
    fused = [dcmp.diagram(fil).in_dimension(d) for d in range(3)]
    assert _dgms_equal(oracle, fused)

    # the method form with dims_to_restore_elz (needs compute_v)
    dcmp_m = oin.Decomposition(fil, False)
    dcmp_m.reduce(n_threads=2, compute_v=True, dims_to_restore_elz=[0, 1], chunk_size=32)
    assert dcmp_m.is_reduced
    assert dcmp_m.has_matrix_v()


@pytest.mark.skipif(np.dtype("float32") not in oin._dtype.REAL_MODULES,
                    reason="extension built without the float32 backend")
def test_reduce_timings_float32_backend():
    # timings must be readable on a float32 filtration's decomposition too
    # (the Decomposition class and its timings are Real-independent)
    rng = np.random.default_rng(6)
    fil = oin.freudenthal_filtration(data=np.ascontiguousarray(rng.random((16, 16), dtype=np.float32)))
    dcmp = oin.reduce(fil, oin.ReductionParams())
    assert dcmp.timings.reduce >= 0.0
    assert dcmp.timings.total >= dcmp.timings.reduce


@pytest.mark.skipif(np.dtype("float32") not in oin._dtype.REAL_MODULES,
                    reason="extension built without the float32 backend")
def test_reduce_float32_advanced_kwargs():
    # ReductionParamsAdvanced is Real-independent: a float32 decomposition
    # must reduce with flat advanced kwargs like the float64 one
    rng = np.random.default_rng(8)
    fil = oin.freudenthal_filtration(data=np.ascontiguousarray(rng.random((16, 16), dtype=np.float32)))
    dcmp = oin.reduce(fil, n_threads=2, chunk_size=64, col_repr=oin.ColumnRepr.Full)
    assert dcmp.is_reduced
    assert len(dcmp.diagram(fil).in_dimension(0)) > 0


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


@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("compute_v", [False, True])
def test_reduce_vr_matches_dionysus_oracle(dualize, n_threads, compute_v):
    # Fused oin.reduce on a Vietoris-Rips (Simplex) filtration, cross-checked
    # against an INDEPENDENT oracle (dionysus). This exercises the Simplex
    # reduce overload -- the other fused tests only build grids, which hit the
    # Cube overloads -- and catches any bug shared by Oineus's fused and classic
    # paths (which a fused-vs-classic check cannot).
    dion = pytest.importorskip("dionysus")
    pts = np.ascontiguousarray(np.random.default_rng(0).random((30, 3)))
    R = 1.0
    fil = oin.vr_filtration(pts, max_dim=2, max_diameter=R, n_threads=1)

    p = oin.ReductionParams()
    p.compute_v = compute_v
    p.n_threads = n_threads
    dcmp = oin.reduce(fil, p, dualize)

    f = dion.fill_rips(pts.astype("f4"), 2, float(R))
    m = dion.homology_persistence(f)
    dion_dgms = dion.init_diagrams(m, f)

    for d in (0, 1):
        oin_fin = _finite(dcmp.diagram(fil).in_dimension(d)).astype(np.float32)
        dion_fin = np.array([[q.birth, q.death] for q in dion_dgms[d] if q.death < np.inf],
                            dtype=np.float32).reshape(-1, 2)
        bd = dion.bottleneck_distance(dion.Diagram(oin_fin), dion.Diagram(dion_fin))
        assert bd < 1e-5, f"dim {d}: bottleneck {bd}"

    # exactly one essential H0 component for a (connected) point cloud
    h0 = dcmp.diagram(fil).in_dimension(0)
    assert int(np.isinf(h0).any(axis=1).sum()) == 1


@pytest.mark.parametrize("ndim", [1, 3])
@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("compute_v", [False, True])
def test_reduce_cube_1d_3d_match_classic(ndim, n_threads, compute_v):
    # The other fused tests only build 2D grids (the Cube_2D reduce overload);
    # cover the Cube_1D and Cube_3D overloads too.
    shape = (40,) if ndim == 1 else (8, 8, 8)
    data = np.ascontiguousarray(np.random.default_rng(0).random(shape))
    fil = oin.freudenthal_filtration(data=data)
    ndims = ndim + 1
    oracle = _classic_dgms(fil, ndims=ndims)

    p = oin.ReductionParams()
    p.compute_v = compute_v
    p.n_threads = n_threads
    dcmp = oin.reduce(fil, p)
    fused = [dcmp.diagram(fil).in_dimension(d) for d in range(ndims)]
    assert _dgms_equal(oracle, fused, ndims=ndims)


@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("compute_v", [False, True])
def test_reduce_dim0_only(n_threads, compute_v):
    # Vertices-only filtration: every boundary column is empty, so the fused
    # parallel R-only path is "pivots-only with an empty pairing". The H0
    # diagram is then all-essential and must match the classic path; nothing
    # should crash on this degenerate input.
    pts = np.ascontiguousarray(np.random.default_rng(0).random((10, 2)))
    fil = oin.vr_filtration(pts, max_dim=0, max_diameter=1.0, n_threads=1)

    p = oin.ReductionParams()
    p.compute_v = compute_v
    p.n_threads = n_threads
    dcmp = oin.reduce(fil, p)

    h0 = dcmp.diagram(fil).in_dimension(0)
    # one essential point per vertex, all with infinite death, no finite pairs
    assert len(h0) == fil.size()
    assert np.isinf(h0[:, 1]).all()
    assert len(_finite(h0)) == 0

    classic = oin.Decomposition(fil, False)
    pc = oin.ReductionParams()
    pc.compute_v = compute_v
    classic.reduce(pc)
    assert len(classic.diagram(fil).in_dimension(0)) == len(h0)


@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("compute_v", [False, True])
def test_reduce_empty_filtration_no_crash(n_threads, compute_v):
    # Size-0 filtration: the fused path must not crash. There are no cells of
    # any dimension, so a subsequent diagram request raises a clean error
    # (not a segfault).
    fil = oin.Filtration([])
    assert fil.size() == 0

    p = oin.ReductionParams()
    p.compute_v = compute_v
    p.n_threads = n_threads
    dcmp = oin.reduce(fil, p)  # must return without crashing

    with pytest.raises(Exception):
        dcmp.diagram(fil).in_dimension(0)
