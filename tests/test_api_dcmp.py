import numpy as np
import oineus as oin


def _make_simplex_filtration():
    v0 = oin.Simplex([0], 0.0)
    v1 = oin.Simplex([1], 0.1)
    v2 = oin.Simplex([2], 0.2)
    v3 = oin.Simplex([3], 0.3)

    e01 = oin.Simplex([0, 1], 0.4)
    e02 = oin.Simplex([0, 2], 0.5)
    e03 = oin.Simplex([0, 3], 0.6)
    e12 = oin.Simplex([1, 2], 0.7)
    e13 = oin.Simplex([1, 3], 0.8)
    e23 = oin.Simplex([2, 3], 0.9)

    t012 = oin.Simplex([0, 1, 2], 1.0)
    t013 = oin.Simplex([0, 1, 3], 1.1)
    t023 = oin.Simplex([0, 2, 3], 1.2)
    t123 = oin.Simplex([1, 2, 3], 1.3)

    tet = oin.Simplex([0, 1, 2, 3], 1.4)

    simplices = [v0, v1, v2, v3, e01, e02, e03, e12, e13, e23, t012, t013, t023, t123, tet]
    return oin.Filtration(simplices, negate=False, n_threads=1)


def _make_prod_filtration():
    v0 = oin.Simplex([0], 0.0)
    v1 = oin.Simplex([1], 0.1)
    p01 = oin.ProdSimplex(v0, v1, 0.2)
    p10 = oin.ProdSimplex(v1, v0, 0.3)
    return oin.Filtration([p01, p10], negate=False, n_threads=1)


def _make_cube_filtration_1d():
    dom = oin.GridDomain_1D(2)
    v0 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom, value=0.0)
    v1 = oin.Cube_1D(anchor_vertex=[1], spanning_dims=[], domain=dom, value=0.1)
    e01 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom, value=0.5)
    return oin.Filtration([v0, v1, e01], negate=False, n_threads=1)


def test_decomposition_api():
    fil = _make_simplex_filtration()

    dcmp = oin.Decomposition(fil, dualize=False, n_threads=1)

    params = oin.ReductionParams()
    params.n_threads = 1
    params.compute_v = True
    params.compute_u = True
    dcmp.reduce(params)

    dcmp.r_data = dcmp.r_data
    dcmp.v_data = dcmp.v_data
    dcmp.u_data_t = dcmp.u_data_t
    _ = dcmp.d_data

    _ = dcmp.r_as_csc()
    _ = dcmp.v_as_csc()
    _ = dcmp.d_as_csc()
    _ = dcmp.u_as_csr()

    _ = dcmp.dualize
    _ = dcmp.dim_first
    _ = dcmp.dim_last

    _ = dcmp.is_elz(n_threads=1)
    _ = dcmp.n_elz_violators(n_threads=1)
    _ = dcmp.n_elz_violators_in_dim(0, n_threads=1)
    _ = dcmp.is_column_elz(0)

    _ = dcmp.restore_elz()
    _ = dcmp.compute_u_from_v(n_threads=1)

    _ = dcmp.densify_v_for_selinv(rows_to_invert={0}, n_threads=1)
    _ = dcmp.densify_v_for_selinv_with_targets(fil, rows_to_invert=[0], targets=[0.0])

    _ = dcmp.sanity_check()

    _ = dcmp.diagram(fil, include_inf_points=True)
    _ = dcmp.zero_pers_diagram(fil)
    _ = dcmp.filtration_index(0)

    dcmp_from_matrix = oin.Decomposition([], 0, False, True)
    _ = dcmp_from_matrix.dualize

#
# def test_decomposition_diagram_overloads_prod():
#     fil = _make_prod_filtration()
#     dcmp = oin.Decomposition(fil, dualize=False, n_threads=1)
#     dcmp.reduce(oin.ReductionParams())
#
#     _ = dcmp.diagram(fil, include_inf_points=True)
#     _ = dcmp.zero_pers_diagram(fil)


def test_decomposition_diagram_overloads_cube_1d():
    fil = _make_cube_filtration_1d()
    dcmp = oin.Decomposition(fil, dualize=False, n_threads=1)
    dcmp.reduce(oin.ReductionParams())

    _ = dcmp.diagram(fil, include_inf_points=True)
    _ = dcmp.zero_pers_diagram(fil)


def _random_grid_filtration():
    np.random.seed(1)
    return oin.freudenthal_filtration(np.random.rand(12, 12))


def test_compute_u_from_v_1_fused_keep_working():
    import pytest

    # fused RV reduce keeps the working columns (r_data/v_data empty at rest);
    # compute_u_from_v_1 used to index the empty at-rest V and segfault.
    # It must materialize and produce a correct U (U V == I on the solved block).
    fil = _random_grid_filtration()
    dcmp = oin.reduce(fil, oin.ReductionParams(compute_v=True, n_threads=4), False)
    dcmp.compute_u_from_v_1(dim=1)

    v = dcmp.v_as_csc()
    lo, hi = dcmp.dim_first[1], dcmp.dim_last[1] + 1
    u_rows = np.zeros((hi - lo, fil.size()))
    for row in range(lo, hi):
        u_rows[row - lo, np.asarray(dcmp.u_row(row), dtype=np.int64)] = 1
    prod = np.asarray(u_rows @ v) % 2
    identity_rows = np.eye(fil.size())[lo:hi]
    assert np.array_equal(prod, identity_rows)

    # A dimension-only solve is not a complete global U matrix
    with pytest.raises(RuntimeError, match="selected canonical U rows"):
        dcmp.u_as_csr()


def test_compute_u_from_v_guards():
    import pytest

    fil = _random_grid_filtration()

    # fused diagram-only reduce leaves pivots only: no V to solve U from
    dcmp = oin.reduce(fil, oin.ReductionParams(compute_v=False, n_threads=4), False)
    with pytest.raises(RuntimeError, match="V was not computed"):
        dcmp.compute_u_from_v_1()
    with pytest.raises(RuntimeError, match="V was not computed"):
        dcmp.compute_u_from_v()

    # classic reduce without compute_v: same clean error, not a crash
    dcmp2 = oin.Decomposition(fil, dualize=False, n_threads=1)
    dcmp2.reduce(oin.ReductionParams(compute_v=False, n_threads=1))
    with pytest.raises(RuntimeError, match="V was not computed"):
        dcmp2.compute_u_from_v_1()

    # not reduced yet
    dcmp3 = oin.Decomposition(fil, dualize=False, n_threads=1)
    with pytest.raises(RuntimeError, match="not reduced"):
        dcmp3.compute_u_from_v_1()

    # fused RV reduce never stores D, and compute_u_from_v reads it
    dcmp4 = oin.reduce(fil, oin.ReductionParams(compute_v=True, n_threads=4), False)
    with pytest.raises(RuntimeError, match="boundary matrix"):
        dcmp4.compute_u_from_v()


def test_densify_v_for_selinv_fused_keep_working():
    # densify used to read the empty at-rest r_data/v_data on a fused
    # keep-working decomposition and silently return a (0, 0) matrix
    fil = _random_grid_filtration()
    dcmp = oin.reduce(fil, oin.ReductionParams(compute_v=True, n_threads=4), False)
    m = dcmp.densify_v_for_selinv(rows_to_invert={0}, n_threads=1)
    assert m.shape == (fil.size(), fil.size())
    assert m.nnz > 0


def test_fused_reduce_records_col_repr():
    # the fused factories used to skip recording col_repr_ (only member
    # reduce() set it). Heap reductions now use a BitTree residual for the
    # row-form solve, so compare those rows with the default representation.
    fil = _random_grid_filtration()
    params = oin.ReductionParams(compute_v=True, n_threads=4)
    params.advanced.col_repr = oin.ColumnRepr.Heap
    params.advanced.dims_to_restore_elz = [0]

    dcmp = oin.reduce(fil, params, False)
    dcmp.compute_full_u_rows(fil, dim=0)

    # same fused config minus Heap is the oracle
    params2 = oin.ReductionParams(compute_v=True, n_threads=4)
    params2.advanced.dims_to_restore_elz = [0]
    dcmp2 = oin.reduce(fil, params2, False)
    dcmp2.compute_full_u_rows(fil, dim=0)

    lo, hi = dcmp.dim_first[0], dcmp.dim_last[0] + 1
    for row in range(lo, hi):
        assert list(dcmp.u_row(row)) == list(dcmp2.u_row(row))


def test_csc_exports_raise_when_matrix_absent():
    import pytest

    # fused reduce never stores D: d_as_csc must raise, not return an empty matrix
    fil = _make_simplex_filtration()
    dcmp = oin.reduce(fil, oin.ReductionParams(compute_v=True, n_threads=4), False)
    with pytest.raises(RuntimeError, match="does not retain the boundary matrix"):
        dcmp.d_as_csc()

    # classic reduce without compute_u: u_as_csr must raise, not return (0, 0)
    dcmp2 = oin.Decomposition(fil, dualize=False, n_threads=1)
    dcmp2.reduce(oin.ReductionParams(compute_v=True, n_threads=1))
    with pytest.raises(RuntimeError, match="U was not computed"):
        dcmp2.u_as_csr()

    # classic path keeps D: square export works even without compute_v
    dcmp3 = oin.Decomposition(fil, dualize=False, n_threads=1)
    dcmp3.reduce(oin.ReductionParams(n_threads=1))
    d = dcmp3.d_as_csc()
    assert d.shape == (fil.size(), fil.size())


def test_rectangular_csc_export_shape():
    # 3 rows x 2 cols: col 0 has boundary rows {0, 1}, col 1 has {1, 2}. The CSC
    # helper used to reverse rows/cols and callers passed the column count as the
    # row dimension, so this returned (2, 2) instead of (3, 2) (square tests never
    # caught it). R and D live in the boundary row space (n_rows x n_cols); V is
    # the column change-of-basis and stays square (n_cols x n_cols).
    d = [[0, 1], [1, 2]]
    dcmp = oin.Decomposition(d, 3, False, False)  # (matrix, n_rows, dualize, skip_check)
    params = oin.ReductionParams()
    params.compute_v = True
    dcmp.reduce(params)

    assert dcmp.d_as_csc().shape == (3, 2)
    assert dcmp.r_as_csc().shape == (3, 2)
    assert dcmp.v_as_csc().shape == (2, 2)

    # D round-trips to the input boundary (mod 2)
    dense = (np.asarray(dcmp.d_as_csc().todense()) % 2).astype(int)
    assert np.array_equal(dense, np.array([[1, 0], [1, 1], [0, 1]]))
