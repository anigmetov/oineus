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
    return oin.ProdFiltration([p01, p10], negate=False, n_threads=1)


def _make_cube_filtration_1d():
    dom = oin.GridDomain_1D(2)
    v0 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom, value=0.0)
    v1 = oin.Cube_1D(anchor_vertex=[1], spanning_dims=[], domain=dom, value=0.1)
    e01 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom, value=0.5)
    return oin.CubeFiltration_1D([v0, v1, e01], negate=False, n_threads=1)


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
