import pickle

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


def test_filtration_dispatches_on_cell_type():
    # oin.Filtration(cells) is a facade that dispatches on the fat cell type: a list of
    # Simplex builds the simplicial filtration, a list of Cube_ND builds the cubical one,
    # ProdSimplex the product one. isinstance(x, oin.Filtration) is true for any filtration
    # the library produces (incl. factory-built ones whose concrete type is internal).
    import numpy as np

    fs = oin.Filtration([oin.Simplex([0], 0.0), oin.Simplex([1], 0.0), oin.Simplex([0, 1], 1.0)])
    assert type(fs).__name__ == "Filtration"
    assert isinstance(fs, oin.Filtration)

    dom = oin.GridDomain_2D(2, 2)
    cubes = [oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[], domain=dom, value=0.0),
             oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[0], domain=dom, value=0.5)]
    fc = oin.Filtration(cubes)
    assert type(fc).__name__ == "CubeFiltration_2D"
    assert isinstance(fc, oin.Filtration)

    fp = oin.Filtration([oin.ProdSimplex(oin.Simplex([0], 0.0), oin.Simplex([1], 0.1), 0.2)])
    assert type(fp).__name__ == "ProdFiltration"
    assert isinstance(fp, oin.Filtration)

    # factory-built filtrations are recognized as Filtrations even though their concrete
    # C++ type is an internal (possibly slim/packed) detail
    vr = oin.vr_filtration(np.ascontiguousarray(np.random.default_rng(0).random((8, 3))),
                           max_dim=2, max_diameter=1.0)
    assert isinstance(vr, oin.Filtration)

    # the (vertices, value)-tuple constructor of the universal simplicial filtration still works
    ft = oin.Filtration([([0], 0.0), ([1], 0.0), ([0, 1], 1.0)])
    assert type(ft).__name__ == "Filtration" and ft.size() == 3

    # empty -> the universal (fat Simplex) filtration; a non-cell list -> clear error
    assert oin.Filtration([]).size() == 0
    try:
        oin.Filtration([42])
        assert False, "expected TypeError"
    except TypeError:
        pass


def _make_prod_filtration():
    v0 = oin.Simplex([0], 0.0)
    v1 = oin.Simplex([1], 0.1)
    p01 = oin.ProdSimplex(v0, v1, 0.2)
    p10 = oin.ProdSimplex(v1, v0, 0.3)
    return oin.ProdFiltration([p01, p10], negate=False, n_threads=1)


def _make_cube_filtration_1d():
    dom = oin.GridDomain_1D(2)
    v0 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[], domain=dom, value=0.0)
    e01 = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom, value=0.5)
    return oin.CubeFiltration_1D([v0, e01], negate=False, n_threads=1)


def _make_cube_filtration_2d():
    dom = oin.GridDomain_2D(2, 2)
    v00 = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[], domain=dom, value=0.0)
    e00 = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[0], domain=dom, value=0.2)
    return oin.CubeFiltration_2D([v00, e00], negate=False, n_threads=1)


def _make_cube_filtration_3d():
    dom = oin.GridDomain_3D(2, 2, 2)
    v000 = oin.Cube_3D(anchor_vertex=[0, 0, 0], spanning_dims=[], domain=dom, value=0.0)
    e000 = oin.Cube_3D(anchor_vertex=[0, 0, 0], spanning_dims=[0], domain=dom, value=0.2)
    return oin.CubeFiltration_3D([v000, e000], negate=False, n_threads=1)


def test_filtration_api():
    fil = _make_simplex_filtration()

    assert len(fil) == fil.size()
    _ = list(iter(fil))
    _ = fil[0]
    _ = fil[-1]

    _ = fil.negate
    _ = fil.max_dim
    _ = fil.cells()
    _ = fil.simplices()
    _ = fil.size()
    _ = fil.size_in_dimension(0)
    _ = fil.n_vertices()

    fil.reset_ids_to_sorted_ids()
    _ = fil.simplex_value_by_sorted_id(0)
    _ = fil.id_by_sorted_id(0)
    id0 = fil.id_by_sorted_id(0)
    _ = fil.sorted_id_by_id(id0)
    _ = fil.cell(0)
    _ = fil.simplex(0)

    _ = fil.dim_first
    _ = fil.dim_last
    _ = fil.sorting_permutation()
    _ = fil.inv_sorting_permutation()

    _ = fil.value_by_uid(fil[0].uid)
    _ = fil.sorted_id_by_uid(fil[0].uid)
    _ = fil.cell_by_uid(fil[0].uid)

    _ = fil.boundary_matrix(n_threads=1)
    _ = fil.boundary_matrix_in_dimension(0, n_threads=1)
    _ = fil.coboundary_matrix(n_threads=1)
    # _ = fil.boundary_matrix_rel(set())

    fil.set_values([float(i) for i in range(fil.size())], n_threads=1)
    fil.set_values([float(i) for i in range(fil.size())])

    _ = fil.subfiltration(lambda s: s.dim == 0)

    assert fil == fil
    _ = fil.get_vertices()
    _ = fil.get_edges()
    _ = fil.get_triangles()
    _ = fil.get_tetrahedra()
    _ = fil.get_simplices_as_arr(0)
    _ = repr(fil)

    fil_back = pickle.loads(pickle.dumps(fil))
    assert fil_back == fil


def test_prod_filtration_api():
    fil = _make_prod_filtration()

    assert len(fil) == fil.size()
    _ = list(iter(fil))
    _ = fil[0]

    _ = fil.negate
    _ = fil.max_dim
    _ = fil.cells()
    _ = fil.size()
    _ = fil.dim_first
    _ = fil.dim_last
    _ = fil.size_in_dimension(0)

    fil.reset_ids_to_sorted_ids()
    _ = fil.cell_value_by_sorted_id(0)
    _ = fil.get_id_by_sorted_id(0)
    _ = fil.get_sorted_id_by_id(fil[0].id)
    _ = fil.get_cell(0)

    _ = fil.get_sorting_permutation()
    _ = fil.get_inv_sorting_permutation()

    # unprefixed aliases matching every other filtration binding; must agree with the
    # get_-prefixed forms (kept for backward compatibility)
    assert fil.id_by_sorted_id(0) == fil.get_id_by_sorted_id(0)
    assert fil.sorted_id_by_id(fil[0].id) == fil.get_sorted_id_by_id(fil[0].id)
    assert fil.cell(0).uid == fil.get_cell(0).uid
    assert list(fil.sorting_permutation()) == list(fil.get_sorting_permutation())
    assert list(fil.inv_sorting_permutation()) == list(fil.get_inv_sorting_permutation())

    _ = fil.boundary_matrix(n_threads=1)
    _ = fil.boundary_matrix_in_dimension(0, n_threads=1)
    _ = fil.coboundary_matrix(n_threads=1)

    fil.set_values([0.2, 0.3])

    assert fil == fil
    _ = repr(fil)

    fil_back = pickle.loads(pickle.dumps(fil))
    assert fil_back == fil


def test_cube_filtration_1d_api():
    fil = _make_cube_filtration_1d()

    assert len(fil) == fil.size()
    _ = list(iter(fil))
    _ = fil[0]
    _ = fil[-1]

    _ = fil.negate
    _ = fil.max_dim
    _ = fil.cells()
    _ = fil.cubes()
    _ = fil.size()
    _ = fil.size_in_dimension(0)
    _ = fil.n_vertices()
    #
    # fil.reset_ids_to_sorted_ids()
    # _ = fil.cube_value_by_sorted_id(0)
    # _ = fil.id_by_sorted_id(0)
    # id0 = fil.id_by_sorted_id(0)
    # _ = fil.sorted_id_by_id(id0)
    # _ = fil.cell(0)
    # _ = fil.cube(0)

    # _ = fil.dim_first
    # _ = fil.dim_last
    # _ = fil.sorting_permutation()
    # _ = fil.inv_sorting_permutation()
    #
    # _ = fil.value_by_uid(fil[0].uid)
    # _ = fil.sorted_id_by_uid(fil[0].uid)
    # _ = fil.cell_by_uid(fil[0].uid)
    #
    # _ = fil.boundary_matrix(n_threads=1)
    # _ = fil.boundary_matrix_in_dimension(0, n_threads=1)
    # _ = fil.coboundary_matrix(n_threads=1)
    # _ = fil.boundary_matrix_rel(set())
    #
    # fil.set_values([0.0, 0.5])
    #
    # assert fil == fil
    # _ = repr(fil)
    #
    # fil_back = pickle.loads(pickle.dumps(fil))
    # assert fil_back == fil


# def test_cube_filtration_2d_api():
#     fil = _make_cube_filtration_2d()
#
#     assert len(fil) == fil.size()
#     _ = list(iter(fil))
#     _ = fil[0]
#
#     _ = fil.negate
#     _ = fil.max_dim
#     _ = fil.cells()
#     _ = fil.cubes()
#     _ = fil.size()
#     _ = fil.size_in_dimension(0)
#     _ = fil.n_vertices()
#
#     fil.reset_ids_to_sorted_ids()
#     _ = fil.cube_value_by_sorted_id(0)
#     _ = fil.id_by_sorted_id(0)
#     id0 = fil.id_by_sorted_id(0)
#     _ = fil.sorted_id_by_id(id0)
#     _ = fil.cell(0)
#     _ = fil.cube(0)
#
#     _ = fil.dim_first
#     _ = fil.dim_last
#     _ = fil.sorting_permutation()
#     _ = fil.inv_sorting_permutation()
#
#     _ = fil.value_by_uid(fil[0].uid)
#     _ = fil.sorted_id_by_uid(fil[0].uid)
#     _ = fil.cell_by_uid(fil[0].uid)
#
#     _ = fil.boundary_matrix(n_threads=1)
#     _ = fil.boundary_matrix_in_dimension(0, n_threads=1)
#     _ = fil.coboundary_matrix(n_threads=1)
#     _ = fil.boundary_matrix_rel(set())
#
#     fil.set_values([0.0, 0.2])
#
#     assert fil == fil
#     _ = repr(fil)
#
#     fil_back = pickle.loads(pickle.dumps(fil))
#     assert fil_back == fil


# def test_cube_filtration_3d_api():
#     fil = _make_cube_filtration_3d()
#
#     assert len(fil) == fil.size()
#     _ = list(iter(fil))
#     _ = fil[0]
#
#     _ = fil.negate
#     _ = fil.max_dim
#     _ = fil.cells()
#     _ = fil.cubes()
#     _ = fil.size()
#     _ = fil.size_in_dimension(0)
#     _ = fil.n_vertices()
#
#     fil.reset_ids_to_sorted_ids()
#     _ = fil.cube_value_by_sorted_id(0)
#     _ = fil.id_by_sorted_id(0)
#     _ = fil.sorted_id_by_id(fil[0].id)
#     _ = fil.cell(0)
#     _ = fil.cube(0)
#
#     _ = fil.dim_first
#     _ = fil.dim_last
#     _ = fil.sorting_permutation()
#     _ = fil.inv_sorting_permutation()
#
#     _ = fil.value_by_uid(fil[0].uid)
#     _ = fil.sorted_id_by_uid(fil[0].uid)
#     _ = fil.cell_by_uid(fil[0].uid)
#
#     _ = fil.boundary_matrix(n_threads=1)
#     _ = fil.boundary_matrix_in_dimension(0, n_threads=1)
#     _ = fil.coboundary_matrix(n_threads=1)
#     _ = fil.boundary_matrix_rel(set())
#
#     fil.set_values([0.0, 0.2])
#
#     assert fil == fil
#     _ = repr(fil)
#
#     fil_back = pickle.loads(pickle.dumps(fil))
#     assert fil_back == fil

if __name__ == "__main__":
    test_prod_filtration_api()
