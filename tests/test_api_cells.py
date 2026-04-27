import pickle

import numpy as np
import oineus as oin


def test_combinatorial_simplex_api():
    sigma = oin.CombinatorialSimplex([0, 1])
    sigma_with_id = oin.CombinatorialSimplex(7, [0, 1])

    assert list(iter(sigma)) == [0, 1]
    assert sigma[0] == 0

    sigma.id = 3
    _ = sigma.vertices
    _ = sigma.uid
    _ = sigma.dim
    _ = sigma.boundary()
    _ = sigma.join(new_vertex=2)

    assert sigma == sigma
    _ = (sigma != sigma_with_id)
    _ = hash(sigma)
    _ = repr(sigma)

    sigma_back = pickle.loads(pickle.dumps(sigma))
    assert sigma_back == sigma


def test_combinatorial_prod_simplex_api():
    prod = oin.CombinatorialProdSimplex([0], [1])
    prod.id = 11

    _ = prod.factor_1
    _ = prod.factor_2
    _ = prod.uid
    _ = prod.dim
    _ = prod.boundary()

    assert prod == prod
    _ = hash(prod)
    _ = repr(prod)

    prod_back = pickle.loads(pickle.dumps(prod))
    assert prod_back == prod


def test_simplex_api():
    sigma = oin.Simplex([0, 1], 0.25)
    sigma_with_id = oin.Simplex(5, [0, 1], 0.5)

    assert list(iter(sigma)) == [0, 1]
    assert sigma[1] == 1

    sigma.id = 9
    sigma.sorted_id = 1
    _ = sigma.vertices
    _ = sigma.uid
    _ = sigma.value
    _ = sigma.dim
    _ = sigma.boundary()
    _ = sigma.combinatorial_simplex
    _ = sigma.combinatorial_cell
    _ = sigma.join(new_vertex=2, value=0.75)

    assert sigma == sigma
    _ = (sigma != sigma_with_id)
    _ = hash(sigma)
    _ = repr(sigma)

    sigma_back = pickle.loads(pickle.dumps(sigma))
    assert sigma_back == sigma


def test_prod_simplex_api():
    sigma = oin.Simplex([0], 0.1)
    tau = oin.Simplex([1], 0.2)

    prod = oin.ProdSimplex([0], [1], 0.5)
    prod_alt = oin.ProdSimplex([1], [2], 0.6)

    prod.id = 4
    prod.sorted_id = 2
    _ = prod.factor_1
    _ = prod.factor_2
    _ = prod.cell_1
    _ = prod.cell_2
    _ = prod.uid
    _ = prod.value
    _ = prod.dim
    _ = prod.boundary()
    _ = prod.combinatorial_cell

    assert prod == prod
    _ = (prod != prod_alt)
    _ = hash(prod)
    _ = repr(prod)

    prod_back = pickle.loads(pickle.dumps(prod))
    assert prod_back == prod


def test_griddomain_1d_api():
    dom = oin.GridDomain_1D(2)
    _ = dom.shape
    assert dom == dom
    _ = hash(dom)

    dom_back = pickle.loads(pickle.dumps(dom))
    assert dom_back == dom


def test_griddomain_2d_api():
    dom = oin.GridDomain_2D(2, 3)
    _ = dom.shape
    assert dom == dom
    _ = hash(dom)

    dom_back = pickle.loads(pickle.dumps(dom))
    assert dom_back == dom


def test_griddomain_3d_api():
    dom = oin.GridDomain_3D(2, 2, 2)
    _ = dom.shape
    assert dom == dom
    _ = hash(dom)

    dom_back = pickle.loads(pickle.dumps(dom))
    assert dom_back == dom


def test_grid_1d_api():
    data = np.array([0.0, 1.0], dtype=float)
    grid = oin.Grid_1D(data, wrap=False, values_on="vertices")

    _ = grid.data_location
    _ = grid.cube_filtration(max_dim=1, negate=False, n_threads=1)
    _ = grid.cube_filtration_and_critical_indices(max_dim=1, negate=False, n_threads=1)
    _ = grid.freudenthal_filtration(max_dim=1, negate=False, n_threads=1)
    _ = grid.freudenthal_filtration_and_critical_vertices(max_dim=1, negate=False, n_threads=1)


def test_grid_2d_api():
    data = np.zeros((2, 2), dtype=float)
    grid = oin.Grid_2D(data, wrap=False, values_on="vertices")

    _ = grid.data_location
    _ = grid.cube_filtration(max_dim=2, negate=False, n_threads=1)
    _ = grid.cube_filtration_and_critical_indices(max_dim=2, negate=False, n_threads=1)
    _ = grid.freudenthal_filtration(max_dim=2, negate=False, n_threads=1)
    _ = grid.freudenthal_filtration_and_critical_vertices(max_dim=2, negate=False, n_threads=1)


def test_grid_3d_api():
    data = np.zeros((2, 2, 2), dtype=float)
    grid = oin.Grid_3D(data, wrap=False, values_on="vertices")

    _ = grid.data_location
    _ = grid.cube_filtration(max_dim=3, negate=False, n_threads=1)
    _ = grid.cube_filtration_and_critical_indices(max_dim=3, negate=False, n_threads=1)
    _ = grid.freudenthal_filtration(max_dim=3, negate=False, n_threads=1)
    _ = grid.freudenthal_filtration_and_critical_vertices(max_dim=3, negate=False, n_threads=1)


def test_combinatorial_cube_1d_api():
    dom = oin.GridDomain_1D(2)
    cube = oin.CombinatorialCube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom)
    cube_alt = oin.CombinatorialCube_1D(dom, 0)

    _ = cube.dim
    _ = cube.uid
    _ = cube.vertices
    _ = cube.anchor_vertex
    cube.id = 12
    _ = cube.domain
    _ = cube.boundary()
    _ = cube.coboundary()
    _ = cube.top_cofaces()
    _ = repr(cube)
    _ = str(cube)

    assert cube == cube
    _ = (cube != cube_alt)
    _ = hash(cube)

    cube_back = pickle.loads(pickle.dumps(cube))
    assert cube_back == cube


def test_combinatorial_cube_2d_api():
    dom = oin.GridDomain_2D(2, 2)
    cube = oin.CombinatorialCube_2D(anchor_vertex=[0, 0], spanning_dims=[0, 1], domain=dom)
    cube_alt = oin.CombinatorialCube_2D(dom, 0)

    _ = cube.dim
    _ = cube.uid
    _ = cube.vertices
    _ = cube.anchor_vertex
    cube.id = 7
    _ = cube.domain
    _ = cube.boundary()
    _ = cube.coboundary()
    _ = cube.top_cofaces()
    _ = repr(cube)
    _ = str(cube)

    assert cube == cube
    _ = (cube != cube_alt)
    _ = hash(cube)

    cube_back = pickle.loads(pickle.dumps(cube))
    assert cube_back == cube


def test_combinatorial_cube_3d_api():
    dom = oin.GridDomain_3D(2, 2, 2)
    cube = oin.CombinatorialCube_3D(anchor_vertex=[0, 0, 0], spanning_dims=[0, 1, 2], domain=dom)
    cube_alt = oin.CombinatorialCube_3D(dom, 0)

    _ = cube.dim
    _ = cube.uid
    _ = cube.vertices
    _ = cube.anchor_vertex
    cube.id = 5
    _ = cube.domain
    _ = cube.boundary()
    _ = cube.coboundary()
    _ = cube.top_cofaces()
    _ = repr(cube)
    _ = str(cube)

    assert cube == cube
    _ = (cube != cube_alt)
    _ = hash(cube)

    cube_back = pickle.loads(pickle.dumps(cube))
    assert cube_back == cube


def test_cube_1d_api():
    dom = oin.GridDomain_1D(2)
    cube = oin.Cube_1D(anchor_vertex=[0], spanning_dims=[0], domain=dom, value=1.0)
    cube_alt = oin.Cube_1D(dom, 0, 0.5)

    _ = cube.dim
    _ = cube.uid
    cube.value = 2.0
    _ = cube.vertices
    _ = cube.boundary()
    _ = cube.coboundary()
    _ = cube.top_cofaces()
    _ = repr(cube)
    _ = str(cube)

    assert cube == cube
    _ = (cube != cube_alt)
    _ = hash(cube)

    _ = pickle.dumps(cube)
    cube_back = pickle.loads(pickle.dumps(cube))
    assert cube_back == cube


def test_cube_2d_api():
    dom = oin.GridDomain_2D(2, 2)
    cube = oin.Cube_2D(anchor_vertex=[0, 0], spanning_dims=[0, 1], domain=dom, value=1.0)
    cube_alt = oin.Cube_2D(dom, 0, 0.5)

    _ = cube.dim
    _ = cube.uid
    cube.value = 2.5
    _ = cube.vertices
    _ = cube.boundary()
    _ = cube.coboundary()
    _ = cube.top_cofaces()
    _ = repr(cube)
    _ = str(cube)

    assert cube == cube
    _ = (cube != cube_alt)
    _ = hash(cube)

    cube_back = pickle.loads(pickle.dumps(cube))
    assert cube_back == cube


def test_cube_3d_api():
    dom = oin.GridDomain_3D(2, 2, 2)
    cube = oin.Cube_3D(anchor_vertex=[0, 0, 0], spanning_dims=[0, 1, 2], domain=dom, value=1.0)
    cube_alt = oin.Cube_3D(dom, 0, 0.5)

    _ = cube.dim
    _ = cube.uid
    cube.value = 3.0
    _ = cube.vertices
    _ = cube.boundary()
    _ = cube.coboundary()
    _ = cube.top_cofaces()
    _ = repr(cube)
    _ = str(cube)

    assert cube == cube
    _ = (cube != cube_alt)
    _ = hash(cube)

    cube_back = pickle.loads(pickle.dumps(cube))
    assert cube_back == cube
