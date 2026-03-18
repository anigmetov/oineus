import pickle

import numpy as np

import oineus as oin


def _assert_roundtrip(obj):
    obj_back = pickle.loads(pickle.dumps(obj))
    assert obj_back == obj
    assert not (obj_back != obj)
    return obj_back


def _make_simplex_filtration():
    simplices = [
        oin.Simplex([0], 0.0),
        oin.Simplex([1], 0.1),
        oin.Simplex([2], 0.2),
        oin.Simplex([0, 1], 0.3),
        oin.Simplex([0, 2], 0.4),
        oin.Simplex([1, 2], 0.5),
        oin.Simplex([0, 1, 2], 0.6),
    ]
    return oin.Filtration(simplices, negate=False, n_threads=1)


def _make_prod_filtration():
    v0 = oin.Simplex([0], 0.0)
    v1 = oin.Simplex([1], 0.1)
    p01 = oin.ProdSimplex(v0, v1, 0.2)
    p10 = oin.ProdSimplex(v1, v0, 0.3)
    return oin.ProdFiltration([p01, p10], negate=False, n_threads=1)


def test_cells():
    vs = [0, 1, 2]
    _assert_roundtrip(oin.CombinatorialSimplex(vs))
    _assert_roundtrip(oin.Simplex(vs, 1.0))
    dom = _assert_roundtrip(oin.GridDomain_2D(10, 10))
    _assert_roundtrip(oin.CombinatorialCube_2D([1, 2], [0], dom))


def test_filtrations():
    n_pts = 20
    dim = 3
    pts = np.random.normal(0.0, 1.0, n_pts * dim).reshape(n_pts, dim)
    vr_fil = oin.vr_filtration(pts)
    _ = pickle.loads(pickle.dumps(vr_fil.cells()))
    _assert_roundtrip(vr_fil)


def test_common_types():
    edge = oin.VREdge(0)
    edge.x = 1
    edge.y = 2
    _assert_roundtrip(edge)

    params = oin.ReductionParams()
    params.n_threads = 2
    params.chunk_size = 8
    params.compute_v = True
    params.compute_u = True
    _assert_roundtrip(params)

    kicr_params = oin.KICRParams()
    kicr_params.codomain = True
    kicr_params.kernel = True
    kicr_params.image = True
    kicr_params.cokernel = True
    kicr_params.n_threads = 1
    kicr_params.params_im.compute_v = True
    kicr_params.params_cok.compute_u = True
    _assert_roundtrip(kicr_params)


def test_diagram_types():
    point = oin.DiagramPoint(0.1, 0.9)
    point.birth_index = 2
    point.death_index = 5
    _assert_roundtrip(point)

    _assert_roundtrip(oin.IndexDiagramPoint(2, 5))

    fil = _make_simplex_filtration()
    dcmp = oin.Decomposition(fil, dualize=False, n_threads=1)
    params = oin.ReductionParams()
    params.compute_v = True
    dcmp.reduce(params)
    _assert_roundtrip(dcmp.diagram(fil, include_inf_points=True))
    _assert_roundtrip(oin.Diagrams(2))


def test_decomposition_pickle():
    fil = _make_simplex_filtration()

    dcmp = oin.Decomposition(fil, dualize=False, n_threads=1)
    _assert_roundtrip(dcmp)

    params = oin.ReductionParams()
    params.compute_v = True
    params.compute_u = True
    dcmp.reduce(params)
    _assert_roundtrip(dcmp)


def test_topology_optimizer_pickle():
    fil = _make_simplex_filtration()
    opt = oin.TopologyOptimizer(fil)
    opt.reduce_all()
    _assert_roundtrip(opt)

    indvals = opt.combine_loss(opt.singletons([3], [0.35]), oin.ConflictStrategy.Max)
    _assert_roundtrip(indvals)

    prod_fil = _make_prod_filtration()
    prod_opt = oin.TopologyOptimizerProd(prod_fil)
    prod_opt.reduce_all()
    _assert_roundtrip(prod_opt)

    prod_indvals = prod_opt.combine_loss(prod_opt.singletons([0], [0.15]), oin.ConflictStrategy.Max)
    _assert_roundtrip(prod_indvals)


def test_kicr_pickle():
    fil = _make_simplex_filtration()
    params = oin.KICRParams(kernel=True, image=True, cokernel=True, codomain=True, n_threads=1)
    _assert_roundtrip(oin.KerImCokReduced(fil, fil, params))

    prod_fil = _make_prod_filtration()
    prod_params = oin.KICRParams(kernel=True, image=True, cokernel=True, n_threads=1)
    _assert_roundtrip(oin.KerImCokReducedProd(prod_fil, prod_fil, prod_params))
