import numpy as np
import pytest
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


def test_diagram_point_api():
    p = oin.DiagramPoint(0.0, 1.0)
    p.birth = 0.2
    p.death = 0.9
    p.birth_index = 0
    p.death_index = 1

    _ = p.persistence
    _ = p.index_persistence
    _ = p.is_diagonal()
    _ = p.is_inf()
    _ = p[0]
    _ = hash(p)
    _ = (p == p)
    _ = repr(p)


def test_index_diagram_point_api():
    p = oin.IndexDiagramPoint(0, 2)
    p.birth = 1
    p.death = 3

    _ = p.persistence
    _ = p[0]
    _ = hash(p)
    _ = (p == p)
    _ = repr(p)


def test_diagrams_api():
    fil = _make_simplex_filtration()
    dcmp = oin.Decomposition(fil, dualize=False, n_threads=1)
    params = oin.ReductionParams()
    params.compute_v = True
    dcmp.reduce(params)

    dgm = dcmp.diagram(fil, include_inf_points=True)

    _ = dgm.in_dimension(0, as_numpy=False)
    _ = dgm.in_dimension(0, as_numpy=True)
    _ = dgm.index_diagram_in_dimension(0, as_numpy=False)
    _ = dgm.index_diagram_in_dimension(0, as_numpy=True)
    _ = dgm[0]
    assert len(dgm) > 0

    n_dims = len(dgm)
    with pytest.raises(IndexError):
        _ = dgm[n_dims]

    empty = oin.Diagrams(1)
    _ = empty.in_dimension(0, as_numpy=False)
    _ = empty.index_diagram_in_dimension(0, as_numpy=False)
    _ = empty[0]


def test_diagrams_iteration_stops():
    fil = _make_simplex_filtration()
    dcmp = oin.Decomposition(fil, dualize=False, n_threads=1)
    params = oin.ReductionParams()
    dcmp.reduce(params)

    dgms = dcmp.diagram(fil, include_inf_points=True)
    n_dims = len(dgms)
    it = iter(dgms)

    for _ in range(n_dims):
        arr = next(it)
        assert arr.ndim == 2
        assert arr.shape[1] == 2

    with pytest.raises(StopIteration):
        next(it)


def test_diagrams_pad_and_trim():
    d = oin.Diagrams(1)
    assert len(d) == 2

    d.pad_to_dim(4)
    assert len(d) == 5
    assert d[4].shape == (0, 2)

    # no-op grow
    d.pad_to_dim(2)
    assert len(d) == 5

    d.trim_to_dim(2)
    assert len(d) == 3
    assert d[2].shape == (0, 2)
    with pytest.raises(IndexError):
        _ = d[3]

    # no-op trim
    d.trim_to_dim(5)
    assert len(d) == 3


def test_bottleneck_distance_accepts_list_and_numpy():
    dgm_1_list = [oin.DiagramPoint(0.0, 3.0)]
    dgm_2_list = [oin.DiagramPoint(1.0, 5.0)]
    dgm_1_np = np.array([[0.0, 3.0]], dtype=np.float64)
    dgm_2_np = np.array([[1.0, 5.0]], dtype=np.float64)

    dist_list = oin.bottleneck_distance(dgm_1_list, dgm_2_list, delta=0.0)
    dist_np = oin.bottleneck_distance(dgm_1_np, dgm_2_np, delta=0.0)

    assert dist_list == pytest.approx(2.0, abs=1e-12)
    assert dist_np == pytest.approx(2.0, abs=1e-12)
    assert dist_list == pytest.approx(dist_np, abs=1e-12)


def test_wasserstein_distance_accepts_list_and_numpy():
    dgm_1_list = [oin.DiagramPoint(0.0, 3.0)]
    dgm_2_list = [oin.DiagramPoint(1.0, 5.0)]
    dgm_1_np = np.array([[0.0, 3.0]], dtype=np.float64)
    dgm_2_np = np.array([[1.0, 5.0]], dtype=np.float64)

    ws_inf_list = oin.wasserstein_distance(dgm_1_list, dgm_2_list, q=2.0, delta=0.0, internal_p=np.inf)
    ws_inf_np = oin.wasserstein_distance(dgm_1_np, dgm_2_np, q=2.0, delta=0.0, internal_p=np.inf)
    ws_l1_np = oin.wasserstein_distance(dgm_1_np, dgm_2_np, q=2.0, delta=0.0, internal_p=1.0)
    ws_l1_alias = oin.wasserstein_distance(dgm_1_np, dgm_2_np, wasserstein_q=2.0, delta=0.0, internal_p=1.0)

    assert ws_inf_list == pytest.approx(ws_inf_np, abs=1e-12)
    assert ws_inf_list >= 0.0
    assert ws_l1_np == pytest.approx(3.0, abs=1e-12)
    assert ws_l1_alias == pytest.approx(ws_l1_np, abs=1e-12)


def test_distance_array_shape_validation():
    good = np.array([[0.0, 1.0]], dtype=np.float64)
    bad = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError):
        _ = oin.bottleneck_distance(bad, good)

    with pytest.raises(ValueError):
        _ = oin.wasserstein_distance(good, bad)


def test_distances_match_dionysus_when_available():
    dion = pytest.importorskip("dionysus")
    if not hasattr(dion, "Diagram") or not hasattr(dion, "bottleneck_distance") or not hasattr(dion, "wasserstein_distance"):
        pytest.skip("installed dionysus does not expose Diagram-distance API")

    dgm_1_np = np.array([[0.0, 3.0], [1.0, 2.5]], dtype=np.float64)
    dgm_2_np = np.array([[1.0, 5.0], [1.5, 3.0]], dtype=np.float64)

    dgm_1_dion = dion.Diagram(dgm_1_np)
    dgm_2_dion = dion.Diagram(dgm_2_np)

    ws_dion = dion.wasserstein_distance(dgm_1_dion, dgm_2_dion, q=2, delta=0.0, internal_p=-1.0)
    ws_oin = oin.wasserstein_distance(dgm_1_np, dgm_2_np, q=2.0, delta=0.0, internal_p=np.inf)
    bt_dion = dion.bottleneck_distance(dgm_1_dion, dgm_2_dion, delta=0.0)
    bt_oin = oin.bottleneck_distance(dgm_1_np, dgm_2_np, delta=0.0)

    assert ws_oin == pytest.approx(ws_dion, rel=1e-12, abs=1e-12)
    assert bt_oin == pytest.approx(bt_dion, rel=1e-12, abs=1e-12)
