import numpy as np
import pytest
import oineus as oin
from oineus._dtype import REAL_DTYPE

# Tight tolerance for "two equivalent calls return the same value" checks.
ABS_TOL = 1e-12 if REAL_DTYPE == np.float64 else 1e-5


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
    dgm_1_np = np.array([[0.0, 3.0]], dtype=REAL_DTYPE)
    dgm_2_np = np.array([[1.0, 5.0]], dtype=REAL_DTYPE)

    dist_list = oin.bottleneck_distance(dgm_1_list, dgm_2_list, delta=0.0)
    dist_np = oin.bottleneck_distance(dgm_1_np, dgm_2_np, delta=0.0)

    assert np.isfinite(dist_list)
    assert np.isfinite(dist_np)
    assert dist_list == pytest.approx(dist_np, abs=ABS_TOL)


def test_wasserstein_distance_accepts_list_and_numpy():
    dgm_1_list = [oin.DiagramPoint(0.0, 3.0)]
    dgm_2_list = [oin.DiagramPoint(1.0, 5.0)]
    dgm_1_np = np.array([[0.0, 3.0]], dtype=REAL_DTYPE)
    dgm_2_np = np.array([[1.0, 5.0]], dtype=REAL_DTYPE)

    # delta must be > 0 for Wasserstein (the auction has no exact mode).
    ws_inf_list = oin.wasserstein_distance(dgm_1_list, dgm_2_list, q=2.0, delta=0.01, internal_p=np.inf)
    ws_inf_np = oin.wasserstein_distance(dgm_1_np, dgm_2_np, q=2.0, delta=0.01, internal_p=np.inf)
    ws_l1_list = oin.wasserstein_distance(dgm_1_list, dgm_2_list, q=2.0, delta=0.01, internal_p=1.0)
    ws_l1_np = oin.wasserstein_distance(dgm_1_np, dgm_2_np, q=2.0, delta=0.01, internal_p=1.0)
    ws_l1_alias = oin.wasserstein_distance(dgm_1_np, dgm_2_np, wasserstein_q=2.0, delta=0.01, internal_p=1.0)

    assert ws_inf_list == pytest.approx(ws_inf_np, abs=ABS_TOL)
    assert np.isfinite(ws_inf_list)
    assert ws_l1_list == pytest.approx(ws_l1_np, abs=ABS_TOL)
    assert ws_l1_alias == pytest.approx(ws_l1_np, abs=ABS_TOL)


def test_distance_functions_accept_oineus_diagrams_with_dim():
    fil = _make_simplex_filtration()
    dcmp = oin.Decomposition(fil, dualize=False, n_threads=1)
    params = oin.ReductionParams()
    dcmp.reduce(params)
    dgms = dcmp.diagram(fil, include_inf_points=True)

    dgm_dim_0 = dgms[0]

    bt = oin.bottleneck_distance(dgms.in_dimension(0), dgm_dim_0, delta=0.0)
    ws = oin.wasserstein_distance(dgms.in_dimension(0), dgm_dim_0, q=2.0, delta=1e-3, internal_p=np.inf)

    assert bt == pytest.approx(0.0, abs=ABS_TOL)
    assert ws == pytest.approx(0.0, abs=ABS_TOL)


def test_distance_functions_reject_multidim_oineus_diagrams():
    fil = _make_simplex_filtration()
    dcmp = oin.Decomposition(fil, dualize=False, n_threads=1)
    params = oin.ReductionParams()
    dcmp.reduce(params)
    dgms = dcmp.diagram(fil, include_inf_points=True)

    # Multi-dim Diagrams must be reduced to a single dimension by the caller
    # via dgm.in_dimension(d) before passing to a distance function.
    with pytest.raises(TypeError, match="single-dimension"):
        _ = oin.bottleneck_distance(dgms, dgms)

    with pytest.raises(TypeError, match="single-dimension"):
        _ = oin.wasserstein_distance(dgms, dgms)


def test_distance_array_shape_validation():
    good = np.array([[0.0, 1.0]], dtype=REAL_DTYPE)
    bad = np.array([0.0, 1.0], dtype=REAL_DTYPE)

    with pytest.raises(ValueError):
        _ = oin.bottleneck_distance(bad, good)

    with pytest.raises(ValueError):
        _ = oin.wasserstein_distance(good, bad)


def test_diagram_point_index_persistence():
    # index_persistence was mistakenly bound to DgmPoint::persistence (the value
    # gap) instead of DgmPoint::index_persistence (the sorted-index gap).
    p = oin.DiagramPoint(0.25, 1.0)
    p.birth_index = 2
    p.death_index = 9
    assert p.index_persistence == 7
    assert abs(p.persistence - 0.75) < ABS_TOL


def test_compute_diagrams_vr_max_dim_is_diagram_dimension():
    # 12 points evenly spaced on the unit circle: H_1 has one prominent finite loop
    # killed by 2-simplices across the circle.
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)]).astype(REAL_DTYPE)

    dgms_1 = oin.compute_diagrams_vr(circle, max_dim=1, include_inf_points=True)
    assert len(dgms_1) == 2
    h1 = dgms_1.in_dimension(1)
    assert h1.shape[0] >= 1
    assert np.all(np.isfinite(h1)), "H_1 bars on a closed circle must die finitely via 2-simplices"
    assert float(np.max(h1[:, 1] - h1[:, 0])) > 1.0

    # Default max_dim=-1 on 2D points computes H_0 and H_1 (diagram max_dim = d - 1 = 1)
    dgms_default = oin.compute_diagrams_vr(circle)
    assert len(dgms_default) == 2
    assert np.allclose(dgms_default[1], h1, atol=ABS_TOL)

    # max_dim=0 returns only H_0 (killed by 1-simplices)
    dgms_0 = oin.compute_diagrams_vr(circle, max_dim=0, include_inf_points=True)
    assert len(dgms_0) == 1
    assert np.sum(np.isinf(dgms_0[0][:, 1])) == 1

    # Pairwise distances require explicit max_dim and agree with point-cloud VR
    dists = np.linalg.norm(circle[:, None, :] - circle[None, :, :], axis=-1).astype(REAL_DTYPE)
    with pytest.raises(RuntimeError):
        oin.compute_diagrams_vr(dists, from_pwdists=True)
    dgms_pw = oin.compute_diagrams_vr(dists, from_pwdists=True, max_dim=1, include_inf_points=True)
    assert len(dgms_pw) == 2
    assert np.allclose(dgms_pw[1], h1, atol=ABS_TOL)


def test_pep561_type_stubs_valid_and_complete():
    import ast
    from pathlib import Path
    import oineus.diff
    import oineus.vis

    pkg_dir = Path(oin.__file__).resolve().parent
    assert (pkg_dir / "py.typed").is_file()

    for rel, mod in [
        ("__init__.pyi", oin),
        ("vis/__init__.pyi", oineus.vis),
        ("diff/__init__.pyi", oineus.diff),
        ("diff/cochains.pyi", None),
    ]:
        stub_path = pkg_dir / rel
        assert stub_path.is_file(), f"Missing stub {stub_path}"
        tree = ast.parse(stub_path.read_text(encoding="utf-8"))
        defined = set()
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                defined.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
        if mod is not None:
            missing = set(mod.__all__) - defined
            assert not missing, f"{rel} missing __all__ symbols: {sorted(missing)}"
