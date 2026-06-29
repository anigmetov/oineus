"""float32 Real support.

Oineus instantiates its Real-dependent types for both float64 (the default, on the
top module) and float32 (in the _oineus._f32 submodule). The Python facade detects a
float32 numpy/torch array and routes it to the float32 backend, hiding the _f32
submodule. These tests check that:

- a float32 input builds a genuine float32 filtration (its concrete C++ class lives
  in _f32) and produces float32 diagrams;
- the float32 diagrams match the float64 ones to float32 precision;
- the facade dispatches Decomposition / TopologyOptimizer / isinstance for float32;
- a float64 (or default) input still routes to the top module.

Skipped when the extension was built float64-only (no _f32 submodule).
"""

import numpy as np
import pytest
import oineus as oin

pytestmark = pytest.mark.skipif(
    np.dtype("float32") not in oin._dtype.REAL_MODULES,
    reason="extension built without the float32 backend",
)


def _module_tag(obj):
    # last component of the defining module: '_f32' for float32, '_oineus' for float64
    return type(obj).__module__.rsplit(".", 1)[-1]


def _dgms(fil, top_dim):
    dcmp = oin.Decomposition(fil, False)
    dcmp.reduce(oin.ReductionParams())
    dgm = dcmp.diagram(fil=fil, include_inf_points=True)
    return [np.asarray(dgm.in_dimension(d)) for d in range(top_dim)]


def _finite_sorted(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    a = a[np.isfinite(a).all(axis=1)]
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def test_real_dtypes_attr():
    assert "float32" in oin._oineus.real_dtypes
    assert "float64" in oin._oineus.real_dtypes


@pytest.mark.parametrize("max_dim", [1, 2])
def test_vr_float32(max_dim):
    pts = np.random.default_rng(0).random((40, 3))
    f32 = oin.vr_filtration(pts.astype(np.float32), max_dim=max_dim)
    f64 = oin.vr_filtration(pts.astype(np.float64), max_dim=max_dim)
    assert _module_tag(f32) == "_f32"
    assert _module_tag(f64) == "_oineus"

    d32 = _dgms(f32, max_dim + 1)
    d64 = _dgms(f64, max_dim + 1)
    for d in range(max_dim + 1):
        assert d32[d].dtype == np.float32
        a, b = _finite_sorted(d32[d]), _finite_sorted(d64[d])
        assert a.shape == b.shape
        if a.size:
            assert np.allclose(a, b, atol=1e-5)


def test_freudenthal_float32():
    a = np.random.default_rng(1).random((9, 8))
    f32 = oin.freudenthal_filtration(a.astype(np.float32), max_dim=2)
    f64 = oin.freudenthal_filtration(a.astype(np.float64), max_dim=2)
    assert _module_tag(f32) == "_f32"
    assert type(f32).__name__ == "_FreudenthalFiltration_2D"
    assert _module_tag(f64) == "_oineus"

    d32, d64 = _dgms(f32, 2), _dgms(f64, 2)
    for d in range(2):
        assert d32[d].dtype == np.float32
        a32, b64 = _finite_sorted(d32[d]), _finite_sorted(d64[d])
        assert a32.shape == b64.shape
        if a32.size:
            assert np.allclose(a32, b64, atol=1e-5)


def test_cube_float32():
    a = np.random.default_rng(2).random((8, 7))
    f32 = oin.cube_filtration(a.astype(np.float32))
    assert _module_tag(f32) == "_f32"
    assert type(f32).__name__ == "_CubeFiltration_2D"
    d32 = _dgms(f32, 2)
    assert d32[0].dtype == np.float32
    # one connected component -> one essential H0 class
    assert int(np.sum(np.isinf(d32[0][:, 1]))) == 1


def test_facade_dispatch_float32():
    a = np.random.default_rng(3).random((8, 8)).astype(np.float32)
    fil = oin.freudenthal_filtration(a, max_dim=2)
    assert isinstance(fil, oin.Filtration)
    opt = oin.TopologyOptimizer(fil)
    assert _module_tag(opt) == "_f32"
    assert type(opt).__name__ == "TopologyOptimizerFreudenthal_2D"


def test_distance_across_float32_diagrams():
    # float32 diagrams are upcast for the (double-only) Hera distance; the value is
    # the same as computing on the float64 diagrams
    pts = np.random.default_rng(4).random((30, 2))
    f32 = oin.vr_filtration(pts.astype(np.float32), max_dim=1)
    g32 = oin.vr_filtration((pts + 0.01).astype(np.float32), max_dim=1)
    a = _dgms(f32, 2)[1]
    b = _dgms(g32, 2)[1]
    dist = oin.wasserstein_distance(a, b, q=2)
    assert dist >= 0.0


def test_min_and_multiply_filtration_float32():
    # min_filtration / multiply_filtration / mapping_cylinder route the Real-dependent
    # free functions to the float32 backend (they used to hardcode float64)
    pts = np.random.default_rng(5).random((18, 2)).astype(np.float32)
    f1 = oin.vr_filtration(pts, max_dim=1)
    f2 = oin.vr_filtration((pts + 0.05).astype(np.float32), max_dim=1)

    mf = oin.min_filtration(f1, f2)
    assert _module_tag(mf) == "_f32"

    sigma = oin.CombinatorialSimplex([0])
    mult = oin.multiply_filtration(f1, sigma)
    assert _module_tag(mult) == "_f32"

    # float64 inputs still route to the top module
    g1 = oin.vr_filtration(pts.astype(np.float64), max_dim=1)
    assert _module_tag(oin.min_filtration(g1, g1)) == "_oineus"


def test_diff_float32_end_to_end():
    # a float32 torch tensor builds a genuine float32 differentiable filtration; the
    # persistence diagram is float32 and the gradient flows back as float32
    torch = pytest.importorskip("torch")
    import oineus.diff as od

    x = torch.rand((8, 8), dtype=torch.float32, requires_grad=True)
    df = od.cubical.cube_filtration(x)
    assert _module_tag(df.under_fil) == "_f32"
    assert df.values.dtype == torch.float32

    dgms = od.persistence_diagram(df, include_inf_points=False)
    d1 = dgms.in_dimension(1)
    assert d1.dtype == torch.float32
    loss = (d1[:, 1] - d1[:, 0]).pow(2).sum()
    loss.backward()
    assert x.grad.dtype == torch.float32
    assert bool((x.grad.abs() > 0).any())

    # a float64 tensor still routes to the top (float64) module
    y = torch.rand((8, 8), dtype=torch.float64, requires_grad=True)
    df64 = od.cubical.cube_filtration(y)
    assert _module_tag(df64.under_fil) == "_oineus"


def test_float32_alpha_filtration_routes_to_f32():
    # P2a regression: alpha_filtration hardcoded the float64 module, so a float32 point cloud
    # silently produced a float64 alpha filtration. It must now route by dtype like
    # vr_filtration / freudenthal_filtration, and the float32 diagrams must match float64.
    pytest.importorskip("diode")
    if not getattr(oin, "_HAS_DIODE_ARRAYS", False):
        pytest.skip("alpha float32 routing uses the diode array exporters")
    pts = np.random.default_rng(0).random((40, 2))
    af64 = oin.alpha_filtration(pts.astype(np.float64))
    af32 = oin.alpha_filtration(pts.astype(np.float32))
    assert _module_tag(af64) == "_oineus"
    assert _module_tag(af32) == "_f32"
    for dim in (0, 1):
        g64 = _finite_sorted(_dgms(af64, 2)[dim])
        g32 = _finite_sorted(_dgms(af32, 2)[dim])
        assert g64.shape == g32.shape
        if g64.size:
            assert np.allclose(g64, g32, atol=1e-4)


def test_float32_delaunay_combinatorics_routes_to_f32():
    # the differentiable cech/weak-alpha paths build their combinatorics via
    # _delaunay_combinatorics; a float32 point cloud must yield a float32 Delaunay filtration
    # so the recomputed float32 values stay float32 end-to-end.
    pytest.importorskip("diode")
    if not getattr(oin, "_HAS_DIODE_ARRAYS", False):
        pytest.skip("requires the diode array exporters")
    pts = np.random.default_rng(1).random((30, 2))
    f32 = oin._delaunay_combinatorics(pts.astype(np.float32))
    f64 = oin._delaunay_combinatorics(pts.astype(np.float64))
    assert _module_tag(f32) == "_f32"
    assert _module_tag(f64) == "_oineus"
