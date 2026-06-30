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


def test_float32_diagrams_and_points_isinstance_marker():
    # Diagrams / DiagramPoint are Real-templated, so a float32 diagram is an _f32 class. The
    # public oineus.Diagrams / oineus.DiagramPoint markers must recognize both backends (like
    # Filtration / ProdFiltration), so isinstance works and the distance facade's multi-dim
    # guard fires for float32 too (it previously degraded to a confusing nanobind TypeError).
    a = np.random.default_rng(0).random((12, 12)).astype(np.float32)
    fr = oin.freudenthal_filtration(a, max_dim=2)
    dcmp = oin.Decomposition(fr, False)
    dcmp.reduce(oin.ReductionParams())
    dgm32 = dcmp.diagram(fil=fr, include_inf_points=True)
    assert _module_tag(dgm32) == "_f32"
    assert isinstance(dgm32, oin.Diagrams)
    pts = dgm32.in_dimension(1, as_numpy=False)
    assert all(isinstance(p, oin.DiagramPoint) for p in pts)
    if len(pts):
        assert _module_tag(pts[0]) == "_f32"

    # float64 diagrams still recognized; construction defaults to the float64 concrete class
    fr64 = oin.freudenthal_filtration(a.astype(np.float64), max_dim=2)
    d64 = oin.Decomposition(fr64, False)
    d64.reduce(oin.ReductionParams())
    dgm64 = d64.diagram(fil=fr64, include_inf_points=True)
    assert isinstance(dgm64, oin.Diagrams)
    assert isinstance(oin.DiagramPoint(0.0, 1.0), oin.DiagramPoint)
    assert _module_tag(oin.DiagramPoint(0.0, 1.0)) == "_oineus"

    # the multi-dim Diagrams guard in the distance facade now fires for float32 with the
    # helpful message; single-dimension float32 diagrams still compute (Hera upcasts to float64)
    with pytest.raises(TypeError, match="in_dimension"):
        oin.bottleneck_distance(dgm32, dgm32)
    assert isinstance(oin.bottleneck_distance(dgm32.in_dimension(1), dgm32.in_dimension(1)), float)


def test_float32_alpha_weighted_routes_to_f32():
    # P2 fix: the weighted / periodic / non-array-exporter alpha fallbacks built a float64
    # filtration even for float32 input. They must route by dtype like the fast array path.
    pytest.importorskip("diode")
    pts = np.random.default_rng(0).random((30, 3)).astype(np.float32)
    w = np.random.default_rng(1).random(30).astype(np.float32)
    try:
        afw32 = oin.alpha_filtration(pts, weights=w)
        afw64 = oin.alpha_filtration(pts.astype(np.float64), weights=w.astype(np.float64))
    except (AttributeError, RuntimeError):
        pytest.skip("diode build lacks weighted alpha shapes")
    assert _module_tag(afw32) == "_f32"
    assert _module_tag(afw64) == "_oineus"


def test_float32_reduce_routes_to_f32():
    # P1: the free oin.reduce (fused build+reduce) is Real-templated; it must accept float32
    # filtrations (it raised TypeError before the facade routed by dtype). The Decomposition class
    # itself is Real-independent (shared), so we route on the filtration, not on the result type.
    a = np.random.default_rng(0).random((10, 10)).astype(np.float32)
    f32 = oin.freudenthal_filtration(a, max_dim=2)
    f64 = oin.freudenthal_filtration(a.astype(np.float64), max_dim=2)
    assert _module_tag(f32) == "_f32"
    d32 = oin.reduce(f32, oin.ReductionParams())
    d64 = oin.reduce(f64)                       # default params path
    for dim in (0, 1):
        g32 = _finite_sorted(np.asarray(d32.diagram(f32).in_dimension(dim)))
        g64 = _finite_sorted(np.asarray(d64.diagram(f64).in_dimension(dim)))
        assert g32.shape == g64.shape
        if g32.size:
            assert np.allclose(g32, g64, atol=1e-5)


def test_float32_valued_cells_isinstance_marker():
    # P2: Simplex / ProdSimplex / Cube_ND are Real-templated valued cells; the public markers must
    # recognize float32 instances (so isinstance and the multiply/cylinder wrappers work) while
    # construction defaults to float64.
    pts = np.random.default_rng(1).random((10, 3)).astype(np.float32)
    fvr32 = oin.vr_filtration(pts, max_dim=2)
    fvr64 = oin.vr_filtration(pts.astype(np.float64), max_dim=2)
    assert _module_tag(fvr32[0]) == "_f32"
    assert isinstance(fvr32[0], oin.Simplex)         # float32 valued simplex recognized
    assert isinstance(fvr64[0], oin.Simplex)         # float64 still recognized
    assert not isinstance(5, oin.Simplex)

    a = np.random.default_rng(2).random((8, 8)).astype(np.float32)
    assert isinstance(oin.cube_filtration(a, max_dim=2)[0], oin.Cube_2D)
    assert isinstance(oin.multiply_filtration(fvr32, oin.CombinatorialSimplex([99]))[0],
                      oin.ProdSimplex)

    # construction defaults to the float64 concrete class
    s = oin.Simplex([0, 1], 0.5)
    assert _module_tag(s) == "_oineus"
    assert isinstance(s, oin.Simplex)


def test_float32_multiply_filtration_accepts_f32_valued_cell():
    # P2 regression: passing a valued cell from a float32 filtration as the auxiliary simplex
    # raised a nanobind TypeError (the wrapper's isinstance check was float64-only, so the value
    # was never stripped to the shared combinatorial cell).
    pts = np.random.default_rng(3).random((10, 3)).astype(np.float32)
    fvr32 = oin.vr_filtration(pts, max_dim=1)
    prod = oin.multiply_filtration(fvr32, fvr32[0])   # fvr32[0] is an _f32 valued Simplex
    assert _module_tag(prod) == "_f32"
    assert isinstance(prod, oin.ProdFiltration)


def test_float32_mapping_cylinder_accepts_f32_valued_vertex():
    # P2 regression (same mechanism as multiply_filtration): a valued vertex from the float32
    # backend must be accepted -- and stripped -- by mapping_cylinder. Domain = 1-skeleton,
    # codomain = full 2-complex on the same points (a genuine subcomplex inclusion).
    f32_mod = oin._dtype.REAL_MODULES[np.dtype("float32")]
    pts = np.random.default_rng(4).random((8, 2)).astype(np.float32)
    dom = oin.vr_filtration(pts, max_dim=1, max_diameter=0.6)
    cod = oin.vr_filtration(pts, max_dim=2, max_diameter=0.6)
    v_dom = f32_mod.Simplex([100], 0.0)               # float32 valued vertices, non-colliding ids
    v_cod = f32_mod.Simplex([200], 0.0)
    cyl = oin.mapping_cylinder(dom, cod, v_dom, v_cod)
    assert _module_tag(cyl) == "_f32"
    assert isinstance(cyl, oin.ProdFiltration)


def test_float32_grid_routes_by_dtype():
    # P2: Grid_ND is Real-templated; oin.Grid_2D(float32_array) must build a genuine float32 grid
    # (whose freudenthal_filtration is float32), not silently widen to float64. isinstance spans
    # both backends.
    a = np.random.default_rng(0).random((6, 6))
    g32 = oin.Grid_2D(a.astype(np.float32))
    g64 = oin.Grid_2D(a)
    assert _module_tag(g32) == "_f32"
    assert _module_tag(g64) == "_oineus"
    assert _module_tag(g32.freudenthal_filtration(negate=False)) == "_f32"
    assert isinstance(g32, oin.Grid_2D)
    assert isinstance(g64, oin.Grid_2D)
    assert not isinstance(a, oin.Grid_2D)
    # routing must also work when data is passed by keyword (else float32 widens silently)
    assert _module_tag(oin.Grid_2D(data=a.astype(np.float32))) == "_f32"
    assert _module_tag(oin.Grid_2D(data=a.astype(np.float32), wrap=True)) == "_f32"
    # both grid->filtration methods (freudenthal AND cube) must stay float32 AND carry the same
    # values as the float64 grid (to float32 precision) -- routing to _f32 is not enough if the
    # values were silently widened/narrowed wrong.
    for meth in ("freudenthal_filtration", "cube_filtration"):
        f32 = getattr(g32, meth)(negate=False)
        f64 = getattr(g64, meth)(negate=False)
        assert _module_tag(f32) == "_f32"
        v32 = np.sort([f32.cell_value_by_sorted_id(i) for i in range(f32.size())])
        v64 = np.sort([f64.cell_value_by_sorted_id(i) for i in range(f64.size())])
        assert v32.shape == v64.shape
        np.testing.assert_allclose(v32, v64, atol=1e-6)


def test_float32_kicr_ctor_routes_and_isinstance_marker():
    # P3: oin.KerImCokReduced direct ctor must route by the filtration dtype (it raised TypeError
    # on a float32 K before), and isinstance must recognize float32 / per-cell-type KICR results.
    pts = np.random.default_rng(1).random((12, 3)).astype(np.float32)
    K = oin.vr_filtration(pts, max_dim=2, max_diameter=0.7, packed=False)
    L = oin.vr_filtration(pts, max_dim=1, max_diameter=0.7, packed=False)
    direct = oin.KerImCokReduced(K, L, oin.KICRParams())
    assert _module_tag(direct) == "_f32"
    assert isinstance(direct, oin.KerImCokReduced)
    facade = oin.compute_kernel_image_cokernel_reduction(K, L)
    assert isinstance(facade, oin.KerImCokReduced)
    # generic marker: also recognizes the internal per-cell-type KICR encodings (both backends)
    f32 = oin._dtype.REAL_MODULES[np.dtype("float32")]
    for mod in (oin._oineus, f32):
        for nm in ("KerImCokReduced", "KerImCokReducedProd",
                   "_KerImCokReduced_Cube_2D", "_KerImCokReduced_Packed_64"):
            assert issubclass(getattr(mod, nm), oin.KerImCokReduced), nm
    # float64 direct ctor still works
    K64 = oin.vr_filtration(pts.astype(np.float64), max_dim=2, max_diameter=0.7, packed=False)
    L64 = oin.vr_filtration(pts.astype(np.float64), max_dim=1, max_diameter=0.7, packed=False)
    assert _module_tag(oin.KerImCokReduced(K64, L64, oin.KICRParams())) == "_oineus"


def test_float32_indices_values_generic_marker():
    # P3: IndicesValues is Real-templated AND per-cell-type. A default (slim/packed) optimizer over
    # a float32 filtration returns an IndicesValuesFreudenthal_*/Packed_* (not the bare name); the
    # generic oineus.IndicesValues marker must still recognize it.
    a = np.random.default_rng(3).random((8, 8)).astype(np.float32)
    opt = oin.TopologyOptimizer(oin.freudenthal_filtration(a, max_dim=1))   # slim default
    iv = opt.simplify(0.1, oin.DenoiseStrategy.BirthBirth, 1)
    assert _module_tag(iv) == "_f32"
    assert type(iv).__name__ != "IndicesValues"        # a per-cell-type concrete return type
    assert isinstance(iv, oin.IndicesValues)
    # generic coverage across both backends and all encodings
    f32 = oin._dtype.REAL_MODULES[np.dtype("float32")]
    for mod in (oin._oineus, f32):
        for nm in ("IndicesValues", "IndicesValuesProd", "IndicesValuesCube_2D",
                   "IndicesValuesFreudenthal_2D", "IndicesValuesPacked_64"):
            assert issubclass(getattr(mod, nm), oin.IndicesValues), nm


def test_float32_concrete_optimizer_ctors_route_by_dtype():
    # P3: the concrete per-cell-type optimizer names route construction by the filtration dtype
    # (they rejected a float32 filtration before) and span both backends for isinstance. The
    # generic oineus.TopologyOptimizer facade is the recommended path; these direct names are kept
    # for API consistency.
    a = np.random.default_rng(4).random((6, 6)).astype(np.float32)
    optc = oin.TopologyOptimizerCube_2D(oin.cube_filtration(a, max_dim=2))
    assert _module_tag(optc) == "_f32"
    assert isinstance(optc, oin.TopologyOptimizerCube_2D)
    prod = oin.multiply_filtration(
        oin.vr_filtration(np.random.default_rng(5).random((8, 3)).astype(np.float32), max_dim=1),
        oin.CombinatorialSimplex([999]))
    optp = oin.TopologyOptimizerProd(prod)
    assert _module_tag(optp) == "_f32"
    assert isinstance(optp, oin.TopologyOptimizerProd)
    # float64 still routes correctly
    assert _module_tag(oin.TopologyOptimizerCube_2D(
        oin.cube_filtration(a.astype(np.float64), max_dim=2))) == "_oineus"
