"""Module-level free functions over the now-default packed/slim and float32 filtrations.

get_nth_persistence / get_denoise_target / get_permutation(_dtv) / compute_relative_diagrams /
get_induced_matching were historically bound only for the fat float64 Simplex. Since
vr_filtration / freudenthal_filtration / alpha_filtration now default to packed/slim encodings,
those helpers must accept the default-returned filtration types (and float32 ones). They are
folded over every cell type in C++ and routed by dtype through a thin Python wrapper. These
tests guard that the helpers (a) accept the default packed/slim filtration without raising and
(b) produce results identical to the fat encoding.

The keyword-only tests guard that the slim/packed toggles cannot silently re-bind a positional
n_threads (they were inserted mid-signature; making them keyword-only turns a silent
single-threaded run into a loud TypeError).
"""
import numpy as np
import pytest

import oineus as oin

_HAS_F32 = np.dtype("float32") in oin._dtype.REAL_MODULES


def _reduce(fil):
    d = oin.Decomposition(fil, False)
    d.reduce(oin.ReductionParams())
    return d


def test_get_nth_persistence_packed_matches_fat():
    pts = np.random.default_rng(0).random((25, 3))
    vr_packed = oin.vr_filtration(pts, max_dim=2)               # packed (default)
    vr_fat = oin.vr_filtration(pts, max_dim=2, packed=False)    # fat
    assert "Packed" in type(vr_packed).__name__
    dp, df = _reduce(vr_packed), _reduce(vr_fat)
    for dim in (0, 1):
        a = oin.get_nth_persistence(vr_packed, dp, dim, 1)
        b = oin.get_nth_persistence(vr_fat, df, dim, 1)
        assert a == pytest.approx(b)


def test_compute_relative_diagrams_packed_matches_fat():
    # K = full 2-complex; L = 1-skeleton (a genuine subcomplex sharing uids and values)
    pts = np.random.default_rng(11).random((22, 3))
    out = {}
    for enc, kw in (("packed", {}), ("fat", {"packed": False})):
        K = oin.vr_filtration(pts, max_dim=2, max_diameter=0.7, **kw)
        L = oin.vr_filtration(pts, max_dim=1, max_diameter=0.7, **kw)
        rel = oin.compute_relative_diagrams(K, L)
        out[enc] = {d: np.asarray(rel.in_dimension(d)) for d in (1, 2)}
    for d in (1, 2):
        a, b = out["packed"][d], out["fat"][d]
        a = a[np.lexsort(a.T)] if a.size else a
        b = b[np.lexsort(b.T)] if b.size else b
        assert a.shape == b.shape
        if a.size:
            assert np.allclose(a, b, atol=1e-9)
    # the relative of a 2-complex by its 1-skeleton has a non-trivial dim-2 diagram
    assert out["packed"][2].shape[0] > 0


def test_get_induced_matching_packed_default():
    pts = np.random.default_rng(3).random((20, 3))
    f = oin.vr_filtration(pts, max_dim=2, max_diameter=0.8)
    assert "Packed" in type(f).__name__
    m = oin.get_induced_matching(f, f)            # dim omitted -> all dims (C++ SIZE_MAX default)
    assert isinstance(m, dict)
    m1 = oin.get_induced_matching(f, f, dim=1)    # explicit non-negative dim
    assert isinstance(m1, dict)


@pytest.mark.skipif(not _HAS_F32, reason="extension built without the float32 backend")
def test_free_helpers_accept_float32():
    a = np.random.default_rng(2).random((10, 10)).astype(np.float32)
    fr = oin.freudenthal_filtration(a, max_dim=2)
    assert type(fr).__module__.endswith("._f32")
    d = _reduce(fr)
    assert isinstance(oin.get_nth_persistence(fr, d, 1, 1), float)
    oin.compute_relative_diagrams(fr, fr)         # must not raise
    assert isinstance(oin.get_induced_matching(fr, fr), dict)
    t = oin.get_denoise_target(1, fr, d, 0.1, oin.DenoiseStrategy.BirthBirth)
    assert isinstance(t, dict)


@pytest.mark.parametrize("call", [
    lambda pts: oin.vr_filtration(pts, False, 2, -1.0, False, 4),   # 6th positional was n_threads
])
def test_vr_filtration_packed_is_keyword_only(call):
    pts = np.random.default_rng(4).random((15, 3))
    with pytest.raises(TypeError):
        call(pts)


def test_freudenthal_filtration_slim_is_keyword_only():
    a = np.random.default_rng(5).random((8, 8))
    with pytest.raises(TypeError):
        oin.freudenthal_filtration(a, False, False, 2, False, 4)   # 6th positional was n_threads
    # keyword form still works and honors the toggle
    f = oin.freudenthal_filtration(a, max_dim=2, slim=False, n_threads=2)
    assert type(f).__name__ == "_Filtration"


def test_prod_filtration_isinstance_marker():
    # oineus.ProdFiltration is an isinstance-only marker for product-cell filtrations,
    # mirroring oineus.Filtration. It must recognize every way a product filtration is built
    # (hand-built, multiply_filtration, mapping_cylinder), be distinct from simplicial
    # filtrations, and refuse construction.
    prod = oin.Filtration([oin.ProdSimplex([0], [0], 0.0),
                           oin.ProdSimplex([1], [0], 0.0),
                           oin.ProdSimplex([0, 1], [0], 1.0)], False, 1)
    assert isinstance(prod, oin.ProdFiltration)
    assert isinstance(prod, oin.Filtration)          # also a filtration

    simplicial = oin.Filtration([oin.Simplex([0], 0.0), oin.Simplex([1], 0.0),
                                 oin.Simplex([0, 1], 1.0)], False, 1)
    assert not isinstance(simplicial, oin.ProdFiltration)
    assert isinstance(simplicial, oin.Filtration)

    mult = oin.multiply_filtration(simplicial, oin.CombinatorialSimplex([99]))
    assert isinstance(mult, oin.ProdFiltration)

    dom = simplicial
    cod = oin.Filtration([oin.Simplex([0], 0.0), oin.Simplex([1], 0.0), oin.Simplex([2], 0.0),
                          oin.Simplex([0, 1], 1.0), oin.Simplex([0, 2], 1.0),
                          oin.Simplex([1, 2], 1.0), oin.Simplex([0, 1, 2], 2.0)], False, 1)
    cyl = oin.mapping_cylinder(dom, cod, oin.CombinatorialSimplex([100]),
                               oin.CombinatorialSimplex([200]))
    assert isinstance(cyl, oin.ProdFiltration)

    # marker only: not a constructor
    with pytest.raises(TypeError):
        oin.ProdFiltration([oin.ProdSimplex([0], [0], 0.0)])


@pytest.mark.skipif(not _HAS_F32, reason="extension built without the float32 backend")
def test_prod_filtration_marker_float32():
    # hand-built ProdSimplex cells are always float64, so exercise the _f32 branch of the
    # marker via a product filtration derived from a float32 input (multiply_filtration).
    a = np.random.default_rng(7).random((6, 6)).astype(np.float32)
    K = oin.freudenthal_filtration(a, max_dim=1)
    cyl32 = oin.multiply_filtration(K, oin.CombinatorialSimplex([999]))
    assert type(cyl32).__module__.endswith("._f32")
    assert isinstance(cyl32, oin.ProdFiltration)
    assert isinstance(cyl32, oin.Filtration)


def test_compute_relative_diagrams_slim_freudenthal_matches_fat():
    # The B1 C++ change added a HasPackedBoundary boundary_into dispatch to
    # boundary_matrix_in_dimension_rel so compute_relative_diagrams works on slim/packed/cube
    # cells. The committed oracle test only covers packed VR; this pins the SLIM Freudenthal
    # path against the fat-Simplex oracle (slim uses boundary_into, fat the uid-returning
    # boundary -- the two must agree). K = full 2-complex, L = 1-skeleton on the same grid, a
    # genuine subcomplex sharing uids and values.
    a = np.random.default_rng(21).random((7, 8))
    out = {}
    for enc, slim in (("slim", True), ("fat", False)):
        K = oin.freudenthal_filtration(a, max_dim=2, slim=slim)
        L = oin.freudenthal_filtration(a, max_dim=1, slim=slim)
        rel = oin.compute_relative_diagrams(K, L)
        out[enc] = {d: np.asarray(rel.in_dimension(d)) for d in (1, 2)}
    assert "Freudenthal" in type(oin.freudenthal_filtration(a, max_dim=2, slim=True)).__name__
    for d in (1, 2):
        s, f = out["slim"][d], out["fat"][d]
        s = s[np.lexsort(s.T)] if s.size else s
        f = f[np.lexsort(f.T)] if f.size else f
        assert s.shape == f.shape
        if s.size:
            assert np.allclose(s, f, atol=1e-9)
    # H_2(K, L) of a 2-complex relative to its 1-skeleton is non-trivial: actually exercise
    # the slim boundary_into rel dispatch, not an empty diagram
    assert out["slim"][2].shape[0] > 0


def test_compute_relative_diagrams_cube_dispatch_nonempty():
    # Cube cells also go through the new boundary_into rel dispatch. There is no "fat cube"
    # oracle (cubes always carry a packed boundary), so this pins the cube path by building a
    # genuine subcomplex (drop the top cells) and asserting the relative diagram is computed
    # without raising and is non-trivial -- i.e. the dispatch actually ran.
    a = np.random.default_rng(22).random((7, 8))
    K = oin.cube_filtration(a, max_dim=2)
    L = K.without_cells([i for i in range(K.size()) if K.cell(i).dim == 2])
    rel = oin.compute_relative_diagrams(K, L)
    assert np.asarray(rel.in_dimension(2)).shape[0] > 0


def test_get_permutation_slim_matches_fat():
    # get_permutation / get_permutation_dtv (warm-start permutations) were folded over every
    # cell type in B1 and routed by dtype in the facade, but had NO test on any encoding. A
    # per-type binding regression or a facade-routing typo would be silent. The permutation is
    # a function of the (shared) complex and the targets, so slim and fat must agree exactly.
    a = np.random.default_rng(23).random((6, 7))
    fr_slim = oin.freudenthal_filtration(a, max_dim=2, slim=True)
    fr_fat = oin.freudenthal_filtration(a, max_dim=2, slim=False)
    target_matching = [(0, 5.0)]   # TargetMatching: list of (sorted_id, new_value)
    ps = oin.get_permutation(target_matching, fr_slim)
    pf = oin.get_permutation(target_matching, fr_fat)
    assert isinstance(ps, dict)
    assert ps == pf

    # get_permutation_dtv chains a denoise target (DiagramToValues) -> permutation; must accept
    # the slim filtration (the folded overload + the module_of_oineus_obj routing)
    d = _reduce(fr_slim)
    dtv = oin.get_denoise_target(1, fr_slim, d, 0.1, oin.DenoiseStrategy.BirthBirth)
    perm = oin.get_permutation_dtv(dtv, fr_slim)
    assert isinstance(perm, dict)


@pytest.mark.skipif(not _HAS_F32, reason="extension built without the float32 backend")
def test_get_permutation_float32_routes():
    # the facade wrapper routes get_permutation to the _f32 overload via module_of_oineus_obj
    a = np.random.default_rng(24).random((6, 7)).astype(np.float32)
    fr = oin.freudenthal_filtration(a, max_dim=2)
    assert type(fr).__module__.endswith("._f32")
    perm = oin.get_permutation([(0, 5.0)], fr)
    assert isinstance(perm, dict)


def test_alpha_filtration_packed_is_keyword_only():
    # B4: the packed= toggle in alpha_filtration is keyword-only, so a positional argument at
    # its old slot (now n_threads-adjacent) raises loudly instead of silently rebinding. The
    # signature rejects the extra positional before any diode work, so no diode dep is needed.
    pts = np.random.default_rng(25).random((10, 3))
    with pytest.raises(TypeError):
        # 8th positional was `packed` (after points..bbox_max)
        oin.alpha_filtration(pts, None, False, False, True, None, None, True)
