import pickle

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


def _make_prod_segment_inclusion_filtrations():
    # One segment as a product cell complex:
    # ([0]x[2]) -- ([1]x[2]), with L appearing later than K.
    s0 = oin.Simplex([0], 0.0)
    s1 = oin.Simplex([1], 0.0)
    s01 = oin.Simplex([0, 1], 0.0)
    t2 = oin.Simplex([2], 0.0)

    k_v0 = oin.ProdSimplex(s0, t2, 0.00)
    k_v1 = oin.ProdSimplex(s1, t2, 0.10)
    k_e01 = oin.ProdSimplex(s01, t2, 0.30)
    k = oin.Filtration([k_v0, k_v1, k_e01], negate=False, n_threads=1)

    l_v0 = oin.ProdSimplex(s0, t2, 0.20)
    l_v1 = oin.ProdSimplex(s1, t2, 0.25)
    l_e01 = oin.ProdSimplex(s01, t2, 0.40)
    l = oin.Filtration([l_v0, l_v1, l_e01], negate=False, n_threads=1)

    return k, l


def test_kicr_reduced_simplex_api():
    k = _make_simplex_filtration()
    l = _make_simplex_filtration()

    params = oin.KICRParams(kernel=True, image=True, cokernel=True, codomain=True)

    kicr = oin.KerImCokReduced(k, l, params)

    _ = kicr.domain_diagrams()
    _ = kicr.codomain_diagrams()
    _ = kicr.kernel_diagrams()
    _ = kicr.cokernel_diagrams()
    _ = kicr.image_diagrams()

    _ = kicr.old_order_to_new()
    _ = kicr.new_order_to_old()

    kicr.fil_K = kicr.fil_K
    kicr.fil_L = kicr.fil_L

    _ = kicr.decomposition_f
    _ = kicr.decomposition_g
    _ = kicr.decomposition_im
    _ = kicr.decomposition_ker
    _ = kicr.decomposition_cok

    # printable (CLAUDE.md convention): a concise human-readable summary, not the
    # default <object at 0x...> repr
    for s in (repr(kicr), str(kicr)):
        assert s.startswith("KerImCokReduced(")
        assert "max_dim=" in s and "kernel=" in s
        assert "0x" not in s


def test_kicr_reduced_prod_api():
    k, l = _make_prod_segment_inclusion_filtrations()

    params = oin.KICRParams(kernel=True, image=True, cokernel=True)

    kicr = oin.KerImCokReducedProd(k, l, params)

    _ = kicr.kernel_diagrams()
    _ = kicr.cokernel_diagrams()
    _ = kicr.image_diagrams()

    kicr.fil_K = kicr.fil_K
    kicr.fil_L = kicr.fil_L

    _ = kicr.decomposition_f
    _ = kicr.decomposition_g
    _ = kicr.decomposition_im
    _ = kicr.decomposition_ker
    _ = kicr.decomposition_cok

    for s in (repr(kicr), str(kicr)):
        assert s.startswith("KerImCokReduced(")
        assert "0x" not in s


def test_kicr_cube_smoke():
    # KICR is wired for the slim cube cell too (compute_kernel_image_cokernel_reduction
    # dispatches a _CubeFiltration to _KerImCokReduced_Cube_ND). Smoke-test that the path
    # runs end to end, exposes the diagram families + decomposition handles, and pickles.
    a = np.random.default_rng(5).random((6, 6))
    K = oin.cube_filtration(a, max_dim=2)
    L = K.without_cells([K.size() - 1])
    params = oin.KICRParams(kernel=True, image=True, cokernel=True, codomain=True)

    kicr = oin.compute_kernel_image_cokernel_reduction(K, L, params)
    assert type(kicr).__name__ == "_KerImCokReduced_Cube_2D"

    # the inclusion L -> K is the full complex minus one top cube, so the image/domain
    # diagrams are non-trivial
    assert sum(len(kicr.image_diagrams().in_dimension(d)) for d in range(3)) > 0
    for fam in ("domain_diagrams", "codomain_diagrams", "kernel_diagrams",
                "cokernel_diagrams", "image_diagrams"):
        _ = getattr(kicr, fam)()
    _ = kicr.decomposition_f, kicr.decomposition_ker, kicr.decomposition_cok

    assert pickle.loads(pickle.dumps(kicr)) == kicr


if __name__ == "__main__":
    test_kicr_reduced_simplex_api()
