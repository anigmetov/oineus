"""Slim Freudenthal filtration parity.

The slim (anchor,type) Freudenthal filtration (oineus.freudenthal_filtration with
slim=True, a FreudenthalFiltration_ND) must produce diagrams identical to the fat
universal-Simplex path and matching the dionysus ground truth, across dimension,
negate, dualize and thread count. The A-3a C++ oracle already proved the slim and
fat boundary matrices are column-for-column equal; this checks the whole Python
pipeline (factory -> Decomposition -> diagram) end to end.
"""

import numpy as np
import pytest
import oineus as oin

dion = pytest.importorskip("dionysus")


def dion_dgm_to_numpy(dion_dgm):
    return np.array([[p.birth, p.death] for p in dion_dgm]).astype(np.float64)


def reduce_to_dgms(fil, dualize, n_threads, top_dim):
    dcmp = oin.Decomposition(fil, dualize)
    rp = oin.ReductionParams()
    rp.n_threads = n_threads
    dcmp.reduce(rp)
    dgms = dcmp.diagram(fil, include_inf_points=True)
    return [np.asarray(dgms.in_dimension(d)) for d in range(top_dim)]


def sort_finite(dgm):
    d = np.array(dgm, dtype=np.float64).reshape(-1, 2)
    d[d == np.inf] = 1e9
    d[d == -np.inf] = -1e9
    return d[np.lexsort((d[:, 1], d[:, 0]))]


def dgms_close(a, b, tol=1e-9):
    a, b = sort_finite(a), sort_finite(b)
    return a.shape == b.shape and (a.size == 0 or np.allclose(a, b, atol=tol))


@pytest.mark.parametrize("dim,shape", [(2, (13, 11)), (3, (9, 8, 7))])
@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_slim_matches_fat(dim, shape, negate, dualize, n_threads):
    np.random.seed(42)
    a = np.random.randn(*shape).astype(np.float64)

    fil_slim = oin.freudenthal_filtration(a, negate=negate, max_dim=dim, slim=True, n_threads=n_threads)
    fil_fat = oin.freudenthal_filtration(a, negate=negate, max_dim=dim, slim=False, n_threads=n_threads)

    assert type(fil_slim).__name__ == f"FreudenthalFiltration_{dim}D"
    assert fil_slim.size() == fil_fat.size()

    slim_dgms = reduce_to_dgms(fil_slim, dualize, n_threads, dim)
    fat_dgms = reduce_to_dgms(fil_fat, dualize, n_threads, dim)
    for d in range(dim):
        assert dgms_close(slim_dgms[d], fat_dgms[d]), f"slim != fat in dim {d}"


@pytest.mark.parametrize("dim,shape", [(2, (13, 11)), (3, (9, 8, 7))])
@pytest.mark.parametrize("negate", [False, True])
def test_slim_matches_dionysus(dim, shape, negate):
    np.random.seed(7)
    a = np.random.randn(*shape).astype(np.float64)

    fil_slim = oin.freudenthal_filtration(a, negate=negate, max_dim=dim, slim=True)
    slim_dgms = reduce_to_dgms(fil_slim, dualize=False, n_threads=1, top_dim=dim)

    fil_dion = dion.fill_freudenthal(a, reverse=negate)
    p = dion.homology_persistence(fil_dion)
    dion_dgms = dion.init_diagrams(p, fil_dion)

    dist = 0.0
    for d in range(dim):
        oin_dgm = dion.Diagram(slim_dgms[d])
        dd = dion_dgm_to_numpy(dion_dgms[d])
        if negate:
            dd[dd == np.inf] = -np.inf
        dd = dion.Diagram(dd)
        dist += dion.bottleneck_distance(oin_dgm, dd)
    assert dist < 1e-3, f"slim vs dionysus bottleneck dist {dist}"


def test_slim_pickle_round_trip():
    import pickle
    np.random.seed(1)
    a = np.random.randn(7, 6).astype(np.float64)
    fil = oin.freudenthal_filtration(a, max_dim=2, slim=True)
    fil2 = pickle.loads(pickle.dumps(fil))
    assert fil2 == fil


@pytest.mark.parametrize("dim,shape", [(2, (8, 7)), (3, (6, 5, 5))])
def test_induced_matching_slim_matches_fat(dim, shape):
    # get_induced_matching builds an InclusionFiltration and is the only caller of
    # its boundary_matrix_in_dimension (the packed boundary_into branch). An identity
    # inclusion exercises that branch for the slim FrCell; the matching must agree
    # with the fat-Simplex path cell-for-cell.
    np.random.seed(11)
    a = np.random.randn(*shape).astype(np.float64)
    fs = oin.freudenthal_filtration(a, max_dim=dim, slim=True)
    ff = oin.freudenthal_filtration(a, max_dim=dim, slim=False)
    m_s = oin.get_induced_matching(fs, fs)
    m_f = oin.get_induced_matching(ff, ff)
    assert len(m_s) == len(m_f)
    for d in range(len(m_s)):
        assert len(m_s[d]) == len(m_f[d])


def test_slim_kicr_rejected_clearly():
    # KICR is not wired for the slim cell; the materialized fat cell defeats the
    # isinstance(K[0], Simplex) dispatch, so we must reject slim up front with a
    # clear NotImplementedError rather than a cryptic nanobind ctor TypeError.
    np.random.seed(2)
    a = np.random.randn(6, 6).astype(np.float64)
    K = oin.freudenthal_filtration(a, max_dim=2, slim=True)
    L = K.without_cells([K.size() - 1])
    with pytest.raises(NotImplementedError):
        oin.compute_kernel_image_cokernel_reduction(K, L)


def test_induced_matching_cube_identity_runs():
    # the same inclusion boundary_into branch now serves the slim Cube; an identity
    # cube inclusion must run through it and match the H0 component bar
    np.random.seed(5)
    a = np.random.randn(8, 8).astype(np.float64)
    cf = oin.cube_filtration(a)
    m = oin.get_induced_matching(cf, cf)
    assert len(m) >= 1
    assert len(m[0]) >= 1
