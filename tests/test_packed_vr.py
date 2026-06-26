import numpy as np
import pytest

import oineus as oin


def _sorted(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def _reduce(fil, dualize=False, n_threads=1):
    d = oin.Decomposition(fil, dualize)
    rp = oin.ReductionParams()
    rp.n_threads = n_threads
    d.reduce(rp)
    return d


def _dgms_equal(dp, fp, df, ff, max_dim):
    ddp = dp.diagram(fp)
    ddf = df.diagram(ff)
    for d in range(max_dim + 1):
        a = _sorted(ddp.in_dimension(d))
        b = _sorted(ddf.in_dimension(d))
        a[a == np.inf] = 1e9
        b[b == np.inf] = 1e9
        assert a.shape == b.shape, f"dim {d}: {a.shape} vs {b.shape}"
        assert np.allclose(a, b, atol=1e-6), f"dim {d}"


@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("dualize", [False, True])
def test_packed_vr_matches_fat(n_threads, dualize):
    pts = np.ascontiguousarray(np.random.default_rng(0).random((30, 3)))
    R = 1.0
    ff = oin.vr_filtration(pts, max_dim=2, max_diameter=R, n_threads=n_threads, packed=False)
    fp = oin.vr_filtration(pts, max_dim=2, max_diameter=R, n_threads=n_threads, packed=True)
    assert type(fp).__name__ == "PackedSimplexFiltration_64"
    assert fp.size() == ff.size()
    _dgms_equal(_reduce(fp, dualize, n_threads), fp, _reduce(ff, dualize, n_threads), ff, 2)


def test_packed_vr_from_pwdists_matches_fat():
    pts = np.ascontiguousarray(np.random.default_rng(5).random((25, 3)))
    # pairwise Euclidean distances
    diff = pts[:, None, :] - pts[None, :, :]
    pw = np.ascontiguousarray(np.sqrt((diff ** 2).sum(axis=2)))
    R = 1.0
    ff = oin.vr_filtration(pw, from_pwdists=True, max_dim=2, max_diameter=R, packed=False)
    fp = oin.vr_filtration(pw, from_pwdists=True, max_dim=2, max_diameter=R, packed=True)
    assert type(fp).__name__ == "PackedSimplexFiltration_64"
    assert fp.size() == ff.size()
    _dgms_equal(_reduce(fp), fp, _reduce(ff), ff, 2)


def test_packed_vr_critical_edges_aligned():
    pts = np.ascontiguousarray(np.random.default_rng(1).random((25, 3)))
    _, ef = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, with_critical_edges=True, packed=False)
    _, ep = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, with_critical_edges=True, packed=True)
    assert np.array_equal(ef, ep)


def test_packed_vr_matches_dionysus():
    dion = pytest.importorskip("dionysus")
    pts = np.ascontiguousarray(np.random.default_rng(0).random((30, 3)))
    R = 1.0
    fp = oin.vr_filtration(pts, max_dim=2, max_diameter=R, packed=True)
    dp = _reduce(fp)

    f = dion.fill_rips(pts.astype("f4"), 2, float(R))
    m = dion.homology_persistence(f)
    dion_dgms = dion.init_diagrams(m, f)

    for d in (0, 1):
        oin_fin = _sorted(dp.diagram(fp).in_dimension(d))
        oin_fin = oin_fin[np.isfinite(oin_fin).all(axis=1)].astype(np.float32).reshape(-1, 2)
        dion_fin = np.array([[q.birth, q.death] for q in dion_dgms[d] if q.death < np.inf],
                            dtype=np.float32).reshape(-1, 2)
        bd = dion.bottleneck_distance(dion.Diagram(oin_fin), dion.Diagram(dion_fin))
        assert bd < 1e-5, f"dim {d}: bottleneck {bd}"


def test_packed_vr_pickle_round_trip():
    import pickle
    pts = np.ascontiguousarray(np.random.default_rng(3).random((20, 3)))
    fp = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=True)
    assert pickle.loads(pickle.dumps(fp)) == fp


def test_packed_vr_tier_selection():
    # 64-bit tier for small clouds; 128-bit when a (dim+1)*bits field-set exceeds 64;
    # None (fat fallback) when it exceeds 128.
    assert oin._vr_packed_word_suffix(30, 2) == "64"
    assert oin._vr_packed_word_suffix(1100, 5) == "128"      # 11 bits * 6 = 66
    assert oin._vr_packed_word_suffix(2, 100) == "128"       # 1 bit * 101 = 101
    assert oin._vr_packed_word_suffix(1 << 20, 20) is None   # 20 bits * 21 = 420 > 128


def test_packed_vr_128_tier_matches_fat():
    # Force the 128-bit tier with a tiny complex: 1100 points (11-bit ids) at max_dim 5
    # needs 6*11 = 66 > 64 bits. A tight 6-point cluster forms a 5-simplex; the other
    # 1094 points sit far apart and isolated under the diameter, keeping the complex small.
    rng = np.random.default_rng(7)
    cluster = rng.random((6, 3)) * 0.01
    far = np.stack([np.arange(1094) * 100.0 + 100.0,
                    np.zeros(1094), np.zeros(1094)], axis=1)
    pts = np.ascontiguousarray(np.vstack([cluster, far]))
    assert oin._vr_packed_word_suffix(pts.shape[0], 5) == "128"

    R = 0.5
    fp = oin.vr_filtration(pts, max_dim=5, max_diameter=R, packed=True)
    assert type(fp).__name__ == "PackedSimplexFiltration_128"
    ff = oin.vr_filtration(pts, max_dim=5, max_diameter=R, packed=False)
    assert fp.size() == ff.size()
    _dgms_equal(_reduce(fp), fp, _reduce(ff), ff, 5)

    # exercise the 128-tier __setstate__ (slim_simplex_from_packed<unsigned __int128>
    # + rebuild_uid_index_ over __int128 hash keys), distinct from the 64-tier pickle
    import pickle
    assert pickle.loads(pickle.dumps(fp)) == fp


def test_packed_vr_kicr_rejected():
    # KICR is not wired for the bit-packed cell; reject clearly (the materialized fat
    # cell would otherwise slip past the Simplex-isinstance dispatch into a wrong ctor).
    pts = np.ascontiguousarray(np.random.default_rng(4).random((15, 3)))
    K = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=True)
    L = K.without_cells([K.size() - 1])
    with pytest.raises(NotImplementedError):
        oin.compute_kernel_image_cokernel_reduction(K, L)
