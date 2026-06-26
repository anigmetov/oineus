import numpy as np
import pytest

import oineus as oin


def _sorted(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def _kicr_dim(dgms, d):
    # KICR diagram families span fewer dimensions than the complex max_dim, so
    # in_dimension(d) raises IndexError past the family's top dim -- treat as empty.
    try:
        a = _sorted(dgms.in_dimension(d))
    except IndexError:
        a = np.empty((0, 2), dtype=np.float64)
    a[a == np.inf] = 1e9
    return a


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
    assert type(fp).__name__ == "_PackedSimplexFiltration_64"
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
    assert type(fp).__name__ == "_PackedSimplexFiltration_64"
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


def _check_uid_round_trip(fil):
    # The Python-facing uid is the universal COMBINATORIAL uid a materialized fat cell
    # carries; the packed filtration's uid accessors must accept it and re-pack it into
    # the internal Word uid. Every cell must round-trip through value/sorted_id/cell_by_uid.
    for i in range(fil.size()):
        c = fil.cell(i)
        assert fil.sorted_id_by_uid(c.uid) == i
        assert fil.value_by_uid(c.uid) == c.value
        assert list(fil.cell_by_uid(c.uid).vertices) == list(c.vertices)


def test_packed_vr_uid_accessors_round_trip():
    pts = np.ascontiguousarray(np.random.default_rng(2).random((22, 3)))
    fp = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=True)
    assert type(fp).__name__ == "_PackedSimplexFiltration_64"
    _check_uid_round_trip(fp)

    # a vertex-set uid not present in the filtration must raise (not silently 0)
    bogus = oin.vr_filtration(np.ascontiguousarray(np.random.default_rng(8).random((40, 3))),
                              max_dim=2, max_diameter=1.0, packed=True)
    missing = max(bogus.cell(i).uid for i in range(bogus.size()))
    if all(fp.cell(i).uid != missing for i in range(fp.size())):
        with pytest.raises((IndexError, KeyError)):
            fp.sorted_id_by_uid(missing)


def test_packed_vr_uid_foreign_vertex_not_misidentified():
    # A foreign uid whose vertex id exceeds the target field width must report "not
    # present", never silently alias a different present cell. With bits=5 (22 points),
    # the absent 0-cell {32} bit-spills to the same packed word as the present edge
    # {0,1} (pack({32}) == pack({0,1}) == 32); the accessor must raise, not return {0,1}.
    fat = oin.vr_filtration(np.ascontiguousarray(np.random.default_rng(0).random((40, 3))),
                            max_dim=0, max_diameter=10.0, packed=False)
    uid32 = next(fat.cell(i).uid for i in range(fat.size())
                 if list(fat.cell(i).vertices) == [32])

    # tight cluster so every pairwise edge (incl. {0,1}) is present in the packed filtration
    tight = np.ascontiguousarray(np.random.default_rng(1).random((22, 3)) * 0.001)
    fp = oin.vr_filtration(tight, max_dim=2, max_diameter=10.0, packed=True)
    assert type(fp).__name__ == "_PackedSimplexFiltration_64"
    assert all(fp.cell(i).uid != uid32 for i in range(fp.size()))  # {32} genuinely absent
    with pytest.raises((IndexError, KeyError)):
        fp.sorted_id_by_uid(uid32)
    with pytest.raises((IndexError, KeyError)):
        fp.cell_by_uid(uid32)


def test_packed_vr_uid_accessors_round_trip_128_tier():
    # the 128-bit tier exercises the unsigned __int128 Word re-pack + __int128 hash lookup
    rng = np.random.default_rng(7)
    cluster = rng.random((6, 3)) * 0.01
    far = np.stack([np.arange(1094) * 100.0 + 100.0,
                    np.zeros(1094), np.zeros(1094)], axis=1)
    pts = np.ascontiguousarray(np.vstack([cluster, far]))
    fp = oin.vr_filtration(pts, max_dim=5, max_diameter=0.5, packed=True)
    assert type(fp).__name__ == "_PackedSimplexFiltration_128"
    _check_uid_round_trip(fp)


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
    assert type(fp).__name__ == "_PackedSimplexFiltration_128"
    ff = oin.vr_filtration(pts, max_dim=5, max_diameter=R, packed=False)
    assert fp.size() == ff.size()
    _dgms_equal(_reduce(fp), fp, _reduce(ff), ff, 5)

    # exercise the 128-tier __setstate__ (slim_simplex_from_packed<unsigned __int128>
    # + rebuild_uid_index_ over __int128 hash keys), distinct from the 64-tier pickle
    import pickle
    assert pickle.loads(pickle.dumps(fp)) == fp


def test_bare_topology_optimizer_dispatches_packed():
    # the bare oin.TopologyOptimizer must dispatch to the bit-packed C++ optimizer and
    # produce the same diagram as the fat path
    pts = np.ascontiguousarray(np.random.default_rng(6).random((22, 3)))
    fp = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=True)
    opt = oin.TopologyOptimizer(fp)
    assert type(opt).__name__ == "TopologyOptimizerPacked_64"
    opt.reduce_all()
    dp = opt.compute_diagram(include_inf_points=False)

    ff = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=False)
    opt_f = oin.TopologyOptimizer(ff)
    assert type(opt_f).__name__ == "TopologyOptimizer"
    opt_f.reduce_all()
    df = opt_f.compute_diagram(include_inf_points=False)
    for d in range(3):
        a = _sorted(dp.in_dimension(d))
        b = _sorted(df.in_dimension(d))
        a[a == np.inf] = 1e9
        b[b == np.inf] = 1e9
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-6), f"bare opt packed != fat dim {d}"


def test_packed_vr_kicr_matches_fat():
    # KICR is wired for the bit-packed cell (dispatch on the filtration type to
    # _KerImCokReduced_Packed_64). A packed VR K/L pair must give the same
    # kernel/image/cokernel/(co)domain diagrams as the fat path.
    pts = np.ascontiguousarray(np.random.default_rng(4).random((15, 3)))

    def run(packed):
        K = oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=packed)
        L = K.without_cells([K.size() - 1])
        params = oin.KICRParams()
        params.codomain = params.kernel = params.image = params.cokernel = True
        return oin.compute_kernel_image_cokernel_reduction(K, L, params)

    kp, kf = run(True), run(False)
    assert type(kp).__name__ == "_KerImCokReduced_Packed_64"
    for fam in ("kernel_diagrams", "cokernel_diagrams", "image_diagrams",
                "domain_diagrams", "codomain_diagrams"):
        for d in range(3):
            a = _kicr_dim(getattr(kp, fam)(), d)
            b = _kicr_dim(getattr(kf, fam)(), d)
            assert a.shape == b.shape and (a.size == 0 or np.allclose(a, b, atol=1e-6)), f"{fam} dim {d}"


def test_packed_vr_kicr_128_tier_matches_fat():
    # KICR on the 128-bit packed tier (KerImCokReduced<BitPacked<__int128>>). Same tiny-complex
    # trick as test_packed_vr_128_tier_matches_fat: 1100 points (11-bit ids) at max_dim 5 needs
    # 6*11 = 66 > 64 bits, but only a tight 6-point cluster forms higher simplices, so the
    # complex (and the 5 KICR reductions) stay small. Exercises the __int128 KICR class + pickle.
    rng = np.random.default_rng(7)
    cluster = rng.random((6, 3)) * 0.01
    far = np.stack([np.arange(1094) * 100.0 + 100.0,
                    np.zeros(1094), np.zeros(1094)], axis=1)
    pts = np.ascontiguousarray(np.vstack([cluster, far]))
    assert oin._vr_packed_word_suffix(pts.shape[0], 5) == "128"

    def run(packed):
        K = oin.vr_filtration(pts, max_dim=5, max_diameter=0.5, packed=packed)
        L = K.without_cells([K.size() - 1])
        params = oin.KICRParams()
        params.codomain = params.kernel = params.image = params.cokernel = True
        return oin.compute_kernel_image_cokernel_reduction(K, L, params)

    kp, kf = run(True), run(False)
    assert type(kp).__name__ == "_KerImCokReduced_Packed_128"
    for fam in ("kernel_diagrams", "cokernel_diagrams", "image_diagrams",
                "domain_diagrams", "codomain_diagrams"):
        for d in range(6):
            a = _kicr_dim(getattr(kp, fam)(), d)
            b = _kicr_dim(getattr(kf, fam)(), d)
            assert a.shape == b.shape and (a.size == 0 or np.allclose(a, b, atol=1e-6)), f"{fam} dim {d}"

    import pickle
    assert pickle.loads(pickle.dumps(kp)) == kp


def test_vr_default_is_packed():
    # the default (no packed= kwarg) is now the bit-packed path when the tier fits
    pts = np.ascontiguousarray(np.random.default_rng(0).random((15, 3)))
    assert type(oin.vr_filtration(pts, max_dim=2, max_diameter=1.0)).__name__ == "_PackedSimplexFiltration_64"
    # escape hatch yields the fat universal Filtration
    assert type(oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, packed=False)).__name__ == "_Filtration"
