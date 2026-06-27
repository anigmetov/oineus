import numpy as np
import pytest

import oineus as oin

# alpha-shapes need diode (CGAL); skip the whole module if it is absent
pytest.importorskip("diode")

# the packed default for alpha only kicks in on the diode-array fast path
requires_arrays = pytest.mark.skipif(
    not getattr(oin, "_HAS_DIODE_ARRAYS", False),
    reason="requires a diode build with the array exporters",
)


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


@pytest.mark.parametrize("ambient", [2, 3])
def test_packed_alpha_matches_fat(ambient):
    pts = np.ascontiguousarray(np.random.default_rng(0).random((40, ambient)))
    ff = oin.alpha_filtration(pts, packed=False)
    fp = oin.alpha_filtration(pts, packed=True)
    assert type(fp).__name__ == "_PackedSimplexFiltration_64"
    assert fp.size() == ff.size()
    _dgms_equal(_reduce(fp), fp, _reduce(ff), ff, ambient)


def test_packed_alpha_matches_gudhi():
    gudhi = pytest.importorskip("gudhi")
    pts = np.ascontiguousarray(np.random.default_rng(2).random((50, 2)))
    fp = oin.alpha_filtration(pts, packed=True)
    dp = _reduce(fp)

    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()  # gudhi alpha values are squared circumradii, like diode
    st.compute_persistence(homology_coeff_field=2, min_persistence=0.0)

    for d in (0, 1):
        oin_fin = _sorted(dp.diagram(fp).in_dimension(d))
        oin_fin = oin_fin[np.isfinite(oin_fin).all(axis=1)]
        g = np.asarray(st.persistence_intervals_in_dimension(d), dtype=np.float64).reshape(-1, 2)
        g = _sorted(g[np.isfinite(g).all(axis=1)])
        assert oin_fin.shape == g.shape, f"dim {d}: {oin_fin.shape} vs {g.shape}"
        assert np.allclose(oin_fin, g, atol=1e-5), f"dim {d}"


def test_packed_alpha_pickle_round_trip():
    import pickle
    pts = np.ascontiguousarray(np.random.default_rng(3).random((30, 3)))
    fp = oin.alpha_filtration(pts, packed=True)
    assert pickle.loads(pickle.dumps(fp)) == fp


def test_packed_alpha_uid_accessors_round_trip():
    # alpha is a distinct entry point but shares the packed filtration type; the uid
    # accessors must round-trip the combinatorial uid of each materialized cell
    pts = np.ascontiguousarray(np.random.default_rng(4).random((35, 3)))
    fp = oin.alpha_filtration(pts, packed=True)
    for i in range(fp.size()):
        c = fp.cell(i)
        assert fp.sorted_id_by_uid(c.uid) == i
        assert fp.value_by_uid(c.uid) == c.value
        assert list(fp.cell_by_uid(c.uid).vertices) == list(c.vertices)


@requires_arrays
def test_alpha_default_is_packed():
    # the default (no packed= kwarg) is now the bit-packed path on the fast array path
    pts = np.ascontiguousarray(np.random.default_rng(0).random((40, 3)))
    assert type(oin.alpha_filtration(pts)).__name__ == "_PackedSimplexFiltration_64"
    # escape hatch yields the fat universal Filtration
    assert type(oin.alpha_filtration(pts, packed=False)).__name__ == "_Filtration"


def test_alpha_packed_silently_downgrades_on_weighted():
    # only the unweighted/non-periodic array path packs; weighted alpha stays fat even
    # with packed=True (documented behavior -- pin it so a future change that tries to
    # pack the weighted path and breaks is caught)
    pts = np.ascontiguousarray(np.random.default_rng(0).random((20, 3)))
    w = np.ascontiguousarray(np.random.default_rng(1).random(20) * 0.01)
    fil = oin.alpha_filtration(pts, weights=w, packed=True)
    assert type(fil).__name__ == "_Filtration"


def test_alpha_packed_silently_downgrades_on_list_fallback(monkeypatch):
    # without the diode array exporters the list path is taken, which is always fat
    monkeypatch.setattr(oin, "_HAS_DIODE_ARRAYS", False)
    pts = np.ascontiguousarray(np.random.default_rng(0).random((20, 3)))
    fil = oin.alpha_filtration(pts, packed=True)
    assert type(fil).__name__ == "_Filtration"


def test_alpha_tier_selection():
    # a large vertex count pushes the packed tier from 64 to 128 bits. For max_dim 3 the
    # field-set is 4*bits; bits = ceil(log2(n)), so n ~ 40000 -> 16 bits -> width 64 (still
    # the 64 tier), while n ~ 70000 -> 17 bits -> width 68 (the 128 tier).
    assert oin._vr_packed_word_suffix(40, 3) == "64"
    assert oin._vr_packed_word_suffix(40000, 3) == "64"
    assert oin._vr_packed_word_suffix(70000, 3) == "128"
