import numpy as np
import pytest

import oineus as oin

# alpha-shapes need diode (CGAL); skip the whole module if it is absent
pytest.importorskip("diode")


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
    assert type(fp).__name__ == "PackedSimplexFiltration_64"
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
