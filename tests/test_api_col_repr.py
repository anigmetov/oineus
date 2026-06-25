import numpy as np
import pytest
import oineus as oin

# The cached working-column representation used during reduction (PHAT A-Set /
# A-Heap / A-Full / A-Bit-Tree). It is a pure implementation detail of the working
# matrix and must NEVER change the resulting persistence diagram, regardless of
# dualize / threads / compute_v. This pins that invariant -- col_repr was
# configurable + pickled but previously cross-validated by nothing.
ALL_REPRS = [oin.ColumnRepr.Set, oin.ColumnRepr.Heap, oin.ColumnRepr.Full, oin.ColumnRepr.BitTree]


def _grid_fil(n=24, seed=0):
    rng = np.random.default_rng(seed)
    return oin.freudenthal_filtration(data=np.ascontiguousarray(rng.random((n, n))))


def _vr_fil(seed=0):
    pts = np.ascontiguousarray(np.random.default_rng(seed).random((25, 3)))
    return oin.vr_filtration(pts, max_dim=2, max_diameter=1.0, n_threads=1)


def _dgms(fil, col_repr, dualize, n_threads, compute_v, ndims):
    p = oin.ReductionParams()
    p.col_repr = col_repr
    p.compute_v = compute_v
    p.n_threads = n_threads
    dcmp = oin.Decomposition(fil, dualize)
    dcmp.reduce(p)
    return [dcmp.diagram(fil).in_dimension(d) for d in range(ndims)]


def _sorted(dgm):
    return dgm[np.lexsort(dgm.T)] if len(dgm) else dgm


def _dgms_equal(a, b, ndims):
    for d in range(ndims):
        x, y = _sorted(a[d]), _sorted(b[d])
        # np.allclose treats inf == inf as True (essential points have inf death)
        if x.shape != y.shape or not np.allclose(x, y):
            return False
    return True


@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("compute_v", [False, True])
def test_col_repr_invariant_grid(dualize, n_threads, compute_v):
    fil = _grid_fil()
    ndims = 3
    base = _dgms(fil, oin.ColumnRepr.BitTree, dualize, n_threads, compute_v, ndims)
    for cr in ALL_REPRS:
        got = _dgms(fil, cr, dualize, n_threads, compute_v, ndims)
        assert _dgms_equal(got, base, ndims), \
            f"col_repr={cr} dualize={dualize} n_threads={n_threads} compute_v={compute_v}"


@pytest.mark.parametrize("dualize", [False, True])
@pytest.mark.parametrize("n_threads", [1, 4])
def test_col_repr_invariant_vr(dualize, n_threads):
    fil = _vr_fil()
    ndims = 3
    base = _dgms(fil, oin.ColumnRepr.BitTree, dualize, n_threads, True, ndims)
    for cr in ALL_REPRS:
        got = _dgms(fil, cr, dualize, n_threads, True, ndims)
        assert _dgms_equal(got, base, ndims), \
            f"col_repr={cr} dualize={dualize} n_threads={n_threads}"
