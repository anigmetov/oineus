"""Tests for the diode array-exporter fast paths.

Newer diode builds expose fill_delaunay_arrays (combinatorics only) and
fill_alpha_shapes_arrays (combinatorics + alpha values) as per-dimension
NumPy arrays. Oineus consumes them via the _oineus._filtration_from_arrays
factory:

- non-differentiable alpha_filtration uses fill_alpha_shapes_arrays,
- differentiable cech_delaunay/weak_alpha use fill_delaunay_arrays through
  the internal _delaunay_combinatorics helper.

These tests check that the fast paths produce results identical to the
list-of-(vertices, value) fallback. The fast paths are gated on
oin._HAS_DIODE_ARRAYS; tests that specifically exercise them skip when an
older diode is installed.
"""
import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import diode
    HAS_DIODE = True
except ImportError:
    HAS_DIODE = False

import oineus as oin


pytestmark = pytest.mark.skipif(not HAS_DIODE, reason="requires diode")

HAS_DIODE_ARRAYS = getattr(oin, "_HAS_DIODE_ARRAYS", False)
requires_arrays = pytest.mark.skipif(
    not HAS_DIODE_ARRAYS,
    reason="requires a diode build with the array exporters",
)
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")


UNIT_TET = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


def _simplex_set(fil):
    """Set of simplices in a filtration as sorted vertex-index tuples."""
    out = set()
    for d in range(fil.max_dim + 1):
        for row in np.asarray(fil.get_simplices_as_arr(d)):
            out.add(tuple(sorted(int(v) for v in row)))
    return out


def _per_dim_diagrams(fil, maxdim=2, include_inf=True):
    """Reduce fil and return per-dimension diagrams, sorted lexicographically."""
    dcmp = oin.Decomposition(fil, False)
    dcmp.reduce(oin.ReductionParams())
    diag = dcmp.diagram(fil, include_inf_points=include_inf)
    res = []
    for d in range(maxdim + 1):
        a = np.asarray(diag.in_dimension(d), dtype=float)
        if a.size == 0:
            res.append(a.reshape(0, 2))
        else:
            res.append(a[np.lexsort((a[:, 1], a[:, 0]))])
    return res


def _assert_diagrams_equal(da, db, atol=1e-9):
    assert len(da) == len(db)
    for d in range(len(da)):
        a, b = da[d], db[d]
        assert a.shape == b.shape, f"dim {d}: shape {a.shape} vs {b.shape}"
        if a.size:
            np.testing.assert_allclose(a, b, atol=atol)


# ---------------------------------------------------------------------------
# _filtration_from_arrays factory
# ---------------------------------------------------------------------------

@requires_arrays
def test_filtration_from_arrays_unit_tet_shapes_and_placeholder_values():
    verts_by_dim = diode.fill_delaunay_arrays(UNIT_TET)
    fil = oin._oineus._filtration_from_arrays(verts_by_dim, None, n_threads=1)
    # full Delaunay of a tetrahedron: 4 verts, 6 edges, 4 triangles, 1 cell
    assert [fil.size_in_dimension(d) for d in range(4)] == [4, 6, 4, 1]
    # no values supplied -> all placeholders (0)
    assert all(fil.cell(i).value == 0.0 for i in range(fil.size()))


@requires_arrays
def test_filtration_from_arrays_rejects_wrong_width():
    # a (n, 2) array placed at dim 0 (expects width 1) must be rejected
    bad = [np.array([[0, 1]], dtype=np.int64)]
    with pytest.raises(Exception):
        oin._oineus._filtration_from_arrays(bad, None, n_threads=1)


@requires_arrays
def test_filtration_from_arrays_length_mismatch_rejected():
    verts_by_dim = diode.fill_delaunay_arrays(UNIT_TET)
    vals = [np.zeros(arr.shape[0] + 1, dtype=np.float64) for arr in verts_by_dim]
    with pytest.raises(Exception):
        oin._oineus._filtration_from_arrays(verts_by_dim, vals, n_threads=1)


# ---------------------------------------------------------------------------
# combinatorics agreement
# ---------------------------------------------------------------------------

@requires_arrays
@pytest.mark.parametrize("dim", [2, 3])
def test_delaunay_combinatorics_matches_alpha(dim):
    rng = np.random.default_rng(7)
    pts = rng.random((50, dim))
    fast = oin._delaunay_combinatorics(pts)
    alpha = oin.alpha_filtration(pts)
    assert _simplex_set(fast) == _simplex_set(alpha)


# ---------------------------------------------------------------------------
# non-differentiable alpha_filtration: arrays vs tuples
# ---------------------------------------------------------------------------

@requires_arrays
@pytest.mark.parametrize("dim", [2, 3])
def test_alpha_filtration_arrays_match_tuples(monkeypatch, dim):
    rng = np.random.default_rng(11)
    pts = rng.random((60, dim))

    fil_fast = oin.alpha_filtration(pts)

    monkeypatch.setattr(oin, "_HAS_DIODE_ARRAYS", False)
    fil_slow = oin.alpha_filtration(pts)

    assert len(fil_fast) == len(fil_slow)
    _assert_diagrams_equal(
        _per_dim_diagrams(fil_fast, maxdim=dim - 1),
        _per_dim_diagrams(fil_slow, maxdim=dim - 1),
    )


# ---------------------------------------------------------------------------
# differentiable paths: fast (fill_delaunay_arrays) vs fallback
# ---------------------------------------------------------------------------

@requires_arrays
@requires_torch
@pytest.mark.parametrize("dim", [2, 3])
def test_cech_delaunay_fast_vs_fallback(monkeypatch, dim):
    from oineus.diff import cech_delaunay_filtration
    rng = np.random.default_rng(13)
    pts = rng.random((55, dim))

    df_fast = cech_delaunay_filtration(torch.tensor(pts, dtype=torch.float64, requires_grad=True))
    assert df_fast.values.requires_grad

    monkeypatch.setattr(oin, "_HAS_DIODE_ARRAYS", False)
    df_slow = cech_delaunay_filtration(torch.tensor(pts, dtype=torch.float64, requires_grad=True))

    assert len(df_fast.under_fil) == len(df_slow.under_fil)
    _assert_diagrams_equal(
        _per_dim_diagrams(df_fast.under_fil, maxdim=dim - 1),
        _per_dim_diagrams(df_slow.under_fil, maxdim=dim - 1),
    )


@requires_arrays
@requires_torch
@pytest.mark.parametrize("dim", [2, 3])
def test_weak_alpha_fast_vs_fallback(monkeypatch, dim):
    from oineus.diff import weak_alpha_filtration
    rng = np.random.default_rng(17)
    pts = rng.random((55, dim))

    df_fast = weak_alpha_filtration(torch.tensor(pts, dtype=torch.float64, requires_grad=True))
    assert df_fast.values.requires_grad

    monkeypatch.setattr(oin, "_HAS_DIODE_ARRAYS", False)
    df_slow = weak_alpha_filtration(torch.tensor(pts, dtype=torch.float64, requires_grad=True))

    assert len(df_fast.under_fil) == len(df_slow.under_fil)
    _assert_diagrams_equal(
        _per_dim_diagrams(df_fast.under_fil, maxdim=dim - 1),
        _per_dim_diagrams(df_slow.under_fil, maxdim=dim - 1),
    )


@requires_arrays
@requires_torch
def test_cech_delaunay_gradient_flows():
    from oineus.diff import cech_delaunay_filtration
    rng = np.random.default_rng(19)
    pts = torch.tensor(rng.random((30, 3)), dtype=torch.float64, requires_grad=True)
    df = cech_delaunay_filtration(pts)
    loss = df.values.sum()
    loss.backward()
    assert pts.grad is not None
    assert torch.isfinite(pts.grad).all()
    assert pts.grad.abs().sum().item() > 0.0
